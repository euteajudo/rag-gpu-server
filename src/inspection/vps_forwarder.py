"""
VPS Inspection Forwarder — envia artefatos de inspeção do RunPod para a VPS.

Padrão fire-and-forget: falhas no envio NÃO bloqueiam o pipeline de ingestão.
Apenas log warning é emitido.

Env vars:
    VPS_INSPECTION_URL: Base URL da VPS (ex: https://vectorgov.io)
    INSPECT_API_KEY: Chave machine-to-machine (header X-Inspect-Key)
    CF_ACCESS_CLIENT_ID: Cloudflare Access Client ID (segurança RunPod → VPS)
    CF_ACCESS_CLIENT_SECRET: Cloudflare Access Client Secret
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from .models import InspectionMetadata

logger = logging.getLogger(__name__)

# Timeout para requisições HTTP (segundos)
_HTTP_TIMEOUT = 30.0


class VpsInspectionForwarder:
    """
    Cliente HTTP fire-and-forget para enviar artefatos de inspeção à VPS.

    Se VPS_INSPECTION_URL ou INSPECT_API_KEY estiverem vazios,
    o forwarder fica desabilitado silenciosamente.
    """

    def __init__(self):
        self._base_url = os.getenv("VPS_INSPECTION_URL", "").rstrip("/")
        self._api_key = os.getenv("INSPECT_API_KEY", "")
        self._cf_client_id = os.getenv("CF_ACCESS_CLIENT_ID", "")
        self._cf_client_secret = os.getenv("CF_ACCESS_CLIENT_SECRET", "")
        self._enabled = bool(self._base_url and self._api_key)

        if not self._enabled:
            logger.info(
                "[VpsForwarder] Desabilitado — VPS_INSPECTION_URL ou INSPECT_API_KEY não configurados"
            )
        else:
            logger.info(
                f"[VpsForwarder] Habilitado — url={self._base_url}, "
                f"cf_access={'enabled' if self._cf_client_id else 'disabled'}"
            )

    def _build_headers(self) -> dict[str, str]:
        """Constrói headers de autenticação (Inspect Key + Cloudflare Access)."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Inspect-Key": self._api_key,
        }

        # Cloudflare Access headers (segurança adicional RunPod → VPS)
        if self._cf_client_id and self._cf_client_secret:
            headers["CF-Access-Client-Id"] = self._cf_client_id
            headers["CF-Access-Client-Secret"] = self._cf_client_secret

        return headers

    def forward_stage(
        self,
        task_id: str,
        stage: str,
        metadata: dict,
        artifact: dict,
    ) -> bool:
        """
        POST /api/v1/inspection/stages — envia um artefato de stage.

        Returns:
            True se enviado com sucesso, False caso contrário.
        """
        if not self._enabled:
            return False

        url = f"{self._base_url}/api/v1/inspection/stages"
        payload = {
            "task_id": task_id,
            "stage": stage,
            "metadata": metadata,
            "artifact": artifact,
        }

        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                resp = client.post(url, json=payload, headers=self._build_headers())

            if resp.status_code in (200, 201):
                logger.info(
                    f"[VpsForwarder] Stage '{stage}' enviado: task_id={task_id}"
                )
                return True
            else:
                logger.warning(
                    f"[VpsForwarder] Stage '{stage}' falhou: "
                    f"status={resp.status_code}, body={resp.text[:200]}"
                )
                return False

        except Exception as e:
            logger.warning(
                f"[VpsForwarder] Erro ao enviar stage '{stage}' (task_id={task_id}): {e}"
            )
            return False

    def complete_run(
        self,
        task_id: str,
        metadata: InspectionMetadata,
        canonical_hash: Optional[str] = None,
        canonical_length: Optional[int] = None,
    ) -> bool:
        """
        POST /api/v1/inspection/runs/{task_id}/complete — marca inspeção como completa.

        Returns:
            True se enviado com sucesso, False caso contrário.
        """
        if not self._enabled:
            return False

        url = f"{self._base_url}/api/v1/inspection/runs/{task_id}/complete"
        payload = {
            "status": metadata.status.value if metadata.status else "completed",
            "completed_at": metadata.completed_at or datetime.now(timezone.utc).isoformat(),
            "integrity_status": "passed",
        }
        if canonical_hash:
            payload["canonical_hash"] = canonical_hash
        if canonical_length is not None:
            payload["canonical_length"] = canonical_length

        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                resp = client.post(url, json=payload, headers=self._build_headers())

            if resp.status_code in (200, 201):
                logger.info(
                    f"[VpsForwarder] Run completado: task_id={task_id}"
                )
                return True
            else:
                logger.warning(
                    f"[VpsForwarder] Complete run falhou: "
                    f"status={resp.status_code}, body={resp.text[:200]}"
                )
                return False

        except Exception as e:
            logger.warning(
                f"[VpsForwarder] Erro ao completar run (task_id={task_id}): {e}"
            )
            return False

    def forward_full_snapshot(
        self,
        task_id: str,
        metadata: InspectionMetadata,
        stages: dict[str, str],
    ) -> None:
        """
        Envia metadata + todos os stages + complete_run de uma vez.

        Usado pelos métodos _emit_*_inspection_snapshot() do pipeline.

        Args:
            task_id: ID da task de ingestão
            metadata: InspectionMetadata do snapshot
            stages: Dict {stage_name: artifact_json_string}
        """
        if not self._enabled:
            return

        metadata_dict = metadata.model_dump(mode="json")

        # Extrair canonical_hash e canonical_length dos artefatos (se presentes)
        canonical_hash = None
        canonical_length = None

        for stage_name, artifact_json in stages.items():
            try:
                artifact_dict = json.loads(artifact_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"[VpsForwarder] Artifact JSON inválido para stage '{stage_name}'"
                )
                continue

            # Extrair hash do artefato regex_classification
            if stage_name == "regex_classification":
                canonical_hash = artifact_dict.get("canonical_hash")
                canonical_length = artifact_dict.get("canonical_length")

            self.forward_stage(
                task_id=task_id,
                stage=stage_name,
                metadata=metadata_dict,
                artifact=artifact_dict,
            )

        # Marca a run como completa
        self.complete_run(
            task_id=task_id,
            metadata=metadata,
            canonical_hash=canonical_hash,
            canonical_length=canonical_length,
        )
