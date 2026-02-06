# -*- coding: utf-8 -*-
"""
Inspection Artifacts Uploader — Envia artefatos de inspeção aprovada para a VPS.

Segue o mesmo padrão do ArtifactsUploader (PR13/Etapa 4):
- RunPod NÃO acessa MinIO diretamente
- Envia via HTTP POST multipart para a VPS
- VPS recebe e grava no MinIO local (127.0.0.1:9100)

Endpoint VPS: POST /api/v1/inspect/artifacts

Artefatos enviados:
- metadata.json        (InspectionMetadata)
- canonical.md         (texto canônico reconciliado)
- offsets.json         (mapa span_id → offsets no canonical)
- pymupdf_result.json  (artefato fase 1)
- vlm_result.json      (artefato fase 2)
- reconciliation_result.json (artefato fase 3)
- integrity_result.json (artefato fase 4)
- chunks_preview.json  (artefato fase 5)
- page images PNG      (páginas anotadas)

Configuração via env vars (mesmas do ArtifactsUploader):
- ARTIFACTS_BASE_URL: URL base da VPS (ex: https://vectorgov.io)
- VPS_API_KEY: API key para autenticação
- CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET: Cloudflare Access (opcional)
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class InspectionUploadResult:
    """Resultado do upload de artefatos de inspeção."""
    success: bool
    document_id: str
    message: str
    artifacts_persisted: list[str] | None = None
    error: str | None = None
    retries: int = 0


class InspectionUploader:
    """
    Cliente para upload de artefatos de inspeção para a VPS.

    Implementa retry com backoff exponencial e validação de resposta.
    Segue o mesmo padrão do ArtifactsUploader existente.
    """

    DEFAULT_ENDPOINT = "/api/v1/inspect/artifacts"
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 0.5
    TIMEOUT = 120  # segundos (artefatos podem ser grandes)

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        cf_client_id: Optional[str] = None,
        cf_client_secret: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("ARTIFACTS_BASE_URL", "")
        self.endpoint = endpoint or os.getenv(
            "INSPECTION_ARTIFACTS_ENDPOINT", self.DEFAULT_ENDPOINT
        )
        self.api_key = api_key or os.getenv("VPS_API_KEY", "")
        self.cf_client_id = cf_client_id or os.getenv("CF_ACCESS_CLIENT_ID", "")
        self.cf_client_secret = cf_client_secret or os.getenv("CF_ACCESS_CLIENT_SECRET", "")

        self.session = self._create_session()

        logger.info(
            "InspectionUploader inicializado: base_url=%s, endpoint=%s, cf_access=%s",
            self.base_url, self.endpoint,
            "enabled" if self.cf_client_id else "disabled",
        )

    def _create_session(self) -> requests.Session:
        """Cria session HTTP com retry configurado."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _build_headers(self) -> dict[str, str]:
        """Constrói headers de autenticação."""
        headers = {}
        if self.api_key:
            headers["X-Ingest-Key"] = self.api_key
        if self.cf_client_id and self.cf_client_secret:
            headers["CF-Access-Client-Id"] = self.cf_client_id
            headers["CF-Access-Client-Secret"] = self.cf_client_secret
        return headers

    def is_configured(self) -> bool:
        """Verifica se o uploader está configurado."""
        return bool(self.base_url)

    def upload(
        self,
        document_id: str,
        metadata_json: str,
        canonical_md: Optional[str] = None,
        offsets_json: Optional[str] = None,
        pymupdf_result_json: Optional[str] = None,
        vlm_result_json: Optional[str] = None,
        reconciliation_result_json: Optional[str] = None,
        integrity_result_json: Optional[str] = None,
        chunks_preview_json: Optional[str] = None,
        page_images: Optional[dict[str, bytes]] = None,
    ) -> InspectionUploadResult:
        """
        Faz upload de todos os artefatos de inspeção para a VPS.

        Args:
            document_id: ID do documento (ex: LEI-14133-2021)
            metadata_json: JSON string do InspectionMetadata
            canonical_md: Texto canônico reconciliado
            offsets_json: JSON string do mapa de offsets
            pymupdf_result_json: JSON string do PyMuPDFArtifact
            vlm_result_json: JSON string do VLMArtifact
            reconciliation_result_json: JSON string do ReconciliationArtifact
            integrity_result_json: JSON string do IntegrityArtifact
            chunks_preview_json: JSON string do ChunksPreviewArtifact
            page_images: dict[page_key, png_bytes] (ex: {"page_001": b"..."})

        Returns:
            InspectionUploadResult com status do upload
        """
        if not self.is_configured():
            logger.warning("InspectionUploader não configurado (ARTIFACTS_BASE_URL vazio)")
            return InspectionUploadResult(
                success=False,
                document_id=document_id,
                message="Uploader não configurado",
                error="ARTIFACTS_BASE_URL não definido",
            )

        url = f"{self.base_url.rstrip('/')}{self.endpoint}"

        # Monta multipart: files + form data
        files = []
        form_data = {"document_id": document_id}

        # Metadata (obrigatório)
        files.append((
            "metadata_file",
            ("metadata.json", metadata_json.encode("utf-8"), "application/json"),
        ))

        # Artefatos opcionais (JSON)
        _optional_files = [
            ("canonical_md_file", "canonical.md", canonical_md, "text/markdown"),
            ("offsets_json_file", "offsets.json", offsets_json, "application/json"),
            ("pymupdf_file", "pymupdf_result.json", pymupdf_result_json, "application/json"),
            ("vlm_file", "vlm_result.json", vlm_result_json, "application/json"),
            ("reconciliation_file", "reconciliation_result.json", reconciliation_result_json, "application/json"),
            ("integrity_file", "integrity_result.json", integrity_result_json, "application/json"),
            ("chunks_file", "chunks_preview.json", chunks_preview_json, "application/json"),
        ]

        for field_name, filename, content, content_type in _optional_files:
            if content:
                data = content.encode("utf-8") if isinstance(content, str) else content
                files.append((field_name, (filename, data, content_type)))

        # Imagens de páginas (PNG)
        if page_images:
            for page_key, png_bytes in page_images.items():
                files.append((
                    "page_images",
                    (f"{page_key}.png", png_bytes, "image/png"),
                ))

        artifact_count = len(files)
        total_bytes = sum(
            len(f[1][1]) if isinstance(f[1], tuple) else 0
            for f in files
        )

        logger.info(
            "Uploading %d artefatos de inspeção para %s: "
            "document_id=%s, total=%d bytes",
            artifact_count, url, document_id, total_bytes,
        )

        headers = self._build_headers()
        retries = 0
        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self.session.post(
                    url,
                    files=files,
                    data=form_data,
                    headers=headers,
                    timeout=self.TIMEOUT,
                )

                if response.status_code in (200, 201):
                    result_data = response.json()
                    persisted = result_data.get("artifacts_persisted", [])
                    logger.info(
                        "Inspeção uploaded: document_id=%s, %d artefatos persistidos",
                        document_id, len(persisted),
                    )
                    return InspectionUploadResult(
                        success=True,
                        document_id=document_id,
                        message="Artefatos de inspeção enviados com sucesso",
                        artifacts_persisted=persisted,
                        retries=retries,
                    )
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
                    logger.warning(
                        "Upload inspeção falhou (attempt %d): %s",
                        attempt + 1, error_msg,
                    )
                    last_error = error_msg
                    retries += 1

                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        break

                    if attempt < self.MAX_RETRIES:
                        time.sleep(self.BACKOFF_FACTOR * (2 ** attempt))

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(
                    "Upload inspeção exception (attempt %d): %s",
                    attempt + 1, e,
                )
                retries += 1
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.BACKOFF_FACTOR * (2 ** attempt))

        logger.error(
            "Upload de inspeção falhou após %d tentativas: document_id=%s, error=%s",
            retries, document_id, last_error,
        )

        return InspectionUploadResult(
            success=False,
            document_id=document_id,
            message=f"Upload falhou após {retries} tentativas",
            error=last_error,
            retries=retries,
        )


# Singleton
_uploader: Optional[InspectionUploader] = None


def get_inspection_uploader() -> InspectionUploader:
    """Retorna instância singleton do uploader de inspeção."""
    global _uploader
    if _uploader is None:
        _uploader = InspectionUploader()
    return _uploader
