# -*- coding: utf-8 -*-
"""
Drift Detector — Detecta deriva do canonical_hash para mesmo pdf_hash + pipeline_version.

Invariante:
    sha256(pdf_bytes) + pipeline_version → canonical_hash DETERMINÍSTICO

O canonical_text vem do PyMuPDF (determinístico por versão). Se canonical_hash
muda para mesmo pdf_hash + pipeline_version, houve deriva (atualização de
PyMuPDF, bug, etc.).

Backends:
- JSON local (/tmp/drift_registry.json) — dev/testing, volátil em RunPod
- Redis (localhost:6379) — prod, persiste entre pod restarts
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DRIFT_REGISTRY_PATH = "/tmp/drift_registry.json"
DRIFT_REDIS_PREFIX = "drift:"
DRIFT_REDIS_TTL = 60 * 60 * 24 * 30  # 30 days


@dataclass
class DriftCheckResult:
    """Resultado de uma verificação de drift."""
    has_previous_run: bool
    is_drifted: bool
    previous_canonical_hash: str = ""
    current_canonical_hash: str = ""
    previous_run_id: str = ""
    message: str = ""


class DriftDetector:
    """
    Detecta deriva do canonical_hash.

    Chave: f"{pdf_hash}:{pipeline_version}"
    Valor: {canonical_hash, ingest_run_id, timestamp}

    Backends:
    - redis_client: se fornecido, usa Redis (prod)
    - registry_path: senão, usa JSON local (dev)
    """

    def __init__(
        self,
        registry_path: str = DRIFT_REGISTRY_PATH,
        redis_client=None,
    ):
        self._registry_path = registry_path
        self._redis = redis_client

    def _make_key(self, pdf_hash: str, pipeline_version: str) -> str:
        return f"{pdf_hash}:{pipeline_version}"

    # --- JSON backend ---

    def _load_registry(self) -> dict:
        """Carrega o registry JSON do disco."""
        if not os.path.exists(self._registry_path):
            return {}
        try:
            with open(self._registry_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Drift registry corrompido, recriando: {e}")
            return {}

    def _save_registry(self, registry: dict) -> None:
        """Salva o registry JSON no disco."""
        try:
            with open(self._registry_path, "w") as f:
                json.dump(registry, f, indent=2)
        except IOError as e:
            logger.warning(f"Falha ao salvar drift registry: {e}")

    # --- Redis backend ---

    def _redis_get(self, key: str) -> Optional[dict]:
        """Lê entry do Redis."""
        try:
            data = self._redis.get(f"{DRIFT_REDIS_PREFIX}{key}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Drift Redis read failed, falling back: {e}")
        return None

    def _redis_set(self, key: str, entry: dict) -> None:
        """Grava entry no Redis com TTL."""
        try:
            self._redis.setex(
                f"{DRIFT_REDIS_PREFIX}{key}",
                DRIFT_REDIS_TTL,
                json.dumps(entry),
            )
        except Exception as e:
            logger.warning(f"Drift Redis write failed: {e}")

    # --- Unified API ---

    def _get_entry(self, key: str) -> Optional[dict]:
        if self._redis:
            return self._redis_get(key)
        registry = self._load_registry()
        return registry.get(key)

    def _set_entry(self, key: str, entry: dict) -> None:
        if self._redis:
            self._redis_set(key, entry)
            return
        registry = self._load_registry()
        registry[key] = entry
        self._save_registry(registry)

    def check(
        self,
        document_id: str,
        pdf_hash: str,
        pipeline_version: str,
        current_canonical_hash: str,
    ) -> DriftCheckResult:
        """
        Verifica se houve drift para esta combinação pdf_hash + pipeline_version.

        Args:
            document_id: ID do documento (para logs)
            pdf_hash: SHA256 do PDF original
            pipeline_version: Versão do pipeline
            current_canonical_hash: Hash canônico atual

        Returns:
            DriftCheckResult com informações sobre drift
        """
        key = self._make_key(pdf_hash, pipeline_version)
        entry = self._get_entry(key)

        if not entry:
            return DriftCheckResult(
                has_previous_run=False,
                is_drifted=False,
                current_canonical_hash=current_canonical_hash,
                message=f"Primeiro run para {document_id} (pdf_hash={pdf_hash[:16]}...)",
            )

        previous_hash = entry.get("canonical_hash", "")
        previous_run_id = entry.get("ingest_run_id", "")

        if previous_hash == current_canonical_hash:
            return DriftCheckResult(
                has_previous_run=True,
                is_drifted=False,
                previous_canonical_hash=previous_hash,
                current_canonical_hash=current_canonical_hash,
                previous_run_id=previous_run_id,
                message=f"Determinístico: canonical_hash inalterado para {document_id}",
            )

        return DriftCheckResult(
            has_previous_run=True,
            is_drifted=True,
            previous_canonical_hash=previous_hash,
            current_canonical_hash=current_canonical_hash,
            previous_run_id=previous_run_id,
            message=(
                f"DRIFT: canonical_hash mudou para {document_id}! "
                f"previous={previous_hash[:16]}... "
                f"current={current_canonical_hash[:16]}... "
                f"(mesmo pdf_hash + pipeline_version)"
            ),
        )

    def register_run(
        self,
        document_id: str,
        pdf_hash: str,
        pipeline_version: str,
        canonical_hash: str,
        ingest_run_id: str,
    ) -> None:
        """
        Registra um run bem-sucedido.

        Args:
            document_id: ID do documento
            pdf_hash: SHA256 do PDF original
            pipeline_version: Versão do pipeline
            canonical_hash: Hash canônico resultante
            ingest_run_id: ID do run de ingestão
        """
        key = self._make_key(pdf_hash, pipeline_version)
        entry = {
            "document_id": document_id,
            "canonical_hash": canonical_hash,
            "ingest_run_id": ingest_run_id,
            "timestamp": time.time(),
        }
        self._set_entry(key, entry)
        logger.debug(
            f"Drift registry: registrado {document_id} "
            f"(key={key[:32]}..., hash={canonical_hash[:16]}...)"
        )
