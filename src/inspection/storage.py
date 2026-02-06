"""
Storage temporário para artefatos de inspeção (Redis).

Redis: armazenamento temporário durante revisão (TTL 2h, DB 2).

Persistência permanente (MinIO) é feita via HTTP POST para a VPS
pelo InspectionUploader (src/sinks/inspection_uploader.py),
chamado pelo ApprovalService.

Keys Redis:
    inspect:{task_id}:metadata   → InspectionMetadata (JSON)
    inspect:{task_id}:{stage}    → Artefato da fase (JSON)
    inspect:{task_id}:pdf        → PDF original (bytes)
"""

import json
import logging
import os
from typing import Optional

import redis

from .models import (
    InspectionMetadata,
    InspectionStage,
    PyMuPDFArtifact,
    VLMArtifact,
    ReconciliationArtifact,
    IntegrityArtifact,
    ChunksPreviewArtifact,
)

logger = logging.getLogger(__name__)

# TTL para artefatos temporários no Redis (2 horas)
REDIS_TTL_SECONDS = 2 * 60 * 60

# Mapa de stage → modelo Pydantic para desserialização
_STAGE_MODELS = {
    InspectionStage.PYMUPDF: PyMuPDFArtifact,
    InspectionStage.VLM: VLMArtifact,
    InspectionStage.RECONCILIATION: ReconciliationArtifact,
    InspectionStage.INTEGRITY: IntegrityArtifact,
    InspectionStage.CHUNKS: ChunksPreviewArtifact,
}


class InspectionStorage:
    """
    Storage temporário para artefatos de inspeção (Redis).

    Redis: armazenamento temporário durante revisão (TTL 2h, DB 2).
    Persistência no MinIO é feita via HTTP pelo ApprovalService.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_db: int = 2,
    ):
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis_db = redis_db
        self._redis: Optional[redis.Redis] = None

    # =========================================================================
    # Conexão lazy
    # =========================================================================

    def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis.from_url(
                self._redis_url,
                db=self._redis_db,
                decode_responses=False,
            )
        return self._redis

    # =========================================================================
    # Redis — Artefatos temporários
    # =========================================================================

    def _redis_key(self, task_id: str, suffix: str) -> str:
        return f"inspect:{task_id}:{suffix}"

    def save_metadata(self, task_id: str, metadata: InspectionMetadata) -> None:
        """Salva metadados da inspeção no Redis."""
        key = self._redis_key(task_id, "metadata")
        self._get_redis().setex(key, REDIS_TTL_SECONDS, metadata.model_dump_json())

    def get_metadata(self, task_id: str) -> Optional[InspectionMetadata]:
        """Obtém metadados da inspeção do Redis."""
        key = self._redis_key(task_id, "metadata")
        data = self._get_redis().get(key)
        if data is None:
            return None
        return InspectionMetadata.model_validate_json(data)

    def save_pdf(self, task_id: str, pdf_bytes: bytes) -> None:
        """Salva o PDF original no Redis temporariamente."""
        key = self._redis_key(task_id, "pdf")
        self._get_redis().setex(key, REDIS_TTL_SECONDS, pdf_bytes)

    def get_pdf(self, task_id: str) -> Optional[bytes]:
        """Obtém o PDF original do Redis."""
        key = self._redis_key(task_id, "pdf")
        return self._get_redis().get(key)

    def save_artifact(self, task_id: str, stage: InspectionStage, artifact_json: str) -> None:
        """Salva artefato de uma fase no Redis."""
        key = self._redis_key(task_id, stage.value)
        self._get_redis().setex(key, REDIS_TTL_SECONDS, artifact_json.encode("utf-8"))

    def get_artifact(self, task_id: str, stage: InspectionStage) -> Optional[str]:
        """Obtém artefato de uma fase do Redis como JSON string."""
        key = self._redis_key(task_id, stage.value)
        data = self._get_redis().get(key)
        if data is None:
            return None
        return data.decode("utf-8") if isinstance(data, bytes) else data

    def get_artifact_typed(self, task_id: str, stage: InspectionStage):
        """Obtém artefato de uma fase do Redis como modelo Pydantic."""
        raw = self.get_artifact(task_id, stage)
        if raw is None:
            return None
        model_cls = _STAGE_MODELS.get(stage)
        if model_cls is None:
            return json.loads(raw)
        return model_cls.model_validate_json(raw)

    def list_stages(self, task_id: str) -> list[InspectionStage]:
        """Lista as fases que têm artefatos salvos no Redis."""
        stages = []
        for stage in InspectionStage:
            key = self._redis_key(task_id, stage.value)
            if self._get_redis().exists(key):
                stages.append(stage)
        return stages

    def cleanup(self, task_id: str) -> int:
        """Remove todos os dados de uma inspeção do Redis."""
        pattern = self._redis_key(task_id, "*")
        keys = list(self._get_redis().scan_iter(match=pattern))
        if keys:
            return self._get_redis().delete(*keys)
        return 0

