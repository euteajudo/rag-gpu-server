"""
Storage duplo para artefatos de inspeção.

- Redis: temporário durante revisão humana (TTL 2h)
- MinIO: permanente após aprovação

Keys Redis:
    inspect:{task_id}:metadata   → InspectionMetadata (JSON)
    inspect:{task_id}:{stage}    → Artefato da fase (JSON)
    inspect:{task_id}:pdf        → PDF original (bytes)

Paths MinIO:
    inspections/{document_id}/metadata.json
    inspections/{document_id}/canonical.md
    inspections/{document_id}/offsets.json
    inspections/{document_id}/pymupdf_result.json
    inspections/{document_id}/vlm_result.json
    inspections/{document_id}/pages/page_001.png
"""

import json
import logging
import os
from typing import Optional

import redis

from ..storage import ObjectStorageClient
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
    Storage duplo para artefatos de inspeção.

    Redis: armazenamento temporário durante revisão (TTL 2h).
    MinIO: armazenamento permanente após aprovação.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_db: int = 2,
        minio_client: Optional[ObjectStorageClient] = None,
    ):
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis_db = redis_db
        self._redis: Optional[redis.Redis] = None
        self._minio = minio_client

    # =========================================================================
    # Conexões lazy
    # =========================================================================

    def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis.from_url(
                self._redis_url,
                db=self._redis_db,
                decode_responses=False,
            )
        return self._redis

    def _get_minio(self) -> ObjectStorageClient:
        if self._minio is None:
            self._minio = ObjectStorageClient()
        return self._minio

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

    # =========================================================================
    # MinIO — Artefatos permanentes (após aprovação)
    # =========================================================================

    @staticmethod
    def _minio_base_path(document_id: str) -> str:
        return f"inspections/{document_id}"

    def persist_to_minio(
        self,
        task_id: str,
        document_id: str,
    ) -> list[str]:
        """
        Persiste todos os artefatos do Redis para o MinIO.

        Chamado após aprovação humana. Retorna lista de keys persistidas.
        """
        minio = self._get_minio()
        base = self._minio_base_path(document_id)
        persisted: list[str] = []

        # 1. Metadados da inspeção
        metadata = self.get_metadata(task_id)
        if metadata:
            key = f"{base}/metadata.json"
            minio.put_bytes(
                key,
                metadata.model_dump_json(indent=2).encode("utf-8"),
                content_type="application/json; charset=utf-8",
            )
            persisted.append(key)

        # 2. Artefatos de cada fase
        stage_files = {
            InspectionStage.PYMUPDF: "pymupdf_result.json",
            InspectionStage.VLM: "vlm_result.json",
            InspectionStage.RECONCILIATION: "reconciliation_result.json",
            InspectionStage.INTEGRITY: "integrity_result.json",
            InspectionStage.CHUNKS: "chunks_preview.json",
        }

        for stage, filename in stage_files.items():
            artifact_json = self.get_artifact(task_id, stage)
            if artifact_json:
                key = f"{base}/{filename}"
                minio.put_bytes(
                    key,
                    artifact_json.encode("utf-8"),
                    content_type="application/json; charset=utf-8",
                )
                persisted.append(key)

        # 3. Texto canônico reconciliado (extraído do artefato de reconciliação)
        recon_json = self.get_artifact(task_id, InspectionStage.RECONCILIATION)
        if recon_json:
            recon = ReconciliationArtifact.model_validate_json(recon_json)
            if recon.canonical_text:
                key = f"{base}/canonical.md"
                minio.put_bytes(
                    key,
                    recon.canonical_text.encode("utf-8"),
                    content_type="text/markdown; charset=utf-8",
                )
                persisted.append(key)

        # 4. Imagens de páginas anotadas (do artefato PyMuPDF)
        pymupdf_json = self.get_artifact(task_id, InspectionStage.PYMUPDF)
        if pymupdf_json:
            import base64
            pymupdf = PyMuPDFArtifact.model_validate_json(pymupdf_json)
            for page_result in pymupdf.pages:
                if page_result.image_base64:
                    page_num = page_result.page_number + 1
                    key = f"{base}/pages/page_{page_num:03d}.png"
                    png_bytes = base64.b64decode(page_result.image_base64)
                    minio.put_bytes(key, png_bytes, content_type="image/png")
                    persisted.append(key)

        logger.info(
            "Persistidos %d artefatos no MinIO para %s (inspect: %s)",
            len(persisted), document_id, task_id,
        )
        return persisted

    def has_approved_inspection(self, document_id: str) -> bool:
        """Verifica se existe uma inspeção aprovada no MinIO para este documento."""
        minio = self._get_minio()
        base = self._minio_base_path(document_id)
        return minio.exists(f"{base}/metadata.json")

    def get_approved_metadata(self, document_id: str) -> Optional[InspectionMetadata]:
        """Obtém os metadados da inspeção aprovada do MinIO."""
        minio = self._get_minio()
        base = self._minio_base_path(document_id)
        key = f"{base}/metadata.json"
        try:
            data = minio.get_bytes(key)
            return InspectionMetadata.model_validate_json(data)
        except FileNotFoundError:
            return None

    def get_approved_canonical(self, document_id: str) -> Optional[str]:
        """Obtém o canonical.md aprovado do MinIO."""
        minio = self._get_minio()
        base = self._minio_base_path(document_id)
        key = f"{base}/canonical.md"
        try:
            return minio.get_bytes(key).decode("utf-8")
        except FileNotFoundError:
            return None

    def check_pdf_hash(self, document_id: str, pdf_hash: str) -> bool:
        """
        Verifica se o hash do PDF confere com a inspeção aprovada.

        Retorna True se existe inspeção aprovada E o hash confere.
        Usado pelo pipeline para decidir se pula extração.
        """
        metadata = self.get_approved_metadata(document_id)
        if metadata is None:
            return False
        return metadata.pdf_hash == pdf_hash
