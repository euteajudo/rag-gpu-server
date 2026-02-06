"""
Serviço de aprovação de inspeções.

Fluxo:
1. Valida que a inspeção está COMPLETED
2. Atualiza metadados (approved_at, approved_by, status=APPROVED)
3. Gera offsets.json a partir do canonical text
4. Persiste todos os artefatos no MinIO (permanente)
5. Limpa artefatos temporários do Redis
6. Retorna ApprovalResult com detalhes
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from .models import (
    ApprovalResult,
    InspectionMetadata,
    InspectionStage,
    InspectionStatus,
    ReconciliationArtifact,
)
from .storage import InspectionStorage

logger = logging.getLogger(__name__)


class ApprovalService:
    """Serviço responsável pela aprovação de inspeções."""

    def __init__(self, storage: Optional[InspectionStorage] = None):
        self._storage = storage or InspectionStorage()

    def approve(
        self,
        task_id: str,
        approved_by: str = "admin",
    ) -> ApprovalResult:
        """
        Aprova uma inspeção, persistindo artefatos no MinIO.

        Args:
            task_id: ID da inspeção a aprovar.
            approved_by: Nome/email de quem aprovou.

        Returns:
            ApprovalResult com detalhes da persistência.

        Raises:
            ValueError: Se a inspeção não existe ou não está no estado correto.
        """
        # 1. Busca metadados
        metadata = self._storage.get_metadata(task_id)
        if metadata is None:
            return ApprovalResult(
                success=False,
                inspection_id=task_id,
                document_id="",
                message=f"Inspeção {task_id} não encontrada no Redis (pode ter expirado após 2h)",
            )

        # 2. Valida estado
        if metadata.status == InspectionStatus.APPROVED:
            return ApprovalResult(
                success=False,
                inspection_id=task_id,
                document_id=metadata.document_id,
                message=f"Inspeção {task_id} já foi aprovada em {metadata.approved_at}",
            )

        if metadata.status != InspectionStatus.COMPLETED:
            return ApprovalResult(
                success=False,
                inspection_id=task_id,
                document_id=metadata.document_id,
                message=(
                    f"Inspeção {task_id} não está completa "
                    f"(status atual: {metadata.status.value})"
                ),
            )

        # 3. Verifica que existem artefatos mínimos
        stages = self._storage.list_stages(task_id)
        if InspectionStage.PYMUPDF not in stages:
            return ApprovalResult(
                success=False,
                inspection_id=task_id,
                document_id=metadata.document_id,
                message="Artefato PyMuPDF não encontrado — inspeção incompleta",
            )

        document_id = metadata.document_id

        try:
            # 4. Atualiza metadata com aprovação
            metadata.status = InspectionStatus.APPROVED
            metadata.approved_at = datetime.now(timezone.utc).isoformat()
            metadata.approved_by = approved_by
            self._storage.save_metadata(task_id, metadata)

            # 5. Gera e persiste offsets.json
            offsets_persisted = self._persist_offsets(task_id, document_id)

            # 6. Persiste todos os artefatos no MinIO
            persisted_keys = self._storage.persist_to_minio(task_id, document_id)

            if offsets_persisted:
                persisted_keys.append(offsets_persisted)

            # 7. Calcula tamanho do canonical.md
            canonical_md_size = 0
            recon_json = self._storage.get_artifact(
                task_id, InspectionStage.RECONCILIATION,
            )
            if recon_json:
                recon = ReconciliationArtifact.model_validate_json(recon_json)
                canonical_md_size = len(recon.canonical_text.encode("utf-8"))

            # 8. Limpa artefatos temporários do Redis
            cleaned = self._storage.cleanup(task_id)
            logger.info(
                f"Aprovação {task_id}: limpas {cleaned} chaves do Redis"
            )

            minio_path = f"inspections/{document_id}/"
            logger.info(
                f"Inspeção {task_id} aprovada por {approved_by}: "
                f"{len(persisted_keys)} artefatos em {minio_path}"
            )

            return ApprovalResult(
                success=True,
                inspection_id=task_id,
                document_id=document_id,
                minio_path=minio_path,
                artifacts_persisted=persisted_keys,
                canonical_md_size=canonical_md_size,
                message=f"Inspeção aprovada com sucesso. {len(persisted_keys)} artefatos persistidos no MinIO.",
            )

        except Exception as e:
            logger.error(
                f"Erro ao aprovar inspeção {task_id}: {e}", exc_info=True,
            )
            # Reverte status
            metadata.status = InspectionStatus.COMPLETED
            metadata.approved_at = None
            metadata.approved_by = None
            self._storage.save_metadata(task_id, metadata)

            return ApprovalResult(
                success=False,
                inspection_id=task_id,
                document_id=document_id,
                message=f"Erro ao persistir artefatos: {e}",
            )

    def _persist_offsets(self, task_id: str, document_id: str) -> Optional[str]:
        """
        Gera offsets.json a partir do canonical text e chunks preview.

        Offsets mapeia span_ids para posições no canonical text.
        Retorna a key MinIO onde foi salvo, ou None se não aplicável.
        """
        recon_json = self._storage.get_artifact(
            task_id, InspectionStage.RECONCILIATION,
        )
        chunks_json = self._storage.get_artifact(
            task_id, InspectionStage.CHUNKS,
        )

        if not recon_json or not chunks_json:
            return None

        try:
            recon = ReconciliationArtifact.model_validate_json(recon_json)
            from .models import ChunksPreviewArtifact
            chunks_artifact = ChunksPreviewArtifact.model_validate_json(chunks_json)

            if not recon.canonical_text:
                return None

            # Monta mapa de offsets a partir dos chunks preview
            offsets: dict[str, dict] = {}
            for chunk in chunks_artifact.chunks:
                if chunk.canonical_start >= 0 and chunk.canonical_end >= 0:
                    offsets[chunk.span_id] = {
                        "start": chunk.canonical_start,
                        "end": chunk.canonical_end,
                        "node_id": chunk.node_id,
                        "device_type": chunk.device_type,
                    }

            if not offsets:
                # Sem offsets canônicos — tenta gerar offsets simples por regex
                offsets = self._extract_offsets_by_regex(recon.canonical_text)

            if not offsets:
                return None

            offsets_data = {
                "document_id": document_id,
                "canonical_text_length": len(recon.canonical_text),
                "total_spans": len(offsets),
                "offsets": offsets,
            }

            # Persiste no MinIO
            minio = self._storage._get_minio()
            base = self._storage._minio_base_path(document_id)
            key = f"{base}/offsets.json"
            minio.put_bytes(
                key,
                json.dumps(offsets_data, ensure_ascii=False, indent=2).encode("utf-8"),
                content_type="application/json; charset=utf-8",
            )

            logger.info(
                f"Offsets persistidos: {len(offsets)} spans para {document_id}"
            )
            return key

        except Exception as e:
            logger.warning(f"Não foi possível gerar offsets.json: {e}")
            return None

    @staticmethod
    def _extract_offsets_by_regex(canonical_text: str) -> dict[str, dict]:
        """
        Extrai offsets de artigos no canonical text usando regex.

        Fallback quando offsets canônicos do ChunkMaterializer não estão
        disponíveis. Captura apenas artigos (ART-NNN).
        """
        offsets: dict[str, dict] = {}
        pattern = re.compile(
            r'(?:^|\n)(Art\.?\s+(\d+)[°ºo]?.*?)(?=\nArt\.?\s+\d+|\Z)',
            re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(canonical_text):
            article_num = match.group(2)
            span_id = f"ART-{int(article_num):03d}"
            offsets[span_id] = {
                "start": match.start(1),
                "end": match.end(1),
                "device_type": "article",
            }

        return offsets
