"""
Serviço de aprovação de inspeções.

Fluxo:
1. Valida que a inspeção está COMPLETED
2. Atualiza metadados (approved_at, approved_by, status=APPROVED)
3. Gera offsets.json a partir dos chunks preview
4. Envia todos os artefatos para a VPS via HTTP POST
   (VPS grava no MinIO local — RunPod NÃO acessa MinIO diretamente)
5. Limpa artefatos temporários do Redis
6. Retorna ApprovalResult com detalhes
"""

import base64
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from .models import (
    ApprovalResult,
    ChunksPreviewArtifact,
    InspectionMetadata,
    InspectionStage,
    InspectionStatus,
    PyMuPDFArtifact,
    ReconciliationArtifact,
)
from .storage import InspectionStorage

logger = logging.getLogger(__name__)


class ApprovalService:
    """Serviço responsável pela aprovação de inspeções."""

    def __init__(self, storage: Optional[InspectionStorage] = None):
        self._storage = storage or InspectionStorage()
        self._uploader = None

    @property
    def uploader(self):
        """Lazy-load do InspectionUploader."""
        if self._uploader is None:
            from ..sinks.inspection_uploader import get_inspection_uploader
            self._uploader = get_inspection_uploader()
        return self._uploader

    def approve(
        self,
        task_id: str,
        approved_by: str = "admin",
    ) -> ApprovalResult:
        """
        Aprova uma inspeção, enviando artefatos para a VPS via HTTP.

        Args:
            task_id: ID da inspeção a aprovar.
            approved_by: Nome/email de quem aprovou.

        Returns:
            ApprovalResult com detalhes da persistência.
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

            # 5. Coleta todos os artefatos do Redis
            artifacts = self._collect_artifacts(task_id, document_id, metadata)

            # 6. Envia para VPS via HTTP POST
            upload_result = self.uploader.upload(**artifacts)

            if not upload_result.success:
                # Reverte status se upload falhou
                metadata.status = InspectionStatus.COMPLETED
                metadata.approved_at = None
                metadata.approved_by = None
                self._storage.save_metadata(task_id, metadata)

                return ApprovalResult(
                    success=False,
                    inspection_id=task_id,
                    document_id=document_id,
                    message=f"Falha ao enviar artefatos para VPS: {upload_result.error}",
                )

            persisted_keys = upload_result.artifacts_persisted or []

            # 7. Limpa artefatos temporários do Redis
            cleaned = self._storage.cleanup(task_id)
            logger.info(
                "Aprovação %s: limpas %d chaves do Redis", task_id, cleaned,
            )

            minio_path = f"inspections/{document_id}/"
            logger.info(
                "Inspeção %s aprovada por %s: %d artefatos enviados para VPS (%s)",
                task_id, approved_by, len(persisted_keys), minio_path,
            )

            # Calcula tamanho do canonical.md
            canonical_md_size = 0
            if artifacts.get("canonical_md"):
                canonical_md_size = len(artifacts["canonical_md"].encode("utf-8"))

            return ApprovalResult(
                success=True,
                inspection_id=task_id,
                document_id=document_id,
                minio_path=minio_path,
                artifacts_persisted=persisted_keys,
                canonical_md_size=canonical_md_size,
                message=f"Inspeção aprovada. {len(persisted_keys)} artefatos enviados para VPS.",
            )

        except Exception as e:
            logger.error(
                "Erro ao aprovar inspeção %s: %s", task_id, e, exc_info=True,
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
                message=f"Erro ao processar aprovação: {e}",
            )

    def _collect_artifacts(
        self,
        task_id: str,
        document_id: str,
        metadata: InspectionMetadata,
    ) -> dict:
        """
        Coleta todos os artefatos do Redis para envio.

        Returns:
            dict com kwargs para InspectionUploader.upload()
        """
        artifacts: dict = {
            "document_id": document_id,
            "metadata_json": metadata.model_dump_json(indent=2),
        }

        # Artefatos por fase
        stage_args = {
            InspectionStage.PYMUPDF: "pymupdf_result_json",
            InspectionStage.VLM: "vlm_result_json",
            InspectionStage.RECONCILIATION: "reconciliation_result_json",
            InspectionStage.INTEGRITY: "integrity_result_json",
            InspectionStage.CHUNKS: "chunks_preview_json",
        }

        for stage, arg_name in stage_args.items():
            artifact_json = self._storage.get_artifact(task_id, stage)
            if artifact_json:
                artifacts[arg_name] = artifact_json

        # Extrai canonical.md do artefato de reconciliação
        recon_json = artifacts.get("reconciliation_result_json")
        if recon_json:
            try:
                recon = ReconciliationArtifact.model_validate_json(recon_json)
                if recon.canonical_text:
                    artifacts["canonical_md"] = recon.canonical_text
            except Exception as e:
                logger.warning("Não foi possível extrair canonical_text: %s", e)

        # Gera offsets.json
        offsets_json = self._build_offsets(task_id, recon_json)
        if offsets_json:
            artifacts["offsets_json"] = offsets_json

        # Extrai imagens de páginas do artefato PyMuPDF
        pymupdf_json = artifacts.get("pymupdf_result_json")
        if pymupdf_json:
            page_images = self._extract_page_images(pymupdf_json)
            if page_images:
                artifacts["page_images"] = page_images

        return artifacts

    def _build_offsets(
        self,
        task_id: str,
        recon_json: Optional[str],
    ) -> Optional[str]:
        """
        Gera offsets.json a partir dos chunks preview.

        Returns:
            JSON string do mapa de offsets, ou None.
        """
        chunks_json = self._storage.get_artifact(task_id, InspectionStage.CHUNKS)
        if not chunks_json:
            return None

        try:
            chunks_artifact = ChunksPreviewArtifact.model_validate_json(chunks_json)

            offsets: dict[str, dict] = {}
            for chunk in chunks_artifact.chunks:
                if chunk.canonical_start >= 0 and chunk.canonical_end >= 0:
                    offsets[chunk.span_id] = {
                        "start": chunk.canonical_start,
                        "end": chunk.canonical_end,
                        "node_id": chunk.node_id,
                        "device_type": chunk.device_type,
                    }

            if not offsets and recon_json:
                # Fallback: extrai offsets por regex do canonical text
                recon = ReconciliationArtifact.model_validate_json(recon_json)
                if recon.canonical_text:
                    offsets = self._extract_offsets_by_regex(recon.canonical_text)

            if not offsets:
                return None

            # Calcula tamanho do canonical_text
            canonical_len = 0
            if recon_json:
                recon = ReconciliationArtifact.model_validate_json(recon_json)
                canonical_len = len(recon.canonical_text)

            offsets_data = {
                "document_id": self._storage.get_metadata(task_id).document_id
                if self._storage.get_metadata(task_id) else "",
                "canonical_text_length": canonical_len,
                "total_spans": len(offsets),
                "offsets": offsets,
            }

            return json.dumps(offsets_data, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning("Não foi possível gerar offsets.json: %s", e)
            return None

    @staticmethod
    def _extract_page_images(pymupdf_json: str) -> dict[str, bytes]:
        """
        Extrai imagens PNG das páginas do artefato PyMuPDF.

        Returns:
            dict[page_key, png_bytes] (ex: {"page_001": b"..."})
        """
        images: dict[str, bytes] = {}
        try:
            pymupdf = PyMuPDFArtifact.model_validate_json(pymupdf_json)
            for page_result in pymupdf.pages:
                if page_result.image_base64:
                    page_num = page_result.page_number + 1
                    page_key = f"page_{page_num:03d}"
                    images[page_key] = base64.b64decode(page_result.image_base64)
        except Exception as e:
            logger.warning("Não foi possível extrair imagens das páginas: %s", e)
        return images

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
