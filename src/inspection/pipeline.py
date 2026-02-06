"""
Pipeline dry-run de inspeção.

Executa as mesmas fases do pipeline de ingestão, mas:
- NÃO indexa no Milvus
- NÃO gera embeddings
- Salva artefatos intermediários no Redis via InspectionStorage
- Reporta progresso via callback
- Gera artefatos visuais (PyMuPDFArtifact com imagens anotadas)
"""

import hashlib
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from .models import (
    BBox,
    ChunkPreview,
    ChunksPreviewArtifact,
    InspectionMetadata,
    InspectionStage,
    InspectionStatus,
    IntegrityArtifact,
    IntegrityCheck,
    PyMuPDFArtifact,
    PyMuPDFBlock,
    PyMuPDFPageResult,
    ReconciliationArtifact,
    VLMArtifact,
)
from .page_renderer import PageRenderer
from .storage import InspectionStorage

logger = logging.getLogger(__name__)


class InspectionPipeline:
    """
    Pipeline dry-run para inspeção de documentos.

    Executa fases de processamento, salva artefatos intermediários,
    mas NÃO indexa no Milvus. Usado para revisão humana antes da
    ingestão definitiva.
    """

    def __init__(self, storage: Optional[InspectionStorage] = None):
        self._storage = storage or InspectionStorage()

        # Lazy-loaded components (same as IngestionPipeline)
        self._docling_converter = None
        self._span_parser = None
        self._llm_client = None
        self._orchestrator = None

    # =========================================================================
    # Lazy properties (reutiliza componentes do pipeline de ingestão)
    # =========================================================================

    @property
    def docling_converter(self):
        if self._docling_converter is None:
            from ..ingestion.pipeline import IngestionPipeline
            temp = IngestionPipeline()
            self._docling_converter = temp.docling_converter
        return self._docling_converter

    @property
    def span_parser(self):
        if self._span_parser is None:
            from ..parsing import SpanParser
            self._span_parser = SpanParser()
        return self._span_parser

    @property
    def llm_client(self):
        if self._llm_client is None:
            from ..llm.vllm_client import VLLMClient, LLMConfig
            self._llm_client = VLLMClient(LLMConfig.for_extraction())
        return self._llm_client

    @property
    def orchestrator(self):
        if self._orchestrator is None:
            from ..parsing import ArticleOrchestrator
            self._orchestrator = ArticleOrchestrator(llm_client=self.llm_client)
        return self._orchestrator

    # =========================================================================
    # Processo principal
    # =========================================================================

    def run(
        self,
        task_id: str,
        pdf_bytes: bytes,
        document_id: str,
        tipo_documento: str,
        numero: str,
        ano: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> InspectionMetadata:
        """
        Executa pipeline dry-run completo.

        Args:
            task_id: ID da tarefa de inspeção.
            pdf_bytes: Bytes do PDF.
            document_id: ID do documento (ex: LEI-14133-2021).
            tipo_documento: Tipo (LEI, DECRETO, IN, etc).
            numero: Número do documento.
            ano: Ano do documento.
            progress_callback: Callback(stage, progress) para progresso.

        Returns:
            InspectionMetadata com status final.
        """
        def report(stage: str, progress: float):
            if progress_callback:
                try:
                    progress_callback(stage, progress)
                except Exception as e:
                    logger.warning(f"Erro no progress_callback: {e}")

        start_time = time.perf_counter()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        # Conta páginas do PDF
        import fitz
        temp_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(temp_doc)
        temp_doc.close()

        # Cria e salva metadados iniciais
        metadata = InspectionMetadata(
            inspection_id=task_id,
            document_id=document_id,
            tipo_documento=tipo_documento,
            numero=numero,
            ano=ano,
            pdf_hash=pdf_hash,
            pdf_size_bytes=len(pdf_bytes),
            total_pages=total_pages,
            started_at=datetime.now(timezone.utc).isoformat(),
            status=InspectionStatus.PROCESSING,
        )
        self._storage.save_metadata(task_id, metadata)
        self._storage.save_pdf(task_id, pdf_bytes)

        report("initializing", 0.05)

        try:
            # Fase 1: PyMuPDF — extração de blocos e renderização
            report("pymupdf", 0.10)
            pymupdf_artifact = self._phase_pymupdf(task_id, pdf_bytes)
            report("pymupdf", 0.30)

            # Fase 2: VLM — placeholder (será implementado na migração VLM)
            report("vlm", 0.32)
            vlm_artifact = self._phase_vlm_placeholder(task_id, total_pages)
            report("vlm", 0.40)

            # Fase 3: Reconciliação — placeholder (depende do VLM)
            report("reconciliation", 0.42)
            recon_artifact = self._phase_reconciliation_placeholder(
                task_id, pdf_bytes, document_id, tipo_documento, numero, ano,
            )
            report("reconciliation", 0.60)

            # Fase 4: Integridade — valida a extração
            report("integrity", 0.62)
            integrity_artifact = self._phase_integrity(
                task_id, recon_artifact,
            )
            report("integrity", 0.75)

            # Fase 5: Chunks — preview dos chunks que seriam criados
            report("chunks", 0.77)
            chunks_artifact = self._phase_chunks_preview(
                task_id, pdf_bytes, document_id, tipo_documento, numero, ano,
            )
            report("chunks", 0.95)

            # Finaliza
            metadata.status = InspectionStatus.COMPLETED
            metadata.completed_at = datetime.now(timezone.utc).isoformat()
            self._storage.save_metadata(task_id, metadata)

            report("completed", 1.0)

        except Exception as e:
            logger.error(f"Erro no pipeline de inspeção: {e}", exc_info=True)
            metadata.status = InspectionStatus.FAILED
            metadata.completed_at = datetime.now(timezone.utc).isoformat()
            self._storage.save_metadata(task_id, metadata)

        total_time = round(time.perf_counter() - start_time, 2)
        logger.info(
            f"Inspeção {task_id} finalizada em {total_time}s — status={metadata.status.value}"
        )
        return metadata

    # =========================================================================
    # Fase 1: PyMuPDF — Blocos de texto extraídos com bboxes
    # =========================================================================

    def _phase_pymupdf(self, task_id: str, pdf_bytes: bytes) -> PyMuPDFArtifact:
        """Extrai blocos de texto de cada página e renderiza imagens anotadas."""
        phase_start = time.perf_counter()
        logger.info("Inspeção Fase 1: PyMuPDF — extraindo blocos...")

        pages: list[PyMuPDFPageResult] = []
        total_blocks = 0
        total_chars = 0

        with PageRenderer(pdf_bytes) as renderer:
            for page_num in range(renderer.page_count):
                width, height = renderer.get_page_size(page_num)

                # Extrai blocos de texto
                raw_blocks = renderer.extract_blocks(page_num)
                pymupdf_blocks: list[PyMuPDFBlock] = []

                bboxes_for_render: list[tuple[BBox, str]] = []
                for block_data in raw_blocks:
                    bbox = BBox(**block_data["bbox"])
                    pymupdf_blocks.append(PyMuPDFBlock(
                        block_index=block_data["block_index"],
                        text=block_data["text"],
                        bbox=bbox,
                        font_size=block_data["font_size"],
                        is_bold=block_data["is_bold"],
                        page=page_num,
                    ))
                    bboxes_for_render.append((bbox, "pymupdf"))

                total_blocks += len(pymupdf_blocks)
                total_chars += sum(len(b.text) for b in pymupdf_blocks)

                # Renderiza página com bboxes anotados
                image_b64 = renderer.render_page_base64(page_num, bboxes_for_render)

                pages.append(PyMuPDFPageResult(
                    page_number=page_num,
                    width=width,
                    height=height,
                    blocks=pymupdf_blocks,
                    image_base64=image_b64,
                ))

        duration_ms = (time.perf_counter() - phase_start) * 1000
        artifact = PyMuPDFArtifact(
            pages=pages,
            total_blocks=total_blocks,
            total_pages=len(pages),
            total_chars=total_chars,
            duration_ms=duration_ms,
        )

        self._storage.save_artifact(
            task_id, InspectionStage.PYMUPDF, artifact.model_dump_json(),
        )
        logger.info(
            f"Inspeção Fase 1: PyMuPDF — {total_blocks} blocos, "
            f"{len(pages)} páginas em {duration_ms:.0f}ms"
        )
        return artifact

    # =========================================================================
    # Fase 2: VLM — Placeholder (Fase 0, será populado na migração)
    # =========================================================================

    def _phase_vlm_placeholder(
        self, task_id: str, total_pages: int,
    ) -> VLMArtifact:
        """Placeholder para a fase VLM. Será implementado na migração."""
        logger.info("Inspeção Fase 2: VLM — placeholder (será implementado na migração)")

        artifact = VLMArtifact(
            pages=[],
            total_elements=0,
            total_pages=total_pages,
            hierarchy_depth=0,
            duration_ms=0.0,
        )

        self._storage.save_artifact(
            task_id, InspectionStage.VLM, artifact.model_dump_json(),
        )
        return artifact

    # =========================================================================
    # Fase 3: Reconciliação — Usa Docling+SpanParser como proxy na Fase 0
    # =========================================================================

    def _phase_reconciliation_placeholder(
        self,
        task_id: str,
        pdf_bytes: bytes,
        document_id: str,
        tipo_documento: str,
        numero: str,
        ano: int,
    ) -> ReconciliationArtifact:
        """
        Na Fase 0 (scaffold), a reconciliação usa o pipeline atual
        (Docling + SpanParser) para gerar o canonical text.

        Quando o VLM estiver implementado, esta fase fará matching
        real entre blocos PyMuPDF e elementos VLM.
        """
        phase_start = time.perf_counter()
        logger.info("Inspeção Fase 3: Reconciliação — usando Docling como proxy...")

        canonical_text = ""
        try:
            # Usa Docling para gerar markdown (proxy para canonical text)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_bytes)
                temp_path = f.name

            try:
                doc_result = self.docling_converter.convert(temp_path)
                canonical_text = doc_result.document.export_to_markdown()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"Erro na fase de reconciliação: {e}", exc_info=True)

        duration_ms = (time.perf_counter() - phase_start) * 1000
        artifact = ReconciliationArtifact(
            pages=[],
            canonical_text=canonical_text,
            duration_ms=duration_ms,
        )

        self._storage.save_artifact(
            task_id, InspectionStage.RECONCILIATION, artifact.model_dump_json(),
        )
        logger.info(
            f"Inspeção Fase 3: Reconciliação — "
            f"{len(canonical_text)} chars em {duration_ms:.0f}ms"
        )
        return artifact

    # =========================================================================
    # Fase 4: Integridade — Validação da extração
    # =========================================================================

    def _phase_integrity(
        self,
        task_id: str,
        recon_artifact: ReconciliationArtifact,
    ) -> IntegrityArtifact:
        """Executa validações de integridade no texto extraído."""
        phase_start = time.perf_counter()
        logger.info("Inspeção Fase 4: Integridade — validando...")

        checks: list[IntegrityCheck] = []
        warnings: list[str] = []

        canonical = recon_artifact.canonical_text

        # Check 1: Texto não vazio
        has_content = len(canonical.strip()) > 0
        checks.append(IntegrityCheck(
            check_name="content_not_empty",
            passed=has_content,
            message="Texto canônico extraído com sucesso" if has_content else "Texto canônico vazio",
        ))

        # Check 2: Contém artigos
        import re
        article_pattern = re.compile(r'(?:^|\n)\s*Art\.?\s+\d+', re.IGNORECASE)
        article_matches = article_pattern.findall(canonical)
        has_articles = len(article_matches) > 0
        checks.append(IntegrityCheck(
            check_name="has_articles",
            passed=has_articles,
            message=f"Encontrados {len(article_matches)} artigos" if has_articles else "Nenhum artigo encontrado",
            details={"article_count": len(article_matches)},
        ))

        # Check 3: Tamanho mínimo
        min_chars = 100
        has_min_length = len(canonical) >= min_chars
        checks.append(IntegrityCheck(
            check_name="min_length",
            passed=has_min_length,
            message=f"Texto com {len(canonical)} caracteres" if has_min_length
                    else f"Texto muito curto: {len(canonical)} < {min_chars}",
            details={"char_count": len(canonical), "threshold": min_chars},
        ))

        # Check 4: Sem lixo (excesso de caracteres não-ASCII problemáticos)
        non_ascii_ratio = sum(1 for c in canonical if ord(c) > 127 and not c.isalpha()) / max(len(canonical), 1)
        clean_text = non_ascii_ratio < 0.1
        checks.append(IntegrityCheck(
            check_name="clean_text",
            passed=clean_text,
            message="Texto limpo" if clean_text else f"Proporção alta de caracteres especiais: {non_ascii_ratio:.1%}",
            details={"non_ascii_ratio": round(non_ascii_ratio, 4)},
        ))

        if not has_articles:
            warnings.append("Nenhum artigo detectado — documento pode não ser legislação")

        passed_count = sum(1 for c in checks if c.passed)
        failed_count = len(checks) - passed_count
        overall_score = passed_count / len(checks) if checks else 0.0

        duration_ms = (time.perf_counter() - phase_start) * 1000
        artifact = IntegrityArtifact(
            checks=checks,
            overall_score=overall_score,
            passed=failed_count == 0,
            total_checks=len(checks),
            passed_checks=passed_count,
            failed_checks=failed_count,
            warnings=warnings,
            duration_ms=duration_ms,
        )

        self._storage.save_artifact(
            task_id, InspectionStage.INTEGRITY, artifact.model_dump_json(),
        )
        logger.info(
            f"Inspeção Fase 4: Integridade — {passed_count}/{len(checks)} checks, "
            f"score={overall_score:.2f} em {duration_ms:.0f}ms"
        )
        return artifact

    # =========================================================================
    # Fase 5: Chunks — Preview dos chunks que seriam criados
    # =========================================================================

    def _phase_chunks_preview(
        self,
        task_id: str,
        pdf_bytes: bytes,
        document_id: str,
        tipo_documento: str,
        numero: str,
        ano: int,
    ) -> ChunksPreviewArtifact:
        """
        Executa o pipeline real (SpanParser + ArticleOrchestrator + ChunkMaterializer)
        para gerar preview dos chunks que SERIAM criados na ingestão.
        """
        phase_start = time.perf_counter()
        logger.info("Inspeção Fase 5: Chunks — gerando preview...")

        chunks_preview: list[ChunkPreview] = []
        articles_count = 0
        paragraphs_count = 0
        incisos_count = 0
        alineas_count = 0
        max_depth = 0

        try:
            # Converte PDF para markdown via Docling
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_bytes)
                temp_path = f.name

            try:
                doc_result = self.docling_converter.convert(temp_path)
                markdown_content = doc_result.document.export_to_markdown()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # SpanParser
            parsed_doc = self.span_parser.parse(markdown_content)
            logger.info(f"  SpanParser: {len(parsed_doc.spans)} spans")

            # ArticleOrchestrator (usa LLM)
            extraction_result = self.orchestrator.extract_all_articles(parsed_doc)
            logger.info(
                f"  ArticleOrchestrator: {len(extraction_result.chunks)} artigos "
                f"({extraction_result.valid_articles} válidos)"
            )

            # ChunkMaterializer
            from ..chunking import ChunkMaterializer, ChunkMetadata
            metadata = ChunkMetadata(
                schema_version="1.0.0",
                extractor_version="1.0.0",
                ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
                document_hash=hashlib.sha256(pdf_bytes).hexdigest(),
            )

            # Tenta extrair offsets canônicos
            offsets_map = {}
            canonical_hash = ""
            try:
                from ..chunking.canonical_offsets import (
                    extract_offsets_from_parsed_doc,
                    normalize_canonical_text,
                )
                offsets_map, canonical_hash = extract_offsets_from_parsed_doc(parsed_doc)
                canonical_text = normalize_canonical_text(
                    getattr(parsed_doc, 'source_text', '') or ''
                )
            except Exception as e:
                logger.warning(f"Offsets canônicos não disponíveis: {e}")
                canonical_text = ""

            materializer = ChunkMaterializer(
                document_id=document_id,
                tipo_documento=tipo_documento,
                numero=numero,
                ano=ano,
                metadata=metadata,
                offsets_map=offsets_map,
                canonical_hash=canonical_hash,
                canonical_text=canonical_text,
            )
            materialized = materializer.materialize_all(
                extraction_result.chunks, parsed_doc,
            )
            logger.info(f"  ChunkMaterializer: {len(materialized)} chunks")

            # Monta o collection_prefix para o node_id
            collection_prefix = "leis"

            # Converte para ChunkPreview
            for chunk in materialized:
                device_type = getattr(chunk, 'device_type', 'article')
                if hasattr(device_type, 'value'):
                    device_type = device_type.value
                dt_str = str(device_type).lower()

                chunk_level = getattr(chunk, 'chunk_level', 'article')
                if hasattr(chunk_level, 'value'):
                    chunk_level = chunk_level.value
                cl_str = str(chunk_level).lower()

                span_id = getattr(chunk, 'span_id', '')
                node_id = f"{collection_prefix}:{document_id}#{span_id}"
                chunk_id = f"{document_id}#{span_id}"

                parent_span_id = getattr(chunk, 'parent_chunk_id', '') or ''
                parent_node_id = ""
                if parent_span_id:
                    parent_node_id = f"{collection_prefix}:{parent_span_id}"

                # Offsets canônicos
                canonical_start = getattr(chunk, 'canonical_start', -1)
                canonical_end = getattr(chunk, 'canonical_end', -1)

                # Conta filhos
                children_count = sum(
                    1 for c in materialized
                    if (getattr(c, 'parent_chunk_id', '') or '') == chunk_id
                    or (getattr(c, 'parent_chunk_id', '') or '') == f"{document_id}#{span_id}"
                )

                chunks_preview.append(ChunkPreview(
                    node_id=node_id,
                    chunk_id=chunk_id,
                    parent_node_id=parent_node_id,
                    span_id=span_id,
                    device_type=dt_str,
                    chunk_level=cl_str,
                    text=chunk.text or "",
                    canonical_start=canonical_start if canonical_start is not None else -1,
                    canonical_end=canonical_end if canonical_end is not None else -1,
                    children_count=children_count,
                ))

                # Conta por tipo
                if dt_str == "article":
                    articles_count += 1
                    max_depth = max(max_depth, 1)
                elif dt_str == "paragraph":
                    paragraphs_count += 1
                    max_depth = max(max_depth, 2)
                elif dt_str == "inciso":
                    incisos_count += 1
                    max_depth = max(max_depth, 3)
                elif dt_str == "alinea":
                    alineas_count += 1
                    max_depth = max(max_depth, 4)

        except Exception as e:
            logger.error(f"Erro na fase de chunks preview: {e}", exc_info=True)

        duration_ms = (time.perf_counter() - phase_start) * 1000
        artifact = ChunksPreviewArtifact(
            chunks=chunks_preview,
            total_chunks=len(chunks_preview),
            articles_count=articles_count,
            paragraphs_count=paragraphs_count,
            incisos_count=incisos_count,
            alineas_count=alineas_count,
            max_depth=max_depth,
            duration_ms=duration_ms,
        )

        self._storage.save_artifact(
            task_id, InspectionStage.CHUNKS, artifact.model_dump_json(),
        )
        logger.info(
            f"Inspeção Fase 5: Chunks — {len(chunks_preview)} chunks "
            f"(ART={articles_count}, PAR={paragraphs_count}, "
            f"INC={incisos_count}, ALI={alineas_count}) em {duration_ms:.0f}ms"
        )
        return artifact
