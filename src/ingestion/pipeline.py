"""
Pipeline de Ingestao de PDFs com GPU.

Pipeline completo:
1. Docling (PDF -> Markdown) - GPU accelerated
2. SpanParser (Markdown -> Spans)
3. ArticleOrchestrator (LLM extraction)
4. ChunkMaterializer (parent-child chunks)
5. Embeddings (BGE-M3)

Retorna chunks prontos para indexacao (VPS faz o insert no Milvus).
"""

import os
import time
import hashlib
import logging
import tempfile
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from .models import IngestRequest, ProcessedChunk, IngestStatus, IngestError

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Resultado do processamento do pipeline."""
    status: IngestStatus
    document_id: str
    chunks: List[ProcessedChunk] = field(default_factory=list)
    phases: List[dict] = field(default_factory=list)
    errors: List[IngestError] = field(default_factory=list)
    total_time_seconds: float = 0.0
    markdown_content: str = ""
    document_hash: str = ""


class IngestionPipeline:
    """Pipeline de ingestao de documentos legais."""

    def __init__(self):
        self._docling_converter = None
        self._span_parser = None
        self._llm_client = None
        self._orchestrator = None
        self._embedder = None

    @property
    def docling_converter(self):
        if self._docling_converter is None:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, TesseractOcrOptions
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

            pipeline_options = ThreadedPdfPipelineOptions(
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
                layout_batch_size=64,
                table_batch_size=4,
                ocr_batch_size=4,
            )
            pipeline_options.do_ocr = False
            pipeline_options.ocr_options = TesseractOcrOptions(
                lang=['por', 'eng'],
                force_full_page_ocr=True,
            )

            self._docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=ThreadedStandardPdfPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )
            logger.info(f"Docling converter inicializado com GPU (ThreadedStandardPdfPipeline)")
        return self._docling_converter

    @property
    def span_parser(self):
        if self._span_parser is None:
            from ..parsing import SpanParser
            self._span_parser = SpanParser()
            logger.info("SpanParser inicializado")
        return self._span_parser

    @property
    def llm_client(self):
        if self._llm_client is None:
            from ..llm.vllm_client import VLLMClient, LLMConfig
            llm_config = LLMConfig.for_extraction()
            self._llm_client = VLLMClient(llm_config)
            logger.info("VLLMClient inicializado")
        return self._llm_client

    @property
    def orchestrator(self):
        if self._orchestrator is None:
            from ..parsing import ArticleOrchestrator
            self._orchestrator = ArticleOrchestrator(llm_client=self.llm_client)
            logger.info("ArticleOrchestrator inicializado")
        return self._orchestrator

    @property
    def embedder(self):
        if self._embedder is None:
            from ..embedder import get_embedder
            self._embedder = get_embedder()
            logger.info("BGE-M3 Embedder inicializado")
        return self._embedder

    def warmup(self) -> dict:
        """
        Pré-carrega modelos Docling na GPU.

        Executa uma conversão dummy para carregar todos os modelos
        (layout, table structure, etc.) na VRAM.

        Returns:
            dict com tempos de warmup
        """
        import tempfile
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        logger.info("=== Iniciando warmup do Docling ===")

        logger.info("Criando PDF de warmup...")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_pdf = f.name
            c = canvas.Canvas(f.name, pagesize=letter)
            c.drawString(100, 750, "Art. 1 Documento de warmup.")
            c.drawString(100, 730, "I - inciso de teste;")
            c.drawString(100, 710, "II - outro inciso.")
            c.save()

        try:
            logger.info("Inicializando Docling converter...")
            init_start = time.perf_counter()
            converter = self.docling_converter
            init_time = time.perf_counter() - init_start
            logger.info(f"Converter inicializado em {init_time:.2f}s")

            logger.info("Executando conversão de warmup (carrega modelos na GPU)...")
            warmup_start = time.perf_counter()
            result = converter.convert(temp_pdf)
            warmup_time = time.perf_counter() - warmup_start

            md = result.document.export_to_markdown()
            logger.info(f"Warmup concluído em {warmup_time:.2f}s ({len(md)} chars)")
            logger.info("=== Docling pronto para uso! ===")

            return {
                "status": "ready",
                "init_time_seconds": round(init_time, 2),
                "warmup_time_seconds": round(warmup_time, 2),
                "total_time_seconds": round(init_time + warmup_time, 2),
            }

        finally:
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)

    def is_warmed_up(self) -> bool:
        """Verifica se o Docling está carregado."""
        return self._docling_converter is not None

    def process(
        self,
        pdf_content: bytes,
        request: IngestRequest,
    ) -> PipelineResult:
        """
        Processa um PDF e retorna chunks prontos para indexacao.

        Args:
            pdf_content: Conteudo binario do PDF
            request: Metadados do documento

        Returns:
            PipelineResult com chunks processados
        """
        start_time = time.perf_counter()
        result = PipelineResult(
            status=IngestStatus.PROCESSING,
            document_id=request.document_id,
        )

        result.document_hash = hashlib.sha256(pdf_content).hexdigest()

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_content)
                temp_path = f.name

            try:
                result = self._phase_docling(temp_path, result)
                if result.status == IngestStatus.FAILED:
                    return result

                parsed_doc = self._phase_parsing(result)
                if result.status == IngestStatus.FAILED:
                    return result

                article_chunks = self._phase_extraction(parsed_doc, result, request.max_articles)
                if result.status == IngestStatus.FAILED:
                    return result

                materialized = self._phase_materialization(
                    article_chunks, parsed_doc, request, result
                )
                if result.status == IngestStatus.FAILED:
                    return result

                if not request.skip_embeddings:
                    self._phase_embeddings(materialized, result)
                    if result.status == IngestStatus.FAILED:
                        return result

                result.chunks = self._to_processed_chunks(materialized, request, result)
                result.status = IngestStatus.COMPLETED

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            logger.error(f"Erro no pipeline: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(
                phase="pipeline",
                message=str(e),
            ))

        result.total_time_seconds = round(time.perf_counter() - start_time, 2)
        return result

    def _phase_docling(self, pdf_path: str, result: PipelineResult) -> PipelineResult:
        """Fase 1: Docling - PDF -> Markdown"""
        phase_start = time.perf_counter()
        try:
            logger.info("Fase 1: Docling iniciando...")
            doc_result = self.docling_converter.convert(pdf_path)
            result.markdown_content = doc_result.document.export_to_markdown()

            duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "docling",
                "duration_seconds": duration,
                "output": f"Extraido {len(result.markdown_content)} caracteres",
                "success": True,
            })
            logger.info(f"Fase 1: Docling concluida em {duration}s")

        except Exception as e:
            logger.error(f"Erro no Docling: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(phase="docling", message=str(e)))

        return result

    def _phase_parsing(self, result: PipelineResult):
        """Fase 2: SpanParser - Markdown -> Spans"""
        phase_start = time.perf_counter()
        try:
            logger.info("Fase 2: SpanParser iniciando...")
            parsed_doc = self.span_parser.parse(result.markdown_content)

            duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "parsing",
                "duration_seconds": duration,
                "output": f"Encontrados {len(parsed_doc.spans)} spans",
                "success": True,
            })
            logger.info(f"Fase 2: SpanParser concluido em {duration}s")
            return parsed_doc

        except Exception as e:
            logger.error(f"Erro no SpanParser: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(phase="parsing", message=str(e)))
            return None

    def _phase_extraction(self, parsed_doc, result: PipelineResult, max_articles: Optional[int] = None):
        """Fase 3: ArticleOrchestrator - LLM extraction"""
        phase_start = time.perf_counter()
        try:
            logger.info("Fase 3: ArticleOrchestrator iniciando...")
            extraction_result = self.orchestrator.extract_all_articles(parsed_doc)

            duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "extraction",
                "duration_seconds": duration,
                "output": f"Extraidos {len(extraction_result.chunks)} artigos ({extraction_result.valid_articles} validos)",
                "success": True,
            })
            logger.info(f"Fase 3: ArticleOrchestrator concluido em {duration}s")
            return extraction_result.chunks

        except Exception as e:
            logger.error(f"Erro no ArticleOrchestrator: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(phase="extraction", message=str(e)))
            return None

    def _phase_materialization(self, article_chunks, parsed_doc, request: IngestRequest, result: PipelineResult):
        """Fase 4: ChunkMaterializer - parent-child chunks"""
        phase_start = time.perf_counter()
        try:
            logger.info("Fase 4: ChunkMaterializer iniciando...")
            from ..chunking import ChunkMaterializer, ChunkMetadata

            metadata = ChunkMetadata(
                schema_version="1.0.0",
                extractor_version="1.0.0",
                ingestion_timestamp=datetime.utcnow().isoformat(),
                document_hash=result.document_hash,
            )

            materializer = ChunkMaterializer(
                document_id=request.document_id,
                tipo_documento=request.tipo_documento,
                numero=request.numero,
                ano=request.ano,
                metadata=metadata,
            )
            materialized = materializer.materialize_all(article_chunks, parsed_doc)

            duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "materialization",
                "duration_seconds": duration,
                "output": f"Materializados {len(materialized)} chunks",
                "success": True,
            })
            logger.info(f"Fase 4: ChunkMaterializer concluido em {duration}s")
            return materialized

        except Exception as e:
            logger.error(f"Erro no ChunkMaterializer: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(phase="materialization", message=str(e)))
            return None

    def _phase_embeddings(self, materialized, result: PipelineResult):
        """Fase 5: Embeddings com BGE-M3"""
        phase_start = time.perf_counter()
        try:
            logger.info("Fase 5: Embeddings iniciando...")

            for chunk in materialized:
                text_for_embedding = chunk.enriched_text or chunk.text
                embed_result = self.embedder.encode([text_for_embedding])

                chunk._dense_vector = embed_result.dense_embeddings[0]
                chunk._sparse_vector = (embed_result.sparse_embeddings if embed_result.sparse_embeddings else [{}])[0]

                if chunk.thesis_text:
                    thesis_result = self.embedder.encode([chunk.thesis_text])
                    chunk._thesis_vector = thesis_result.dense_embeddings[0]
                else:
                    chunk._thesis_vector = [0.0] * 1024

            duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "embedding",
                "duration_seconds": duration,
                "output": f"Embeddings para {len(materialized)} chunks",
                "success": True,
            })
            logger.info(f"Fase 5: Embeddings concluido em {duration}s")

        except Exception as e:
            logger.error(f"Erro nos embeddings: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(phase="embedding", message=str(e)))

    def _to_processed_chunks(
        self,
        materialized,
        request: IngestRequest,
        result: PipelineResult
    ) -> List[ProcessedChunk]:
        """Converte MaterializedChunk para ProcessedChunk (Pydantic)."""
        chunks = []
        for chunk in materialized:
            pc = ProcessedChunk(
                chunk_id=chunk.chunk_id,
                parent_chunk_id=chunk.parent_chunk_id or "",
                span_id=chunk.span_id,
                device_type=chunk.device_type.value if hasattr(chunk.device_type, "value") else str(chunk.device_type),
                chunk_level=chunk.chunk_level.name.lower() if hasattr(chunk.chunk_level, "name") else str(chunk.chunk_level),
                text=chunk.text or "",
                enriched_text=chunk.enriched_text or chunk.text or "",
                context_header=chunk.context_header or "",
                thesis_text=chunk.thesis_text or "",
                thesis_type=chunk.thesis_type or "",
                synthetic_questions="",
                document_id=request.document_id,
                tipo_documento=request.tipo_documento,
                numero=request.numero,
                ano=request.ano,
                article_number=chunk.article_number or "",
            )

            if hasattr(chunk, "_dense_vector"):
                pc.dense_vector = chunk._dense_vector
            if hasattr(chunk, "_sparse_vector"):
                pc.sparse_vector = chunk._sparse_vector
            if hasattr(chunk, "_thesis_vector"):
                pc.thesis_vector = chunk._thesis_vector

            chunks.append(pc)

        return chunks


_pipeline: Optional[IngestionPipeline] = None


def get_pipeline() -> IngestionPipeline:
    """Retorna instancia singleton do pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline
