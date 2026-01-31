"""
Pipeline de Ingestao de PDFs com GPU.

Pipeline completo:
1. Docling (PDF -> Markdown) - GPU accelerated
2. Validacao de Qualidade (detecta texto corrompido)
3. OCR Fallback (se qualidade baixa)
4. SpanParser (Markdown -> Spans)
5. ArticleOrchestrator (LLM extraction)
6. ChunkMaterializer (parent-child chunks)
7. Embeddings (BGE-M3)

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
from enum import Enum

from .models import IngestRequest, ProcessedChunk, IngestStatus, IngestError
from .quality_validator import QualityValidator
from .markdown_sanitizer import MarkdownSanitizer
from .article_validator import ArticleValidator
from ..chunking.citation_extractor import extract_citations_from_chunk


class ExtractionMethod(str, Enum):
    """Metodo usado para extrair o texto do PDF."""
    NATIVE_TEXT = "native_text"  # Texto nativo do PDF
    OCR_EASYOCR = "ocr_easyocr"  # OCR com EasyOCR
    OCR_TESSERACT = "ocr_tesseract"  # OCR com Tesseract

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
    # Novas métricas de qualidade
    extraction_method: ExtractionMethod = ExtractionMethod.NATIVE_TEXT
    quality_score: float = 0.0
    quality_issues: List[str] = field(default_factory=list)
    ocr_fallback_used: bool = False
    # Validação de artigos (Fase Docling)
    validation_docling: Optional[dict] = None


class IngestionPipeline:
    """Pipeline de ingestao de documentos legais."""

    def __init__(self):
        self._docling_converter = None
        self._docling_converter_ocr = None  # Converter com OCR para fallback
        self._span_parser = None
        self._llm_client = None
        self._orchestrator = None
        self._embedder = None
        self._quality_validator = QualityValidator(
            min_readable_ratio=0.7,
            min_article_count=1,
            min_word_ratio=0.3,
        )

    def _create_docling_converter(self, enable_ocr: bool = False, fast_mode: bool = True):
        """
        Cria um DocumentConverter do Docling.

        Args:
            enable_ocr: Se True, habilita OCR para PDFs com texto corrompido
            fast_mode: Se True, desabilita análise de tabelas (muito mais rápido)

        Referência: https://docling-project.github.io/docling/examples/full_page_ocr/
        """
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

        # Configuração GPU otimizada
        # Nota: Algumas versões do Docling podem requerer ocr_options no construtor
        try:
            pipeline_options = PdfPipelineOptions()
        except TypeError:
            # Fallback para versões que requerem ocr_options
            pipeline_options = PdfPipelineOptions(ocr_options=EasyOcrOptions())

        # OTIMIZAÇÃO: Modo rápido para documentos legais (majoritariamente texto)
        # force_backend_text=True: extrai texto do PDF sem layout analysis (GPU)
        # do_table_structure=False: desabilita análise de tabelas
        # Resultado: de 30+ min para ~30s em PDFs de 60 páginas
        if fast_mode:
            pipeline_options.force_backend_text = True
            pipeline_options.do_table_structure = False
            logger.info("Modo rápido: force_backend_text + sem análise de tabelas")

        # Tenta configurar aceleração GPU (pode não estar disponível em todas versões)
        try:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CUDA
            )
        except (ImportError, AttributeError) as e:
            logger.warning(f"Aceleração GPU não disponível: {e}")

        if enable_ocr:
            # OCR habilitado para PDFs com texto corrompido
            # IMPORTANTE: force_full_page_ocr deve ser passado para EasyOcrOptions
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = EasyOcrOptions(
                force_full_page_ocr=True,  # Ignora camada de texto, força OCR
                use_gpu=True,
                lang=["pt", "en"],  # Português e inglês
            )
            logger.info("Docling converter criado com OCR (force_full_page_ocr=True)")
        else:
            # Extração de texto nativo (rápido)
            pipeline_options.do_ocr = False
            logger.info("Docling converter criado sem OCR (texto nativo)")

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    @property
    def docling_converter(self):
        """Converter padrão (sem OCR) - para PDFs com texto nativo."""
        if self._docling_converter is None:
            self._docling_converter = self._create_docling_converter(enable_ocr=False)
        return self._docling_converter

    @property
    def docling_converter_ocr(self):
        """Converter com OCR - para PDFs com texto corrompido."""
        if self._docling_converter_ocr is None:
            self._docling_converter_ocr = self._create_docling_converter(enable_ocr=True)
        return self._docling_converter_ocr

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

    # === Acórdãos TCU ===
    _acordao_parser = None
    _acordao_chunker = None
    _chunk_enricher = None

    @property
    def chunk_enricher(self):
        """Enricher para adicionar contexto semântico aos chunks."""
        if self._chunk_enricher is None:
            from ..enrichment import ChunkEnricher
            self._chunk_enricher = ChunkEnricher(llm_client=self.llm_client)
            logger.info("ChunkEnricher inicializado")
        return self._chunk_enricher

    @property
    def acordao_parser(self):
        """Parser específico para acórdãos do TCU."""
        if self._acordao_parser is None:
            from ..parsing import AcordaoSpanParser
            self._acordao_parser = AcordaoSpanParser()
            logger.info("AcordaoSpanParser inicializado")
        return self._acordao_parser

    @property
    def acordao_chunker(self):
        """Chunker específico para acórdãos do TCU."""
        if self._acordao_chunker is None:
            from ..chunking import AcordaoChunker
            self._acordao_chunker = AcordaoChunker()
            logger.info("AcordaoChunker inicializado")
        return self._acordao_chunker

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

        # Cria um PDF mínimo para warmup
        logger.info("Criando PDF de warmup...")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_pdf = f.name
            c = canvas.Canvas(f.name, pagesize=letter)
            c.drawString(100, 750, "Art. 1 Documento de warmup.")
            c.drawString(100, 730, "I - inciso de teste;")
            c.drawString(100, 710, "II - outro inciso.")
            c.save()

        try:
            # Força inicialização do converter (carrega config, não modelos)
            logger.info("Inicializando Docling converter...")
            init_start = time.perf_counter()
            converter = self.docling_converter
            init_time = time.perf_counter() - init_start
            logger.info(f"Converter inicializado em {init_time:.2f}s")

            # Executa conversão para carregar modelos na GPU
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
            # Remove PDF temporário
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)

    def is_warmed_up(self) -> bool:
        """Verifica se o Docling está carregado."""
        return self._docling_converter is not None

    def process(
        self,
        pdf_content: bytes,
        request: IngestRequest,
        progress_callback=None,
    ) -> PipelineResult:
        """
        Processa um PDF e retorna chunks prontos para indexacao.

        Args:
            pdf_content: Conteudo binario do PDF
            request: Metadados do documento
            progress_callback: Funcao callback(phase: str, progress: float)
                               para reportar progresso (0.0 a 1.0)

        Returns:
            PipelineResult com chunks processados
        """
        def report_progress(phase: str, progress: float):
            if progress_callback:
                try:
                    progress_callback(phase, progress)
                except Exception as e:
                    logger.warning(f"Erro no progress_callback: {e}")

        start_time = time.perf_counter()
        result = PipelineResult(
            status=IngestStatus.PROCESSING,
            document_id=request.document_id,
        )
        report_progress("initializing", 0.05)

        # Calcula hash do documento
        result.document_hash = hashlib.sha256(pdf_content).hexdigest()

        try:
            # Salva PDF temporariamente
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_content)
                temp_path = f.name

            try:
                # Fase 1: Docling (PDF -> Markdown) - 10% a 40%
                report_progress("docling", 0.1)
                result = self._phase_docling(temp_path, result)
                if result.status == IngestStatus.FAILED:
                    return result
                report_progress("docling", 0.4)

                # Detecta se é acórdão TCU
                is_acordao = request.tipo_documento.upper() == "ACORDAO"

                if is_acordao:
                    # === PIPELINE ACÓRDÃO TCU ===
                    logger.info("Detectado tipo ACORDAO - usando pipeline específico")

                    # Fase 2A: AcordaoSpanParser - 40% a 55%
                    report_progress("parsing_acordao", 0.42)
                    parsed_acordao = self._phase_parsing_acordao(result, request)
                    if result.status == IngestStatus.FAILED:
                        return result
                    report_progress("parsing_acordao", 0.55)

                    # Fase 3A: AcordaoChunker (materializa) - 55% a 65%
                    report_progress("materialization_acordao", 0.55)
                    materialized = self._phase_materialization_acordao(
                        parsed_acordao, request, result
                    )
                    if result.status == IngestStatus.FAILED:
                        return result
                    report_progress("materialization_acordao", 0.65)

                    # Fase 4A: Enriquecimento com LLM - 65% a 75%
                    if not request.skip_enrichment:
                        report_progress("enrichment_acordao", 0.65)
                        self._phase_enrichment_acordao(materialized, request, result)
                        if result.status == IngestStatus.FAILED:
                            return result
                        report_progress("enrichment_acordao", 0.75)

                else:
                    # === PIPELINE LEIS/DECRETOS ===
                    # Fase 2: SpanParser (Markdown -> Spans) - 40% a 45%
                    report_progress("parsing", 0.42)
                    parsed_doc = self._phase_parsing(result)
                    if result.status == IngestStatus.FAILED:
                        return result
                    report_progress("parsing", 0.45)

                    # Fase 3: ArticleOrchestrator (LLM extraction) - 45% a 70%
                    report_progress("extraction", 0.45)
                    article_chunks = self._phase_extraction(parsed_doc, result, request.max_articles)
                    if result.status == IngestStatus.FAILED:
                        return result
                    report_progress("extraction", 0.70)

                    # Fase 4: ChunkMaterializer (parent-child chunks) - 70% a 75%
                    report_progress("materialization", 0.72)
                    materialized = self._phase_materialization(
                        article_chunks, parsed_doc, request, result
                    )
                    if result.status == IngestStatus.FAILED:
                        return result
                    report_progress("materialization", 0.75)

                # Fase 5: Embeddings (se nao pular) - 75% a 95%
                if not request.skip_embeddings:
                    report_progress("embedding", 0.75)
                    self._phase_embeddings(materialized, result)
                    if result.status == IngestStatus.FAILED:
                        return result
                    report_progress("embedding", 0.95)

                # Fase 6: Validacao de artigos (se habilitada ou sempre para gerar manifesto)
                if not is_acordao:
                    validator = ArticleValidator(
                        validate_enabled=request.validate_articles,
                        expected_first=request.expected_first_article,
                        expected_last=request.expected_last_article,
                    )
                    validation_result = validator.validate(materialized)
                    result.validation_docling = validation_result.to_dict()
                    logger.info(
                        f"Validacao de artigos: {validation_result.total_found} encontrados, "
                        f"{len(validation_result.missing_articles)} faltando, "
                        f"status={validation_result.status}"
                    )

                # Converte para ProcessedChunk - 95% a 100%
                report_progress("finalizing", 0.95)
                result.chunks = self._to_processed_chunks(materialized, request, result, is_acordao=is_acordao)
                result.status = IngestStatus.COMPLETED
                report_progress("completed", 1.0)

            finally:
                # Remove arquivo temporario
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
        """
        Fase 1: Docling - PDF -> Markdown

        Pipeline robusto:
        1. Tenta extração de texto nativo (rápido)
        2. Valida qualidade do texto extraído
        3. Se qualidade baixa, faz fallback para OCR
        4. Valida novamente após OCR
        """
        phase_start = time.perf_counter()

        try:
            # === TENTATIVA 1: Texto Nativo ===
            logger.info("Fase 1.1: Docling iniciando (texto nativo)...")
            doc_result = self.docling_converter.convert(pdf_path)
            markdown_content = doc_result.document.export_to_markdown()

            # Sanitiza markdown removendo anomalias (<!-- image -->, etc.)
            sanitizer = MarkdownSanitizer()
            markdown_content, sanitization_report = sanitizer.sanitize(markdown_content)
            if sanitization_report.anomalies_removed > 0:
                logger.info(
                    f"Fase 1.1b: Sanitização removeu {sanitization_report.anomalies_removed} anomalias: "
                    f"{', '.join(sanitization_report.changes_made)}"
                )

            # Valida qualidade do texto extraído
            quality_report = self._quality_validator.validate(markdown_content)
            logger.info(
                f"Fase 1.2: Qualidade do texto nativo: score={quality_report.score:.2f}, "
                f"readable={quality_report.readable_ratio:.1%}, "
                f"articles={quality_report.article_count}"
            )

            if quality_report.is_valid:
                # Texto nativo OK - continua pipeline
                result.markdown_content = markdown_content
                result.extraction_method = ExtractionMethod.NATIVE_TEXT
                result.quality_score = quality_report.score
                result.quality_issues = quality_report.issues

                duration = round(time.perf_counter() - phase_start, 2)
                result.phases.append({
                    "name": "docling",
                    "duration_seconds": duration,
                    "output": f"Extraido {len(markdown_content)} caracteres (texto nativo)",
                    "quality_score": quality_report.score,
                    "method": "native_text",
                    "success": True,
                })
                logger.info(f"Fase 1: Docling concluida em {duration}s (texto nativo OK)")
                return result

            # === TENTATIVA 2: OCR Fallback ===
            logger.warning(
                f"Fase 1.3: Qualidade baixa ({quality_report.score:.2f}), "
                f"issues: {quality_report.issues}. Tentando OCR..."
            )

            ocr_start = time.perf_counter()
            doc_result_ocr = self.docling_converter_ocr.convert(pdf_path)
            markdown_content_ocr = doc_result_ocr.document.export_to_markdown()
            ocr_duration = round(time.perf_counter() - ocr_start, 2)

            # Sanitiza markdown OCR
            markdown_content_ocr, sanitization_report_ocr = sanitizer.sanitize(markdown_content_ocr)
            if sanitization_report_ocr.anomalies_removed > 0:
                logger.info(
                    f"Fase 1.3b: Sanitização (OCR) removeu {sanitization_report_ocr.anomalies_removed} anomalias"
                )

            # Valida qualidade do OCR
            quality_report_ocr = self._quality_validator.validate(markdown_content_ocr)
            logger.info(
                f"Fase 1.4: Qualidade após OCR: score={quality_report_ocr.score:.2f}, "
                f"readable={quality_report_ocr.readable_ratio:.1%}, "
                f"articles={quality_report_ocr.article_count} (tempo: {ocr_duration}s)"
            )

            if quality_report_ocr.is_valid:
                # OCR OK - continua pipeline
                result.markdown_content = markdown_content_ocr
                result.extraction_method = ExtractionMethod.OCR_EASYOCR
                result.quality_score = quality_report_ocr.score
                result.quality_issues = quality_report_ocr.issues
                result.ocr_fallback_used = True

                duration = round(time.perf_counter() - phase_start, 2)
                result.phases.append({
                    "name": "docling",
                    "duration_seconds": duration,
                    "output": f"Extraido {len(markdown_content_ocr)} caracteres (OCR)",
                    "quality_score": quality_report_ocr.score,
                    "method": "ocr_easyocr",
                    "ocr_duration_seconds": ocr_duration,
                    "success": True,
                })
                logger.info(f"Fase 1: Docling concluida em {duration}s (OCR fallback)")
                return result

            # === FALHA: Nem texto nativo nem OCR funcionaram ===
            logger.error(
                f"Fase 1.5: FALHA - Texto nativo e OCR falharam. "
                f"Score texto nativo: {quality_report.score:.2f}, "
                f"Score OCR: {quality_report_ocr.score:.2f}"
            )

            # Usa o melhor dos dois (mesmo que ruim)
            if quality_report_ocr.score > quality_report.score:
                result.markdown_content = markdown_content_ocr
                result.extraction_method = ExtractionMethod.OCR_EASYOCR
                result.quality_score = quality_report_ocr.score
                result.quality_issues = quality_report_ocr.issues
                result.ocr_fallback_used = True
            else:
                result.markdown_content = markdown_content
                result.extraction_method = ExtractionMethod.NATIVE_TEXT
                result.quality_score = quality_report.score
                result.quality_issues = quality_report.issues

            duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "docling",
                "duration_seconds": duration,
                "output": f"Extraido {len(result.markdown_content)} caracteres (QUALIDADE BAIXA)",
                "quality_score": result.quality_score,
                "method": result.extraction_method.value,
                "success": True,  # Sucesso técnico, mas qualidade baixa
                "warning": "Qualidade abaixo do esperado - documento pode ter problemas",
            })

            # Adiciona aviso nos erros
            result.errors.append(IngestError(
                phase="docling",
                message=f"Qualidade de extração baixa ({result.quality_score:.2f}). "
                        f"Issues: {', '.join(result.quality_issues[:3])}",
            ))

            logger.warning(f"Fase 1: Docling concluida em {duration}s (QUALIDADE BAIXA)")

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

    def _phase_parsing_acordao(self, result: PipelineResult, request: IngestRequest):
        """Fase 2A: AcordaoSpanParser - Markdown -> ParsedAcordao"""
        phase_start = time.perf_counter()
        try:
            logger.info("Fase 2A: AcordaoSpanParser iniciando...")
            parsed_acordao = self.acordao_parser.parse(result.markdown_content)

            # Complementa metadados do request se disponíveis
            if request.colegiado and not parsed_acordao.metadata.colegiado:
                from ..parsing import parse_colegiado
                parsed_acordao.metadata.colegiado = parse_colegiado(request.colegiado)
            if request.processo and not parsed_acordao.metadata.processo:
                parsed_acordao.metadata.processo = request.processo
            if request.relator and not parsed_acordao.metadata.relator:
                parsed_acordao.metadata.relator = request.relator
            if request.data_sessao and not parsed_acordao.metadata.data_sessao:
                parsed_acordao.metadata.data_sessao = request.data_sessao
            if request.unidade_tecnica and not parsed_acordao.metadata.unidade_tecnica:
                parsed_acordao.metadata.unidade_tecnica = request.unidade_tecnica
            if request.unidade_jurisdicionada and not parsed_acordao.metadata.unidade_jurisdicionada:
                parsed_acordao.metadata.unidade_jurisdicionada = request.unidade_jurisdicionada
            if request.titulo and not parsed_acordao.metadata.titulo:
                parsed_acordao.metadata.titulo = request.titulo
            if request.numero:
                parsed_acordao.metadata.numero = int(request.numero)
            if request.ano:
                parsed_acordao.metadata.ano = request.ano

            duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "parsing_acordao",
                "duration_seconds": duration,
                "output": f"Encontrados {len(parsed_acordao.spans)} spans, acordao_id={parsed_acordao.acordao_id}",
                "success": True,
            })
            logger.info(f"Fase 2A: AcordaoSpanParser concluido em {duration}s - {len(parsed_acordao.spans)} spans")
            return parsed_acordao

        except Exception as e:
            logger.error(f"Erro no AcordaoSpanParser: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(phase="parsing_acordao", message=str(e)))
            return None

    def _phase_materialization_acordao(self, parsed_acordao, request: IngestRequest, result: PipelineResult):
        """Fase 3A: AcordaoChunker - ParsedAcordao -> MaterializedAcordaoChunk"""
        phase_start = time.perf_counter()
        try:
            logger.info("Fase 3A: AcordaoChunker iniciando...")

            materialized = self.acordao_chunker.materialize(
                parsed_acordao,
                document_hash=result.document_hash,
            )

            duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "materialization_acordao",
                "duration_seconds": duration,
                "output": f"Materializados {len(materialized)} chunks de acordao",
                "success": True,
            })
            logger.info(f"Fase 3A: AcordaoChunker concluido em {duration}s - {len(materialized)} chunks")
            return materialized

        except Exception as e:
            logger.error(f"Erro no AcordaoChunker: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(phase="materialization_acordao", message=str(e)))
            return None

    def _phase_enrichment_acordao(self, materialized, request: IngestRequest, result: PipelineResult):
        """Fase 4A: Enriquecimento de chunks de acordao com LLM."""
        phase_start = time.perf_counter()
        try:
            logger.info(f"Fase 4A: Enriquecimento de {len(materialized)} chunks de acordao...")

            from ..enrichment import DocumentMetadata

            # Cria metadados do documento para o enricher
            doc_meta = DocumentMetadata(
                document_id=request.document_id,
                document_type="ACORDAO",
                number=request.numero,
                year=request.ano,
                issuing_body="TCU",  # Tribunal de Contas da União
            )

            # Enriquece chunks (modifica in-place)
            self.chunk_enricher.apply_to_chunks(
                chunks=materialized,
                doc_meta=doc_meta,
                batch_size=5,  # Processa 5 chunks por chamada LLM
            )

            # Estatísticas do enricher
            stats = self.chunk_enricher.get_stats()

            duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "enrichment_acordao",
                "duration_seconds": duration,
                "output": f"Enriquecidos {len(materialized)} chunks de acordao",
                "enricher_stats": stats,
                "success": True,
            })
            logger.info(f"Fase 4A: Enriquecimento concluido em {duration}s - {stats.get('chunks_processed', 0)} processados")

        except Exception as e:
            logger.error(f"Erro no enriquecimento de acordao: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(phase="enrichment_acordao", message=str(e)))

    def _phase_extraction(self, parsed_doc, result: PipelineResult, max_articles: Optional[int] = None):
        """Fase 3: ArticleOrchestrator - LLM extraction"""
        phase_start = time.perf_counter()
        try:
            logger.info("Fase 3: ArticleOrchestrator iniciando...")
            extraction_result = self.orchestrator.extract_all_articles(
                parsed_doc
            )

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
            from ..chunking import ChunkMaterializer, ChunkMetadata, extract_offsets_from_parsed_doc

            metadata = ChunkMetadata(
                schema_version="1.0.0",
                extractor_version="1.0.0",
                ingestion_timestamp=datetime.utcnow().isoformat(),
                document_hash=result.document_hash,
            )

            # PR13: Extrai offsets canônicos do ParsedDocument
            offsets_map, canonical_hash = extract_offsets_from_parsed_doc(parsed_doc)
            logger.info(f"PR13: Extraídos offsets de {len(offsets_map)} spans (hash: {canonical_hash[:16]}...)")

            materializer = ChunkMaterializer(
                document_id=request.document_id,
                tipo_documento=request.tipo_documento,
                numero=request.numero,
                ano=request.ano,
                metadata=metadata,
                offsets_map=offsets_map,
                canonical_hash=canonical_hash,
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
                # Skip chunks marcados para nao indexar (pais de artigos splitados)
                if getattr(chunk, '_skip_milvus_index', False):
                    logger.debug(f"Skipping embedding for {chunk.chunk_id} (_skip_milvus_index=True)")
                    continue

                # retrieval_text determinístico para embeddings, fallback para text (legados)
                text_for_embedding = getattr(chunk, 'retrieval_text', '') or chunk.text
                # Usa o embedder do GPU server (ja carregado)
                embed_result = self.embedder.encode([text_for_embedding])

                chunk._dense_vector = embed_result.dense_embeddings[0]
                chunk._sparse_vector = embed_result.sparse_embeddings[0] if embed_result.sparse_embeddings else {}

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
        result: PipelineResult,
        is_acordao: bool = False
    ) -> List[ProcessedChunk]:
        """Converte MaterializedChunk ou MaterializedAcordaoChunk para ProcessedChunk (Pydantic)."""
        chunks = []
        skipped_count = 0
        for chunk in materialized:
            # Skip chunks marcados para nao indexar (pais de artigos splitados)
            if getattr(chunk, '_skip_milvus_index', False):
                skipped_count += 1
                continue

            if is_acordao:
                # MaterializedAcordaoChunk
                # Extrai citações do texto (remove self-loops e parent-loops)
                acordao_citations = extract_citations_from_chunk(
                    text=chunk.text or "",
                    document_id=request.document_id,
                    chunk_node_id=chunk.node_id,  # Remove self-loops
                    parent_chunk_id=chunk.parent_chunk_id if hasattr(chunk, "parent_chunk_id") else None,
                    document_type="ACORDAO",
                )

                # Prepara aliases para acórdão
                acordao_aliases = getattr(chunk, "aliases", "[]")
                acordao_sparse_source = getattr(chunk, "sparse_source", "") or chunk.enriched_text or chunk.text or ""

                pc = ProcessedChunk(
                    node_id=chunk.node_id,
                    chunk_id=chunk.chunk_id,
                    parent_chunk_id=chunk.parent_chunk_id or "",
                    span_id=chunk.span_id,
                    device_type=chunk.device_type,
                    chunk_level="acordao_span",  # Nível específico para acórdãos
                    text=chunk.text or "",
                    enriched_text=chunk.enriched_text or chunk.text or "",
                    context_header=chunk.context_header or "",
                    thesis_text=chunk.thesis_text or "",
                    thesis_type=chunk.thesis_type or "",
                    synthetic_questions=chunk.synthetic_questions or "",
                    document_id=request.document_id,
                    tipo_documento=request.tipo_documento,
                    numero=request.numero,
                    ano=request.ano,
                    article_number="",  # Acórdãos usam span_id
                    # Campos específicos de acórdão (NUNCA None - Milvus nullable=False)
                    colegiado=(getattr(chunk, "colegiado", None) or request.colegiado) or "",
                    processo=(getattr(chunk, "processo", None) or request.processo) or "",
                    relator=(getattr(chunk, "relator", None) or request.relator) or "",
                    data_sessao=(getattr(chunk, "data_sessao", None) or request.data_sessao) or "",
                    unidade_tecnica=(getattr(chunk, "unidade_tecnica", None) or request.unidade_tecnica) or "",
                    citations=acordao_citations,  # Citações extraídas do texto (sem self-loops)
                    aliases=acordao_aliases,
                    sparse_source=acordao_sparse_source,
                    # PR13: Offsets canônicos (zero fallback find)
                    canonical_start=getattr(chunk, 'canonical_start', -1),
                    canonical_end=getattr(chunk, 'canonical_end', -1),
                    canonical_hash=getattr(chunk, 'canonical_hash', ''),
                )
            else:
                # MaterializedChunk (leis/decretos)
                # Extrai citações normativas do texto do chunk
                chunk_citations = extract_citations_from_chunk(
                    text=chunk.text or "",
                    document_id=request.document_id,
                    chunk_node_id=chunk.node_id,  # Remove self-loops
                    parent_chunk_id=chunk.parent_chunk_id,  # Remove parent-loops
                    document_type=request.tipo_documento,
                )

                # Prepara aliases (converte lista para JSON string se necessário)
                chunk_aliases = getattr(chunk, "aliases", [])
                if isinstance(chunk_aliases, list):
                    import json
                    chunk_aliases = json.dumps(chunk_aliases)
                else:
                    chunk_aliases = chunk_aliases or ""

                # Prepara sparse_source
                chunk_sparse_source = getattr(chunk, "sparse_source", "")
                if not chunk_sparse_source:
                    # Usa enriched_text como fallback
                    chunk_sparse_source = chunk.enriched_text or chunk.text or ""

                pc = ProcessedChunk(
                    node_id=chunk.node_id,
                    chunk_id=chunk.chunk_id,
                    parent_chunk_id=chunk.parent_chunk_id or "",
                    span_id=chunk.span_id,
                    device_type=chunk.device_type.value if hasattr(chunk.device_type, "value") else str(chunk.device_type),
                    chunk_level=chunk.chunk_level.name.lower() if hasattr(chunk.chunk_level, "name") else str(chunk.chunk_level),
                    text=chunk.text or "",
                    parent_text=getattr(chunk, 'parent_text', '') or "",
                    retrieval_text=getattr(chunk, 'retrieval_text', '') or "",
                    enriched_text=chunk.enriched_text or "",  # Deprecated
                    context_header=chunk.context_header or "",  # Deprecated
                    thesis_text=chunk.thesis_text or "",
                    thesis_type=chunk.thesis_type or "",
                    synthetic_questions="",
                    document_id=request.document_id,
                    tipo_documento=request.tipo_documento,
                    numero=request.numero,
                    ano=request.ano,
                    article_number=chunk.article_number or "",
                    citations=chunk_citations,  # Lista de target_node_ids citados
                    aliases=chunk_aliases,
                    sparse_source=chunk_sparse_source,
                    # PR13: Offsets canônicos (zero fallback find)
                    canonical_start=getattr(chunk, 'canonical_start', -1),
                    canonical_end=getattr(chunk, 'canonical_end', -1),
                    canonical_hash=getattr(chunk, 'canonical_hash', ''),
                )

            # Adiciona vetores se foram gerados
            if hasattr(chunk, "_dense_vector"):
                pc.dense_vector = chunk._dense_vector
            if hasattr(chunk, "_sparse_vector"):
                pc.sparse_vector = chunk._sparse_vector
            if hasattr(chunk, "_thesis_vector"):
                pc.thesis_vector = chunk._thesis_vector

            chunks.append(pc)

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} parent chunks (_skip_milvus_index=True)")

        return chunks


# Singleton
_pipeline: Optional[IngestionPipeline] = None


def get_pipeline() -> IngestionPipeline:
    """Retorna instancia singleton do pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline
