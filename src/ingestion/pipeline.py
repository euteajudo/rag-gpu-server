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

import asyncio
import os
import time
import hashlib
import logging
import tempfile
import uuid
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .models import IngestRequest, ProcessedChunk, IngestStatus, IngestError
from .quality_validator import QualityValidator
from .markdown_sanitizer import MarkdownSanitizer
from .article_validator import ArticleValidator
from ..chunking.citation_extractor import extract_citations_from_chunk
from ..chunking.canonical_offsets import normalize_canonical_text, compute_canonical_hash
from ..chunking.origin_classifier import OriginClassifier


class ExtractionMethod(str, Enum):
    """Metodo usado para extrair o texto do PDF."""
    NATIVE_TEXT = "native_text"  # Texto nativo do PDF
    OCR_EASYOCR = "ocr_easyocr"  # OCR com EasyOCR
    OCR_TESSERACT = "ocr_tesseract"  # OCR com Tesseract

logger = logging.getLogger(__name__)


# =============================================================================
# INVARIANTES DO CONTRATO RUNPOD → VPS
# =============================================================================

class ContractViolationError(Exception):
    """Erro fatal: violação das invariantes do contrato RunPod → VPS."""
    pass


def validate_chunk_invariants(chunks: List["ProcessedChunk"], document_id: str) -> None:
    """
    Valida invariantes do contrato antes de retornar para VPS.

    Invariantes:
    1. node_id DEVE começar com "leis:" e NÃO conter "@P"
    2. parent_node_id (quando não vazio) DEVE começar com "leis:" e NÃO conter "@P"
    3. Filhos devem ter parent_node_id
    4. PR13 trio deve ser coerente (sentinela ou válido, nunca misturado)
    5. EVIDENCE CHUNKS (article/paragraph/inciso/alinea) DEVEM ter offsets válidos
       - Sentinela (-1,-1,"") é PROIBIDO para evidence
       - Evidence sem offset = snippet não determinístico = ABORT

    Raises:
        ContractViolationError: Se qualquer invariante for violada (aborta pipeline)
    """
    violations = []

    # Device types que são "evidence" (aparecem no Evidence Drawer)
    EVIDENCE_DEVICE_TYPES = {"article", "paragraph", "inciso", "alinea"}

    for i, chunk in enumerate(chunks):
        node_id = chunk.node_id or ""
        parent_node_id = chunk.parent_node_id or ""
        device_type = chunk.device_type or ""
        span_id = chunk.span_id or ""
        is_evidence = device_type in EVIDENCE_DEVICE_TYPES

        # Invariante 1: node_id deve começar com "leis:"
        if not node_id.startswith("leis:"):
            violations.append(
                f"[{span_id}] node_id sem prefixo 'leis:': '{node_id}'"
            )

        # Invariante 2: node_id não pode conter @Pxx
        if "@P" in node_id:
            violations.append(
                f"[{span_id}] node_id contém @Pxx (proibido): '{node_id}'"
            )

        # Invariante 3: parent_node_id (quando não vazio) deve começar com "leis:"
        if parent_node_id and not parent_node_id.startswith("leis:"):
            violations.append(
                f"[{span_id}] parent_node_id sem prefixo 'leis:': '{parent_node_id}'"
            )

        # Invariante 4: parent_node_id não pode conter @Pxx
        if "@P" in parent_node_id:
            violations.append(
                f"[{span_id}] parent_node_id contém @Pxx (proibido): '{parent_node_id}'"
            )

        # Invariante 5: filhos (não-artigos) devem ter parent_node_id não vazio
        if device_type in ("paragraph", "inciso", "alinea") and not parent_node_id:
            violations.append(
                f"[{span_id}] {device_type} sem parent_node_id (obrigatório para filhos)"
            )

        # Invariante 6: PR13 trio - campos devem existir e ser coerentes
        c_start = getattr(chunk, 'canonical_start', None)
        c_end = getattr(chunk, 'canonical_end', None)
        c_hash = getattr(chunk, 'canonical_hash', None)

        # Campos devem existir
        if c_start is None:
            violations.append(f"[{span_id}] canonical_start ausente")
        if c_end is None:
            violations.append(f"[{span_id}] canonical_end ausente")
        if c_hash is None:
            violations.append(f"[{span_id}] canonical_hash ausente")

        # Se campos existem, validar coerência do trio
        if c_start is not None and c_end is not None and c_hash is not None:
            is_sentinel = (c_start == -1 and c_end == -1 and c_hash == "")
            is_valid_offset = (c_start >= 0 and c_end > c_start and c_hash != "")

            # Trio deve ser sentinela OU válido, nunca misturado
            if not is_sentinel and not is_valid_offset:
                violations.append(
                    f"[{span_id}] PR13 trio incoerente: start={c_start}, end={c_end}, "
                    f"hash={repr(c_hash[:16]) if c_hash else repr('')} "
                    f"(esperado: sentinela [-1,-1,''] ou válido [>=0,>start,hash])"
                )

            # INVARIANTE 7 (CRÍTICA): Evidence chunks DEVEM ter offsets válidos
            # Sentinela é PROIBIDO para evidence - snippet não seria determinístico
            if is_evidence and is_sentinel:
                violations.append(
                    f"[{span_id}] EVIDENCE SEM OFFSET: document_id={document_id}, "
                    f"node_id={node_id}, device_type={device_type} - "
                    f"Evidence requer offsets válidos para snippet determinístico. "
                    f"Sentinela (-1,-1,'') é PROIBIDO para evidence chunks."
                )

    if violations:
        # Contagem por tipo de violação para diagnóstico
        evidence_offset_violations = [v for v in violations if "EVIDENCE SEM OFFSET" in v]
        other_violations = [v for v in violations if "EVIDENCE SEM OFFSET" not in v]

        error_msg = (
            f"ABORT: {len(violations)} violações de contrato no documento '{document_id}':\n"
        )

        if evidence_offset_violations:
            error_msg += f"\n  ⛔ {len(evidence_offset_violations)} chunks evidence SEM OFFSETS VÁLIDOS:\n"
            for v in evidence_offset_violations[:5]:
                error_msg += f"     • {v}\n"
            if len(evidence_offset_violations) > 5:
                error_msg += f"     ... e mais {len(evidence_offset_violations) - 5} chunks evidence sem offset\n"
            error_msg += "\n  💡 CAUSA PROVÁVEL: SpanParser não fornece start_pos/end_pos para filhos (PAR/INC/ALI)\n"
            error_msg += "  💡 SOLUÇÃO: Garantir que parsed_doc.spans inclui offsets para TODOS os spans\n"

        if other_violations:
            error_msg += f"\n  ⚠️  Outras violações ({len(other_violations)}):\n"
            for v in other_violations[:5]:
                error_msg += f"     • {v}\n"
            if len(other_violations) > 5:
                error_msg += f"     ... e mais {len(other_violations) - 5}\n"

        logger.error(error_msg)
        raise ContractViolationError(error_msg)

    logger.info(f"✓ Invariantes validadas: {len(chunks)} chunks evidence com offsets OK para '{document_id}'")


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
    # PR13: Ingest run ID para rastreabilidade
    ingest_run_id: str = ""
    # PR13: Canonical hash para validação de offsets
    canonical_hash: str = ""
    # OriginClassifier: Estatísticas de classificação de origem
    origin_stats: Optional[dict] = None


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

    def _create_docling_converter(self, enable_ocr: bool = False, fast_mode: bool = False):
        """
        Cria um DocumentConverter do Docling.

        Args:
            enable_ocr: Se True, habilita OCR para PDFs com texto corrompido
            fast_mode: Se True, desabilita análise de tabelas (muito mais rápido)

        Referência: https://docling-project.github.io/docling/examples/full_page_ocr/

        ⚠️  AVISO IMPORTANTE - NÃO HABILITAR fast_mode ⚠️
        ═══════════════════════════════════════════════════════════════════════════
        O fast_mode (force_backend_text=True) foi DESABILITADO intencionalmente
        após investigação detalhada de erros de ingestão em fevereiro de 2026.

        PROBLEMA IDENTIFICADO:
        Com force_backend_text=True, o Docling extrai texto "raw" do PDF sem
        análise de layout. Isso causa:

        1. PERDA DE QUEBRAS DE LINHA: Artigos, parágrafos e incisos ficam
           concatenados em uma única linha, ex:
           "Art. 45. O licenciamento... I - disposição final... II - mitigação..."

        2. FALHA NO SpanParser: Os regex do SpanParser dependem de quebras de
           linha para detectar início de artigos (^Art\.) e estruturas legais.
           Sem quebras, a detecção falha ou produz offsets incorretos.

        3. OffsetResolutionError: Chunks ficam com offsets inválidos, causando
           erro "Chunk 'INC-XXX-XX' não encontrado no range do pai" durante
           a materialização no ChunkMaterializer.

        TRADE-OFF DE PERFORMANCE:
        - fast_mode=True:  ~30 segundos para PDF de 60 páginas
        - fast_mode=False: ~5-10 minutos para PDF de 60 páginas

        A perda de qualidade na extração é INACEITÁVEL para documentos legais
        onde a estrutura hierárquica (artigos > parágrafos > incisos > alíneas)
        é crítica para o funcionamento do RAG.

        REFERÊNCIA: Documentação Docling v2.67+
        - force_backend_text bypasses layout analysis entirely
        - Recommended only for simple text extraction, not structured documents
        ═══════════════════════════════════════════════════════════════════════════
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

        # ⚠️ MODO RÁPIDO - DESABILITADO POR PADRÃO (ver docstring acima)
        # fast_mode=True causa perda de quebras de linha e falha no SpanParser
        # Mantido apenas para casos excepcionais onde qualidade não é crítica
        if fast_mode:
            pipeline_options.force_backend_text = True
            pipeline_options.do_table_structure = False
            logger.warning("⚠️ FAST_MODE ATIVADO - Qualidade de extração reduzida!")
        else:
            # Modo padrão: layout analysis completo para preservar estrutura
            pipeline_options.do_table_structure = False  # Tabelas ainda desabilitadas (não usamos)
            logger.info("Modo padrão: layout analysis habilitado para estrutura legal")

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

    # === Artifacts Uploader (PR13/Etapa 4) ===
    _artifacts_uploader = None

    @property
    def artifacts_uploader(self):
        """Uploader para enviar artifacts para a VPS."""
        if self._artifacts_uploader is None:
            from ..sinks.artifacts_uploader import get_artifacts_uploader
            self._artifacts_uploader = get_artifacts_uploader()
        return self._artifacts_uploader

    # === VLM Pipeline (PyMuPDF + Qwen3-VL) ===
    _vlm_service = None

    @property
    def vlm_service(self):
        """VLM Extraction Service - lazy loaded."""
        if self._vlm_service is None:
            from ..config import config
            from ..extraction.pymupdf_extractor import PyMuPDFExtractor
            from ..extraction.vlm_client import VLMClient
            from ..extraction.vlm_service import VLMExtractionService

            vlm_client = VLMClient(
                base_url=config.vllm_base_url,
                model=config.vllm_model,
                max_retries=config.vlm_max_retries,
            )
            pymupdf_extractor = PyMuPDFExtractor(dpi=config.vlm_page_dpi)
            self._vlm_service = VLMExtractionService(
                vlm_client=vlm_client,
                pymupdf_extractor=pymupdf_extractor,
            )
            logger.info("VLMExtractionService inicializado")
        return self._vlm_service

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

        # Gera ingest_run_id único para rastreabilidade
        ingest_run_id = str(uuid.uuid4())

        result = PipelineResult(
            status=IngestStatus.PROCESSING,
            document_id=request.document_id,
            ingest_run_id=ingest_run_id,
        )
        report_progress("initializing", 0.05)

        # Calcula hash do documento (SHA256 do PDF original)
        result.document_hash = hashlib.sha256(pdf_content).hexdigest()

        try:
            # Salva PDF temporariamente
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_content)
                temp_path = f.name

            try:
                # === Feature flag: VLM Pipeline vs Legacy Pipeline ===
                from ..config import config as app_config
                if app_config.use_vlm_pipeline:
                    # === PIPELINE VLM: PyMuPDF + Qwen3-VL ===
                    logger.info(f"Pipeline VLM ativo para {request.document_id}")
                    report_progress("vlm_extraction", 0.10)

                    vlm_result = self._phase_vlm_extraction(
                        pdf_content, request, result, report_progress
                    )
                    if result.status == IngestStatus.FAILED:
                        return result

                    result.total_time_seconds = round(time.perf_counter() - start_time, 2)
                    return result

                # === PIPELINE LEGADO: Docling + SpanParser ===

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

                    # Fase 2.5: Upload de Artifacts para VPS (PR13/Etapa 4) - 45% a 47%
                    report_progress("artifacts_upload", 0.45)
                    artifacts_result = self._phase_artifacts_upload(
                        pdf_content=pdf_content,
                        parsed_doc=parsed_doc,
                        request=request,
                        result=result,
                    )
                    if result.status == IngestStatus.FAILED:
                        return result
                    report_progress("artifacts_upload", 0.47)

                    # Fase 3: ArticleOrchestrator (LLM extraction) - 47% a 70%
                    report_progress("extraction", 0.47)
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

                    # Fase 4.5: OriginClassifier (classificacao de origem material)
                    origin_stats = self._phase_origin_classification(materialized, result)
                    result.origin_stats = origin_stats  # Salva stats no resultado

                # Fase 5: Embeddings (se nao pular) - 75% a 90%
                if not request.skip_embeddings:
                    report_progress("embedding", 0.76)
                    self._phase_embeddings(materialized, result)
                    if result.status == IngestStatus.FAILED:
                        return result
                    report_progress("embedding", 0.90)

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

    def _phase_vlm_extraction(
        self,
        pdf_content: bytes,
        request: IngestRequest,
        result: PipelineResult,
        report_progress,
    ) -> None:
        """
        Pipeline VLM completo: PyMuPDF + Qwen3-VL -> ProcessedChunks.

        Substitui Docling + SpanParser + ArticleOrchestrator + ChunkMaterializer
        com extração via visão computacional.

        Args:
            pdf_content: Bytes do PDF original
            request: Metadados do documento
            result: PipelineResult em construção
            report_progress: Callback de progresso
        """
        phase_start = time.perf_counter()

        try:
            # Roda o pipeline VLM assíncrono no event loop
            # O process() é chamado de um thread background, então
            # precisamos criar/obter um event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Estamos em um thread com event loop rodando
                    # Cria um novo loop para este thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        extraction = pool.submit(
                            asyncio.run,
                            self.vlm_service.extract_document(
                                pdf_bytes=pdf_content,
                                document_id=request.document_id,
                                progress_callback=report_progress,
                            ),
                        ).result()
                else:
                    extraction = loop.run_until_complete(
                        self.vlm_service.extract_document(
                            pdf_bytes=pdf_content,
                            document_id=request.document_id,
                            progress_callback=report_progress,
                        )
                    )
            except RuntimeError:
                # Nenhum event loop — cria um novo
                extraction = asyncio.run(
                    self.vlm_service.extract_document(
                        pdf_bytes=pdf_content,
                        document_id=request.document_id,
                        progress_callback=report_progress,
                    )
                )

            report_progress("vlm_extraction", 0.80)

            # Registra fase VLM
            vlm_duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "vlm_extraction",
                "duration_seconds": vlm_duration,
                "output": (
                    f"Extraídos {extraction.total_devices} dispositivos de "
                    f"{len(extraction.pages)} páginas via PyMuPDF+Qwen3-VL"
                ),
                "success": True,
                "method": "pymupdf+qwen3vl",
            })

            # Armazena canonical para uso posterior
            result.markdown_content = extraction.canonical_text
            result.canonical_hash = extraction.canonical_hash

            # === Converte DocumentExtraction -> list[ProcessedChunk] ===
            report_progress("vlm_materialization", 0.85)

            chunks = self._vlm_to_processed_chunks(extraction, request, result)

            report_progress("vlm_materialization", 0.90)

            # === Embeddings (se não pular) ===
            if not request.skip_embeddings:
                report_progress("embedding", 0.90)
                # Cria objetos compatíveis para embedding
                for chunk in chunks:
                    text_for_embedding = chunk.retrieval_text or chunk.text
                    embed_result = self.embedder.encode([text_for_embedding])
                    chunk.dense_vector = embed_result.dense_embeddings[0]
                    chunk.sparse_vector = (
                        embed_result.sparse_embeddings[0]
                        if embed_result.sparse_embeddings
                        else {}
                    )
                report_progress("embedding", 0.95)

                result.phases.append({
                    "name": "embedding",
                    "duration_seconds": round(time.perf_counter() - phase_start - vlm_duration, 2),
                    "output": f"Embeddings para {len(chunks)} chunks VLM",
                    "success": True,
                })

            result.chunks = chunks
            result.status = IngestStatus.COMPLETED
            report_progress("completed", 1.0)

            logger.info(
                f"Pipeline VLM concluído: {len(chunks)} chunks, "
                f"{extraction.total_devices} dispositivos"
            )

        except Exception as e:
            logger.error(f"Erro no pipeline VLM: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(
                phase="vlm_extraction",
                message=str(e),
            ))

    def _vlm_to_processed_chunks(
        self,
        extraction,
        request: IngestRequest,
        result: PipelineResult,
    ) -> List[ProcessedChunk]:
        """
        Converte DocumentExtraction (VLM) para list[ProcessedChunk].

        Mapeia os dispositivos extraídos pelo VLM para o formato ProcessedChunk
        esperado pelo pipeline downstream (VPS, Milvus, etc.).
        """
        from ..chunking.citation_extractor import extract_citations_from_chunk

        chunks = []

        for page in extraction.pages:
            for device in page.devices:
                # Gera span_id baseado no tipo e identificador
                span_id = self._device_to_span_id(device)

                # node_id e chunk_id
                chunk_id = f"{request.document_id}#{span_id}"
                node_id = f"leis:{chunk_id}"

                # parent_node_id
                parent_node_id = ""
                if device.parent_identifier and device.device_type != "artigo":
                    parent_span_id = self._identifier_to_span_id(
                        device.parent_identifier, device.device_type
                    )
                    parent_node_id = f"leis:{request.document_id}#{parent_span_id}"

                # device_type normalizado
                device_type_map = {
                    "artigo": "article",
                    "paragrafo": "paragraph",
                    "inciso": "inciso",
                    "alinea": "alinea",
                }
                dt = device_type_map.get(device.device_type, device.device_type)

                # chunk_level
                chunk_level = "article" if dt == "article" else "device"

                # Citações
                citations = extract_citations_from_chunk(
                    text=device.text or "",
                    document_id=request.document_id,
                    chunk_node_id=node_id,
                    parent_chunk_id=parent_node_id or None,
                    document_type=request.tipo_documento,
                )

                pc = ProcessedChunk(
                    node_id=node_id,
                    chunk_id=chunk_id,
                    parent_node_id=parent_node_id,
                    span_id=span_id,
                    device_type=dt,
                    chunk_level=chunk_level,
                    text=device.text or "",
                    parent_text="",
                    retrieval_text=device.text or "",
                    document_id=request.document_id,
                    tipo_documento=request.tipo_documento,
                    numero=request.numero,
                    ano=request.ano,
                    article_number=self._extract_article_number(device),
                    citations=citations,
                    # PR13: canonical offsets (sentinela por enquanto — reconciliator futuro)
                    canonical_start=-1,
                    canonical_end=-1,
                    canonical_hash=extraction.canonical_hash,
                    # VLM: campos novos
                    page_number=page.page_number,
                    bbox=device.bbox,
                    confidence=device.confidence,
                    # Campos Acórdão (vazios para LEI/DECRETO)
                    colegiado="",
                    processo="",
                    relator="",
                    data_sessao="",
                    unidade_tecnica="",
                )
                chunks.append(pc)

        logger.info(f"VLM -> ProcessedChunk: {len(chunks)} chunks gerados")
        return chunks

    @staticmethod
    def _device_to_span_id(device) -> str:
        """Converte DeviceExtraction para span_id (ex: ART-005, PAR-005-1)."""
        import re

        identifier = device.identifier.strip()
        dtype = device.device_type.lower()

        if dtype == "artigo":
            # "Art. 5º" -> "ART-005"
            match = re.search(r"(\d+)", identifier)
            num = match.group(1) if match else "000"
            return f"ART-{num.zfill(3)}"

        elif dtype == "paragrafo":
            # "§ 1º" -> precisa do pai para montar PAR-005-1
            match = re.search(r"(\d+)", identifier)
            num = match.group(1) if match else "0"
            # Tenta extrair número do artigo pai
            parent_id = device.parent_identifier or ""
            parent_match = re.search(r"(\d+)", parent_id)
            parent_num = parent_match.group(1) if parent_match else "000"
            return f"PAR-{parent_num.zfill(3)}-{num}"

        elif dtype == "inciso":
            # "I" / "II" -> INC-005-1
            parent_id = device.parent_identifier or ""
            parent_match = re.search(r"(\d+)", parent_id)
            parent_num = parent_match.group(1) if parent_match else "000"
            # Converte romano ou numérico
            inc_num = IngestionPipeline._roman_to_int(identifier.strip().rstrip(").-"))
            return f"INC-{parent_num.zfill(3)}-{inc_num}"

        elif dtype == "alinea":
            # "a)" -> ALI-005-3-a (parent pode ser inciso romano)
            parent_id = device.parent_identifier or ""
            parent_match = re.search(r"(\d+)", parent_id)
            if parent_match:
                parent_num = parent_match.group(1).zfill(3)
            else:
                # Parent é inciso romano (ex: "III")
                parent_num = str(IngestionPipeline._roman_to_int(parent_id.strip()))
            letter = re.search(r"([a-z])", identifier.lower())
            letter_str = letter.group(1) if letter else "a"
            return f"ALI-{parent_num}-{letter_str}"

        return f"DEV-{identifier[:20]}"

    @staticmethod
    def _identifier_to_span_id(identifier: str, child_type: str) -> str:
        """Converte parent_identifier para span_id do pai."""
        import re
        match = re.search(r"(\d+)", identifier)
        num = match.group(1) if match else "000"

        # Detecta tipo do pai pelo identificador
        ident_lower = identifier.lower().strip()
        if ident_lower.startswith("art"):
            return f"ART-{num.zfill(3)}"
        elif "§" in identifier or ident_lower.startswith("par"):
            parent_match = re.search(r"(\d+)", identifier)
            return f"PAR-{num.zfill(3)}-{parent_match.group(1) if parent_match else '0'}"
        else:
            return f"ART-{num.zfill(3)}"

    @staticmethod
    def _roman_to_int(s: str) -> int:
        """Converte numeral romano para inteiro."""
        roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
        s = s.upper().strip()
        result = 0
        for i, c in enumerate(s):
            if c not in roman:
                # Tenta como número decimal
                try:
                    return int(s)
                except ValueError:
                    return 0
            if i + 1 < len(s) and roman.get(c, 0) < roman.get(s[i + 1], 0):
                result -= roman[c]
            else:
                result += roman[c]
        return result

    @staticmethod
    def _extract_article_number(device) -> str:
        """Extrai número do artigo de um DeviceExtraction."""
        import re
        if device.device_type == "artigo":
            match = re.search(r"(\d+)", device.identifier)
            return match.group(1) if match else ""
        elif device.parent_identifier:
            match = re.search(r"(\d+)", device.parent_identifier)
            return match.group(1) if match else ""
        return ""

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

    def _phase_artifacts_upload(
        self,
        pdf_content: bytes,
        parsed_doc,
        request: IngestRequest,
        result: PipelineResult,
    ) -> bool:
        """
        Fase 2.5: Upload de Artifacts para VPS (PR13/Etapa 4).

        Envia PDF, canonical.md e offsets.json para a VPS antes de
        continuar com o pipeline. Isso garante que as evidências
        estejam armazenadas antes dos chunks irem para o Milvus.

        Se o upload falhar, o pipeline é abortado para evitar
        chunks órfãos no Milvus sem evidência correspondente.

        Args:
            pdf_content: Bytes do PDF original
            parsed_doc: ParsedDocument com source_text (canonical markdown)
            request: Metadados do documento
            result: PipelineResult em construção

        Returns:
            True se upload OK ou uploader não configurado, False se falhou
        """
        phase_start = time.perf_counter()

        try:
            # Verifica se uploader está configurado
            uploader = self.artifacts_uploader
            if not uploader.is_configured():
                logger.info("Fase 2.5: Artifacts uploader não configurado, pulando upload")
                result.phases.append({
                    "name": "artifacts_upload",
                    "duration_seconds": 0.0,
                    "output": "Skipped (uploader não configurado)",
                    "success": True,
                    "skipped": True,
                })
                return True

            logger.info("Fase 2.5: Artifacts upload iniciando...")

            # Extrai canonical markdown e normaliza
            canonical_md = parsed_doc.source_text or result.markdown_content
            canonical_md_normalized = normalize_canonical_text(canonical_md)

            # Computa hashes
            sha256_canonical = compute_canonical_hash(canonical_md_normalized)
            result.canonical_hash = sha256_canonical

            # Extrai offsets de todos os spans
            from ..chunking.canonical_offsets import extract_offsets_from_parsed_doc
            offsets_map, _ = extract_offsets_from_parsed_doc(parsed_doc)

            # Converte offsets para formato JSON-serializable
            from ..sinks.artifacts_uploader import (
                ArtifactMetadata,
                prepare_offsets_map,
                compute_sha256,
            )

            offsets_json = prepare_offsets_map(offsets_map)

            # Prepara metadados
            # document_version como timestamp Unix (inteiro)
            metadata = ArtifactMetadata(
                document_id=request.document_id,
                tipo_documento=request.tipo_documento,
                numero=request.numero,
                ano=request.ano,
                sha256_source=result.document_hash,
                sha256_canonical_md=sha256_canonical,
                canonical_hash=sha256_canonical,
                ingest_run_id=result.ingest_run_id,
                pipeline_version="1.0.0",
                document_version=str(int(datetime.utcnow().timestamp())),
            )

            # Faz upload
            upload_result = uploader.upload(
                pdf_content=pdf_content,
                canonical_md=canonical_md_normalized,
                offsets_json=offsets_json,
                metadata=metadata,
            )

            duration = round(time.perf_counter() - phase_start, 2)

            if upload_result.success:
                result.phases.append({
                    "name": "artifacts_upload",
                    "duration_seconds": duration,
                    "output": f"Artifacts uploaded: {upload_result.message}",
                    "storage_paths": upload_result.storage_paths,
                    "retries": upload_result.retries,
                    "success": True,
                })
                logger.info(
                    f"Fase 2.5: Artifacts upload concluido em {duration}s "
                    f"(retries={upload_result.retries})"
                )
                return True
            else:
                # Upload falhou - continua pipeline com warning (não aborta mais)
                result.phases.append({
                    "name": "artifacts_upload",
                    "duration_seconds": duration,
                    "output": f"WARNING: {upload_result.error}",
                    "retries": upload_result.retries,
                    "success": False,
                })
                logger.warning(
                    f"Fase 2.5: Artifacts upload FALHOU em {duration}s - "
                    f"continuando pipeline (chunks serão inseridos no Milvus)"
                )
                # Retorna True para continuar o pipeline
                return True

        except Exception as e:
            duration = round(time.perf_counter() - phase_start, 2)
            logger.warning(f"Erro no artifacts upload (continuando): {e}")
            result.phases.append({
                "name": "artifacts_upload",
                "duration_seconds": duration,
                "output": f"WARNING: {str(e)}",
                "success": False,
            })
            # Retorna True para continuar o pipeline
            return True

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
            from ..chunking.canonical_offsets import normalize_canonical_text

            metadata = ChunkMetadata(
                schema_version="1.0.0",
                extractor_version="1.0.0",
                ingestion_timestamp=datetime.utcnow().isoformat(),
                document_hash=result.document_hash,
            )

            # PR13: Extrai offsets canônicos do ParsedDocument
            offsets_map, canonical_hash = extract_offsets_from_parsed_doc(parsed_doc)
            logger.info(f"PR13: Extraídos offsets de {len(offsets_map)} spans (hash: {canonical_hash[:16]}...)")

            # PR13 STRICT: Normaliza canonical_text para resolução determinística de offsets
            source_text = getattr(parsed_doc, 'source_text', '') or ''
            canonical_text = normalize_canonical_text(source_text)
            logger.info(f"PR13: canonical_text com {len(canonical_text)} chars para resolução de offsets")

            materializer = ChunkMaterializer(
                document_id=request.document_id,
                tipo_documento=request.tipo_documento,
                numero=request.numero,
                ano=request.ano,
                metadata=metadata,
                offsets_map=offsets_map,
                canonical_hash=canonical_hash,
                canonical_text=canonical_text,
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

    def _phase_origin_classification(self, materialized, result: PipelineResult) -> dict:
        """
        Fase 4.5: Classificacao de origem material dos chunks.

        Detecta chunks que contem material de outras leis citadas/modificadas,
        permitindo tratamento diferenciado no retrieval.

        Exemplo: Art. 337-E esta no PDF da Lei 14.133 mas E do Codigo Penal.

        Args:
            materialized: Lista de MaterializedChunk
            result: PipelineResult para registrar a fase

        Returns:
            dict com estatisticas de classificacao
        """
        phase_start = time.perf_counter()
        try:
            logger.info("Fase 4.5: OriginClassifier iniciando...")

            classifier = OriginClassifier()
            stats = classifier.classify_materialized_batch(materialized)

            duration = round(time.perf_counter() - phase_start, 3)

            # Monta output descritivo
            if stats["external"] > 0:
                refs_str = ", ".join(f"{k}:{v}" for k, v in stats["external_refs"].items())
                output = f"{stats['self']} self, {stats['external']} external ({refs_str})"
            else:
                output = f"{stats['total']} chunks (todos self)"

            result.phases.append({
                "name": "origin_classification",
                "duration_seconds": duration,
                "output": output,
                "success": True,
                "stats": stats,
            })

            logger.info(f"Fase 4.5: OriginClassifier concluido em {duration}s - {output}")
            return stats

        except Exception as e:
            logger.error(f"Erro no OriginClassifier: {e}", exc_info=True)
            # Nao falha o pipeline, apenas loga o erro
            result.phases.append({
                "name": "origin_classification",
                "duration_seconds": 0,
                "output": f"Erro: {str(e)}",
                "success": False,
            })
            return {"total": len(materialized), "self": len(materialized), "external": 0, "external_refs": {}}

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

    def _phase_milvus_sink(self, materialized, request: IngestRequest, result: PipelineResult):
        """Fase 5.5: Inserir chunks no Milvus remoto (se MILVUS_HOST configurado)."""
        phase_start = time.perf_counter()
        try:
            from pymilvus import connections, Collection

            milvus_host = os.getenv("MILVUS_HOST", "127.0.0.1")
            milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
            collection_name = os.getenv("MILVUS_COLLECTION", "leis_v4")

            logger.info(f"Fase 5.5: Milvus Sink iniciando ({milvus_host}:{milvus_port}/{collection_name})...")

            # Conecta ao Milvus
            connections.connect(
                alias="ingest",
                host=milvus_host,
                port=milvus_port,
            )
            collection = Collection(collection_name, using="ingest")
            collection.load()

            # Prepara dados no formato do schema leis_v4
            data_list = []
            for chunk in materialized:
                # Skip chunks marcados para nao indexar
                if getattr(chunk, '_skip_milvus_index', False):
                    continue

                chunk_id = chunk.chunk_id
                # Adiciona @P00 se não tiver parte
                if '@P' not in chunk_id:
                    node_id_base = f"{chunk_id}@P00"
                else:
                    node_id_base = chunk_id
                # Adiciona prefixo leis: para o node_id físico
                node_id = f"leis:{node_id_base}"
                # logical_node_id é sem @Pxx mas com prefixo leis:
                logical_node_id_base = node_id_base.split('@P')[0] if '@P' in node_id_base else node_id_base
                logical_node_id = f"leis:{logical_node_id_base}"

                # PR13: parent_node_id já vem pronto do MaterializedChunk
                # Formato: "leis:DOC#SPAN_ID" para filhos, "" para artigos
                parent_node_id = getattr(chunk, 'parent_node_id', '') or ''

                # Extrai citações para has_citations e citations_count
                citations = getattr(chunk, 'citations', []) or []
                has_citations = len(citations) > 0
                citations_count = len(citations)

                # Device type - manter como "article", "paragraph", etc. conforme schema
                device_type = getattr(chunk, 'device_type', 'article')

                # Prepara aliases como JSON string
                aliases = getattr(chunk, 'aliases', []) or []
                import json
                aliases_str = json.dumps(aliases, ensure_ascii=False)

                data = {
                    "node_id": node_id,
                    "logical_node_id": logical_node_id,
                    "span_id": getattr(chunk, 'span_id', ''),
                    "parent_node_id": parent_node_id,
                    "device_type": device_type,
                    "chunk_level": getattr(chunk, 'chunk_level', 'article'),
                    "part_index": 0,
                    "part_total": 1,
                    "chunk_id": chunk_id,
                    "ingest_run_id": result.ingest_run_id or '',
                    "text": chunk.text,
                    "retrieval_text": getattr(chunk, 'retrieval_text', chunk.text),
                    "document_id": request.document_id,
                    "tipo_documento": request.tipo_documento,
                    "numero": request.numero,
                    "ano": request.ano,
                    "article_number": getattr(chunk, 'article_number', '') or '',
                    "aliases": aliases_str,
                    "canonical_start": getattr(chunk, 'canonical_start', -1),
                    "canonical_end": getattr(chunk, 'canonical_end', -1),
                    "canonical_hash": getattr(chunk, 'canonical_hash', '') or result.canonical_hash or '',
                    "dense_vector": getattr(chunk, '_dense_vector', [0.0] * 1024),
                    "sparse_vector": getattr(chunk, '_sparse_vector', {}),
                    "has_citations": has_citations,
                    "citations_count": citations_count,
                    # VLM: Campos do pipeline Qwen3-VL + PyMuPDF
                    "page_number": getattr(chunk, 'page_number', -1),
                    "bbox": json.dumps(getattr(chunk, 'bbox', []), ensure_ascii=False),
                    "confidence": float(getattr(chunk, 'confidence', 0.0)),
                }
                data_list.append(data)

            # Insere em batch
            if data_list:
                # Delete existentes
                node_ids = [d["node_id"] for d in data_list]
                ids_str = ", ".join([f'"{nid}"' for nid in node_ids[:100]])  # Limita para evitar query muito longa
                try:
                    collection.delete(expr=f"node_id in [{ids_str}]")
                except Exception as del_err:
                    logger.warning(f"Erro ao deletar chunks existentes: {del_err}")

                # Insert batch
                collection.insert(data_list)
                inserted = len(data_list)
                logger.info(f"Milvus: {inserted} chunks inseridos")
            else:
                inserted = 0

            # Desconecta
            connections.disconnect("ingest")

            duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "milvus_sink",
                "duration_seconds": duration,
                "output": f"Inseridos {inserted} chunks no Milvus ({milvus_host})",
                "success": True,
            })
            logger.info(f"Fase 5.5: Milvus Sink concluido em {duration}s - {inserted} chunks inseridos")

        except Exception as e:
            logger.error(f"Erro no Milvus Sink: {e}", exc_info=True)
            # Nao falha o pipeline se Milvus falhar - apenas loga warning
            result.phases.append({
                "name": "milvus_sink",
                "duration_seconds": round(time.perf_counter() - phase_start, 2),
                "output": f"FALHA: {str(e)[:200]}",
                "success": False,
            })
            logger.warning(f"Milvus Sink falhou mas pipeline continua: {e}")

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
                    parent_chunk_id=chunk.parent_node_id if hasattr(chunk, "parent_node_id") else None,
                    document_type="ACORDAO",
                )

                # Prepara aliases para acórdão
                acordao_aliases = getattr(chunk, "aliases", "[]")
                acordao_sparse_source = getattr(chunk, "sparse_source", "") or chunk.enriched_text or chunk.text or ""

                pc = ProcessedChunk(
                    node_id=chunk.node_id,
                    chunk_id=chunk.chunk_id,
                    parent_node_id=chunk.parent_node_id or "",
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
                    # OriginClassifier: Classificação de origem
                    origin_type=getattr(chunk, 'origin_type', 'self'),
                    origin_confidence=getattr(chunk, 'origin_confidence', 'high'),
                    origin_reference=getattr(chunk, 'origin_reference', '') or '',
                    origin_reference_name=getattr(chunk, 'origin_reference_name', '') or '',
                    is_external_material=getattr(chunk, 'is_external_material', False) or False,
                    origin_reason=getattr(chunk, 'origin_reason', '') or '',
                )
            else:
                # MaterializedChunk (leis/decretos)
                # Extrai citações normativas do texto do chunk
                chunk_citations = extract_citations_from_chunk(
                    text=chunk.text or "",
                    document_id=request.document_id,
                    chunk_node_id=chunk.node_id,  # Remove self-loops
                    parent_chunk_id=chunk.parent_node_id,  # Remove parent-loops
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
                    parent_node_id=chunk.parent_node_id or "",
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
                    # OriginClassifier: Classificação de origem
                    origin_type=getattr(chunk, 'origin_type', 'self'),
                    origin_confidence=getattr(chunk, 'origin_confidence', 'high'),
                    origin_reference=getattr(chunk, 'origin_reference', '') or '',
                    origin_reference_name=getattr(chunk, 'origin_reference_name', '') or '',
                    is_external_material=getattr(chunk, 'is_external_material', False) or False,
                    origin_reason=getattr(chunk, 'origin_reason', '') or '',
                    # Campos Acórdão (vazios para LEI/DECRETO)
                    colegiado="",
                    processo="",
                    relator="",
                    data_sessao="",
                    unidade_tecnica="",
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

        # INVARIANTES: Valida contrato antes de retornar para VPS
        # Aborta pipeline se qualquer invariante for violada
        validate_chunk_invariants(chunks, request.document_id)

        return chunks


# Singleton
_pipeline: Optional[IngestionPipeline] = None


def get_pipeline() -> IngestionPipeline:
    """Retorna instancia singleton do pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline
