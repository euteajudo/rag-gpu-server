"""
Pipeline de Ingestao de PDFs — VLM (PyMuPDF + Qwen3-VL).

Pipeline completo:
1. PyMuPDF (PDF -> texto + blocos com offsets)
2. Qwen3-VL (classificação de dispositivos legais)
3. Offset resolution (bbox matching + page find + parent resolve)
4. Embeddings (BGE-M3)
5. Artifacts upload (VPS)

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
from ..chunking.citation_extractor import extract_citations_from_chunk
from ..chunking.canonical_offsets import normalize_canonical_text, compute_canonical_hash


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

            # Invariante cross-page
            is_cross_page = getattr(chunk, 'is_cross_page', False)
            bbox_spans = getattr(chunk, 'bbox_spans', [])
            if is_cross_page and not bbox_spans:
                violations.append(
                    f"[{span_id}] CRITICAL: is_cross_page=True mas bbox_spans vazio"
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
            error_msg += "\n  💡 CAUSA PROVÁVEL: VLM não conseguiu resolver offsets para estes dispositivos\n"
            error_msg += "  💡 SOLUÇÃO: Verificar bbox matching e canonical_text\n"

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
    # Métricas de qualidade
    extraction_method: ExtractionMethod = ExtractionMethod.NATIVE_TEXT
    quality_score: float = 0.0
    quality_issues: List[str] = field(default_factory=list)
    ocr_fallback_used: bool = False
    # PR13: Ingest run ID para rastreabilidade
    ingest_run_id: str = ""
    # PR13: Canonical hash para validação de offsets
    canonical_hash: str = ""
    # Pipeline version para reprodutibilidade
    pipeline_version: str = ""
    # Manifesto de ingestão para reconciliação pela VPS
    manifest: dict = field(default_factory=dict)


class IngestionPipeline:
    """Pipeline de ingestao de documentos legais via VLM (PyMuPDF + Qwen3-VL)."""

    def __init__(self):
        self._embedder = None
        self._artifacts_uploader = None
        self._vlm_service = None
        self._last_resolution_map: Dict[str, dict] = {}

    @property
    def embedder(self):
        if self._embedder is None:
            from ..embedder import get_embedder
            self._embedder = get_embedder()
            logger.info("BGE-M3 Embedder inicializado")
        return self._embedder

    @property
    def artifacts_uploader(self):
        """Uploader para enviar artifacts para a VPS."""
        if self._artifacts_uploader is None:
            from ..sinks.artifacts_uploader import get_artifacts_uploader
            self._artifacts_uploader = get_artifacts_uploader()
        return self._artifacts_uploader

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

        from ..config import config as app_config

        result = PipelineResult(
            status=IngestStatus.PROCESSING,
            document_id=request.document_id,
            ingest_run_id=ingest_run_id,
            pipeline_version=app_config.pipeline_version,
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
                extraction_mode = getattr(request, 'extraction_mode', 'pymupdf_regex')

                if extraction_mode == 'vlm':
                    # === PIPELINE VLM: PyMuPDF + Qwen3-VL ===
                    logger.info(f"Pipeline VLM ativo para {request.document_id}")
                    report_progress("vlm_extraction", 0.10)

                    self._phase_vlm_extraction(
                        pdf_content, request, result, report_progress
                    )
                else:
                    # === PIPELINE PyMuPDF + Regex ===
                    logger.info(f"Pipeline PyMuPDF+Regex ativo para {request.document_id}")
                    report_progress("pymupdf_regex_extraction", 0.10)

                    self._phase_pymupdf_regex_extraction(
                        pdf_content, request, result, report_progress
                    )

                if result.status == IngestStatus.FAILED:
                    return result

                result.total_time_seconds = round(time.perf_counter() - start_time, 2)
                return result

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

        Args:
            pdf_content: Bytes do PDF original
            request: Metadados do documento
            result: PipelineResult em construção
            report_progress: Callback de progresso
        """
        phase_start = time.perf_counter()

        try:
            from ..config import config as app_config
            # Roda o pipeline VLM assíncrono em um event loop limpo.
            # O process() é chamado de um thread background, então
            # criamos sempre um loop novo para evitar "Event loop is closed"
            # quando o httpx.AsyncClient tenta reusar um loop fechado.
            self.vlm_service.vlm_client.reset_client()
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

            # === Drift detection ===
            from ..utils.drift_detector import DriftDetector
            drift_detector = DriftDetector()
            drift = drift_detector.check(
                document_id=request.document_id,
                pdf_hash=result.document_hash,
                pipeline_version=app_config.pipeline_version,
                current_canonical_hash=extraction.canonical_hash,
            )
            if drift.is_drifted:
                logger.warning(f"DRIFT DETECTADO: {drift.message}")
                result.quality_issues.append(
                    f"DRIFT: canonical_hash mudou para mesmo pdf_hash+pipeline_version "
                    f"(prev={drift.previous_canonical_hash[:16]}... "
                    f"curr={drift.current_canonical_hash[:16]}...)"
                )

            # === Converte DocumentExtraction -> list[ProcessedChunk] ===
            report_progress("vlm_materialization", 0.80)

            chunks = self._vlm_to_processed_chunks(extraction, request, result)

            # === Canonical text swap (F1) ===
            # Para chunks com offsets resolvidos, substituir o texto VLM (que pode
            # conter conteúdo de dispositivos adjacentes) pelo slice exato do
            # canonical_text do PyMuPDF. O texto VLM original já está preservado
            # em extraction.debug_artifacts.
            canonical_text = extraction.canonical_text
            swap_count = 0
            for chunk in chunks:
                if chunk.canonical_start >= 0 and chunk.canonical_end > chunk.canonical_start:
                    canon_slice = canonical_text[chunk.canonical_start:chunk.canonical_end]
                    if canon_slice.strip():
                        chunk.text = canon_slice
                        chunk.retrieval_text = canon_slice
                        swap_count += 1
            if swap_count > 0:
                logger.info(
                    f"[{request.document_id}] Canonical text swap: "
                    f"{swap_count}/{len(chunks)} chunks"
                )

            # Origin Classification — detecta material externo (ex: artigos do CP inseridos)
            from ..classification.origin_classifier import classify_document
            chunks = classify_document(chunks, extraction.canonical_text, request.document_id)
            external_count = sum(1 for c in chunks if c.origin_type == "external")
            if external_count > 0:
                logger.info(f"[{request.document_id}] OriginClassifier: {external_count}/{len(chunks)} external")

            # Enriquecer retrieval_text para chunks external (proveniência)
            for chunk in chunks:
                if getattr(chunk, "is_external_material", False):
                    ref_name = getattr(chunk, "origin_reference_name", "") or ""
                    ref_id = getattr(chunk, "origin_reference", "") or ""
                    if ref_name:
                        prefix = f"{ref_name} ({ref_id})" if ref_id else ref_name
                        chunk.retrieval_text = f"{prefix}\n{chunk.retrieval_text}"

            report_progress("vlm_materialization", 0.88)

            # === Embeddings (se não pular) ===
            if not request.skip_embeddings:
                report_progress("embedding", 0.88)
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
                report_progress("embedding", 0.94)

                result.phases.append({
                    "name": "embedding",
                    "duration_seconds": round(time.perf_counter() - phase_start - vlm_duration, 2),
                    "output": f"Embeddings para {len(chunks)} chunks VLM",
                    "success": True,
                })

            # === Upload de artefatos para VPS ===
            report_progress("artifacts_upload", 0.94)
            self._phase_artifacts_upload(
                pdf_content, extraction.canonical_text, extraction.canonical_hash,
                chunks, request, result, debug_artifacts=extraction.debug_artifacts,
            )
            report_progress("artifacts_upload", 0.97)

            # === Valida invariantes do contrato ===
            validate_chunk_invariants(chunks, request.document_id)

            # Manifesto de ingestão
            result.manifest = self._build_manifest(
                chunks, extraction.canonical_text, extraction.canonical_hash, request.document_id,
            )

            result.chunks = chunks
            result.status = IngestStatus.COMPLETED

            # Register successful run for drift detection
            drift_detector.register_run(
                document_id=request.document_id,
                pdf_hash=result.document_hash,
                pipeline_version=app_config.pipeline_version,
                canonical_hash=extraction.canonical_hash,
                ingest_run_id=result.ingest_run_id,
            )

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

        Conversão de coordenadas:
        - device.bbox: normalizado 0-1 (image space) → armazenado em bbox_img
        - image_bbox_to_pdf_bbox(): converte para PDF points → armazenado em bbox
        """
        from ..chunking.citation_extractor import extract_citations_from_chunk
        from ..extraction.coord_utils import image_bbox_to_pdf_bbox

        chunks = []

        # Indexa pages_data por page_number para acesso às dimensões
        pages_data_map: Dict[int, Any] = {}
        for pd in (extraction.pages_data or []):
            pages_data_map[pd.page_number] = pd

        for page in extraction.pages:
            # Dimensões da página para conversão de bbox
            page_data = pages_data_map.get(page.page_number)
            page_width = page_data.width if page_data else 0.0
            page_height = page_data.height if page_data else 0.0
            pix_w = page_data.img_width if page_data else 0
            pix_h = page_data.img_height if page_data else 0

            for device in page.devices:
                # Filhos sem parent_identifier são lixo do VLM (ementa/preâmbulo
                # classificados como parágrafo). Dropar antes de criar chunk.
                if device.device_type in ("paragrafo", "inciso", "alinea") and not device.parent_identifier:
                    logger.debug(
                        f"Dropping orphan {device.device_type} sem parent: "
                        f"page={page.page_number}, text={device.text[:60]!r}"
                    )
                    continue

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

                # Conversão de coordenadas: image space → PDF space
                # O VLM pode retornar bbox em pixel absoluto OU 0-1 normalizado.
                # Se qualquer coordenada > 1.0, são pixels → normalizar por img dims.
                bbox_img = list(device.bbox) if device.bbox else []
                if bbox_img and len(bbox_img) == 4 and pix_w > 0 and pix_h > 0:
                    if any(v > 1.0 for v in bbox_img):
                        bbox_img = [
                            bbox_img[0] / pix_w,
                            bbox_img[1] / pix_h,
                            bbox_img[2] / pix_w,
                            bbox_img[3] / pix_h,
                        ]
                bbox_pdf = image_bbox_to_pdf_bbox(bbox_img, page_width, page_height) if bbox_img else []

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
                    # PR13: canonical offsets (sentinela inicial, resolvido por _resolve_vlm_offsets)
                    canonical_start=-1,
                    canonical_end=-1,
                    canonical_hash=extraction.canonical_hash,
                    # VLM: coordenadas
                    page_number=page.page_number,
                    bbox=bbox_pdf,           # PDF points (72 DPI) — para highlight
                    bbox_img=bbox_img,       # normalizado 0-1 — debug/reprodutibilidade
                    img_width=pix_w,
                    img_height=pix_h,
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

        # Detecção cross-page: agrupa chunks por span_id
        # Se mesmo span_id aparece em >1 página consecutiva, decide entre:
        # - Duplicata verdadeira (Jaccard > 0.7): mantém primeiro + bbox_spans
        # - Continuação (Jaccard ≤ 0.7): concatena textos + bbox_spans
        from collections import defaultdict
        span_pages: Dict[str, List[ProcessedChunk]] = defaultdict(list)
        for chunk in chunks:
            span_pages[chunk.span_id].append(chunk)

        cross_page_count = 0
        continuation_count = 0
        chunks_to_remove: set[int] = set()  # indices to remove

        for span_id, span_chunks in span_pages.items():
            if len(span_chunks) <= 1:
                continue

            # Sort by page_number
            span_chunks.sort(key=lambda c: c.page_number)

            # Check if pages are consecutive
            pages = [c.page_number for c in span_chunks]
            is_consecutive = all(
                pages[i+1] - pages[i] <= 1
                for i in range(len(pages) - 1)
            )

            if is_consecutive and len(pages) > 1:
                first = span_chunks[0]
                bbox_spans = [
                    {
                        "page_number": c.page_number,
                        "bbox_pdf": c.bbox,
                        "bbox_img": c.bbox_img,
                    }
                    for c in span_chunks
                ]

                # Word-Jaccard similarity between first and subsequent chunks
                first_words = set(first.text.lower().split())
                is_duplicate = True
                for other in span_chunks[1:]:
                    other_words = set(other.text.lower().split())
                    union = first_words | other_words
                    intersection = first_words & other_words
                    jaccard = len(intersection) / max(len(union), 1)
                    if jaccard <= 0.7:
                        is_duplicate = False
                        break

                first.is_cross_page = True
                first.bbox_spans = bbox_spans

                if is_duplicate:
                    # True duplicate: keep first chunk only
                    cross_page_count += 1
                    logger.debug(
                        f"Cross-page dedup {span_id}: true duplicate (Jaccard>0.7), "
                        f"keeping page {first.page_number}"
                    )
                else:
                    # Continuation: concatenate texts from subsequent pages
                    continuation_count += 1
                    for other in span_chunks[1:]:
                        first.text = first.text.rstrip() + " " + other.text.lstrip()
                    logger.debug(
                        f"Cross-page continuation {span_id}: texts differ (Jaccard≤0.7), "
                        f"concatenated from pages {pages}"
                    )

                # Mark subsequent chunks for removal in both cases
                for c in span_chunks[1:]:
                    idx = chunks.index(c)
                    chunks_to_remove.add(idx)

        if chunks_to_remove:
            chunks = [c for i, c in enumerate(chunks) if i not in chunks_to_remove]
            logger.info(
                f"Cross-page: {cross_page_count} duplicatas, {continuation_count} continuações, "
                f"{len(chunks_to_remove)} chunks secundários removidos"
            )

        if cross_page_count + continuation_count > 0:
            logger.warning(
                f"[{request.document_id}] {cross_page_count + continuation_count} chunks cross-page "
                f"detectados ({cross_page_count} dedup, {continuation_count} concat)"
            )

        # Resolve offsets reais (substitui sentinelas -1/-1)
        chunks = self._resolve_vlm_offsets(
            chunks, extraction.canonical_text,
            extraction.canonical_hash, request.document_id,
            extraction.pages_data,
        )

        return chunks

    def _resolve_vlm_offsets(
        self,
        chunks: List[ProcessedChunk],
        canonical_text: str,
        canonical_hash: str,
        document_id: str,
        pages_data: Optional[List] = None,
    ) -> List[ProcessedChunk]:
        """
        Computa offsets reais para cada chunk VLM dentro do canonical_text.

        Estratégia principal: bbox matching com blocos PyMuPDF que têm offsets nativos.
        Para cada device VLM, encontra blocos cuja bbox PDF tem overlap significativo.
        Os blocos matched já possuem char_start/char_end calculados durante a concatenação.

        Fallback (se blocos não disponíveis ou matching falha):
        - find() dentro do range da página (não global)
        - resolve_child_offsets() dentro do range do pai

        Args:
            chunks: Lista de ProcessedChunk com offsets sentinela (-1, -1)
            canonical_text: Texto canônico PyMuPDF normalizado
            canonical_hash: SHA256 do canonical_text
            document_id: ID do documento para logs
            pages_data: Lista de PageData com blocos e offsets nativos

        Returns:
            Lista de ProcessedChunk com offsets reais
        """
        import re
        from ..chunking.canonical_offsets import resolve_child_offsets, OffsetResolutionError
        from ..extraction.coord_utils import image_bbox_to_pdf_bbox, compute_bbox_iou
        from ..utils.matching_normalization import normalize_for_matching, normalize_with_offset_map

        if not canonical_text:
            logger.warning(f"[{document_id}] canonical_text vazio — offsets permanecem sentinela")
            return chunks

        # Indexa pages_data por page_number
        page_map: Dict[int, Any] = {}
        if pages_data:
            for pd in pages_data:
                page_map[pd.page_number] = pd

        # Offsets resolvidos: span_id -> (start, end)
        resolved: Dict[str, Tuple[int, int]] = {}
        # Fase que resolveu cada span (para debug/resolution_map)
        resolution_phases: Dict[str, str] = {}

        # =====================================================================
        # FASE A: Resolve via bbox matching com blocos PyMuPDF
        # =====================================================================
        bbox_resolved = 0
        for chunk in chunks:
            if chunk.page_number < 0 or not chunk.bbox:
                continue

            page_data = page_map.get(chunk.page_number)
            if not page_data or not page_data.blocks:
                continue

            # bbox do chunk já está em PDF points (convertido em _vlm_to_processed_chunks)
            chunk_bbox_pdf = chunk.bbox

            # Encontra blocos com overlap significativo
            matched_blocks = []
            for block in page_data.blocks:
                iou = compute_bbox_iou(chunk_bbox_pdf, block.bbox_pdf)
                if iou > 0.1:
                    matched_blocks.append((iou, block))

            if not matched_blocks:
                continue

            # Ordena por char_start para compor range contíguo
            matched_blocks.sort(key=lambda x: x[1].char_start)

            # Range = do início do primeiro bloco matched ao fim do último
            first_block = matched_blocks[0][1]
            last_block = matched_blocks[-1][1]
            start = first_block.char_start
            end = last_block.char_end

            if start >= 0 and end > start and end <= len(canonical_text):
                resolved[chunk.span_id] = (start, end)
                resolution_phases[chunk.span_id] = "A_bbox"
                bbox_resolved += 1
                logger.debug(
                    f"Offset {chunk.span_id}: bbox match [{start}:{end}] "
                    f"({len(matched_blocks)} blocos, IoU melhor={matched_blocks[0][0]:.2f})"
                )

        if bbox_resolved > 0:
            logger.info(f"[{document_id}] Fase A (bbox): {bbox_resolved} chunks resolvidos via blocos")

        # =====================================================================
        # FASE B: Resolve remanescentes via find() dentro do range da página
        # =====================================================================
        unresolved = [c for c in chunks if c.span_id not in resolved]
        find_resolved = 0
        normalized_resolved = 0

        for chunk in unresolved:
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue

            # Determina range de busca: página ou artigo pai
            search_start, search_end = 0, len(canonical_text)

            # Tenta restringir ao range da página
            page_data = page_map.get(chunk.page_number) if chunk.page_number > 0 else None
            if page_data:
                search_start = page_data.char_start
                search_end = page_data.char_end

            # Busca dentro do range da página
            region = canonical_text[search_start:search_end]
            pos = region.find(chunk_text)
            if pos >= 0:
                abs_start = search_start + pos
                abs_end = abs_start + len(chunk_text)
                resolved[chunk.span_id] = (abs_start, abs_end)
                resolution_phases[chunk.span_id] = "B_find"
                find_resolved += 1
                logger.debug(f"Offset {chunk.span_id}: page find [{abs_start}:{abs_end}]")
                continue

            # Fallback B2: find() com whitespace normalizado
            normalized = " ".join(chunk_text.split())
            pos = region.find(normalized)
            if pos >= 0:
                abs_start = search_start + pos
                abs_end = abs_start + len(normalized)
                resolved[chunk.span_id] = (abs_start, abs_end)
                resolution_phases[chunk.span_id] = "B_find_ws"
                find_resolved += 1
                logger.debug(f"Offset {chunk.span_id}: page find normalized [{abs_start}:{abs_end}]")
                continue

            # Fallback B3: normalização agressiva (NFKC, OCR table, etc.)
            norm_region, norm2orig = normalize_with_offset_map(region)
            norm_vlm = normalize_for_matching(chunk_text)
            if norm_vlm and norm_region:
                pos = norm_region.find(norm_vlm)
                if pos >= 0 and pos + len(norm_vlm) - 1 < len(norm2orig):
                    abs_start = search_start + norm2orig[pos]
                    abs_end = search_start + norm2orig[pos + len(norm_vlm) - 1] + 1
                    resolved[chunk.span_id] = (abs_start, abs_end)
                    resolution_phases[chunk.span_id] = "B_normalized"
                    normalized_resolved += 1
                    logger.debug(f"Offset {chunk.span_id}: B3 normalized [{abs_start}:{abs_end}]")
                    continue

        if find_resolved > 0:
            logger.info(f"[{document_id}] Fase B (find): {find_resolved} chunks resolvidos via busca de texto")
        if normalized_resolved > 0:
            logger.info(f"[{document_id}] Fase B3 (normalized): {normalized_resolved} chunks resolvidos via normalização agressiva")

        # =====================================================================
        # FASE C: Resolve filhos remanescentes dentro do range do pai
        # =====================================================================
        still_unresolved = [c for c in chunks
                           if c.span_id not in resolved
                           and c.device_type in ("paragraph", "inciso", "alinea")]
        child_resolved = 0

        # Monta lista ordenada de inícios de artigos para expandir parent_end
        # O range de um artigo vai do seu start até o start do próximo artigo
        art_starts = sorted(
            (start, span_id)
            for span_id, (start, end) in resolved.items()
            if span_id.startswith("ART-")
        )

        # Regex fallback: localiza "Art. Nº" no canonical_text
        _RE_ART_HEADER = re.compile(r"Art\.\s+\d+[º°]?\s")

        # C4: Prefix-anchored + structural delimiter + similarity validation
        # Quando C3 falha (VLM text difere do canonical no meio do texto),
        # ancoramos start pelo prefixo normalizado e end pelo próximo
        # delimitador estrutural. Validamos com similaridade.
        _C4_PREFIX_LEN = 30
        _C4_SIMILARITY_THRESHOLD = 0.80
        _RE_STRUCTURAL_DELIM = re.compile(
            r'(?:^|\n)\s*(?:'
            r'(?:[IVXLCDM]+|[0-9]+)\s*[-\u2013\u2014]\s+'   # inciso
            r'|[a-z]\)\s+'                                     # alínea
            r'|\u00a7\s*\d+'                                   # § parágrafo
            r'|Par[aá]grafo\s+[uú]nico'                        # parágrafo único
            r'|Art\.\s+\d+'                                    # artigo
            r')',
            re.MULTILINE,
        )

        def _expanded_parent_end(parent_start: int, parent_end: int) -> int:
            """Expande parent_end até o início do próximo artigo.

            O range resolvido de um artigo cobre apenas seu cabeçalho
            (ex: 'Art. 2º ... considera-se:'). Os filhos (§, incisos, alíneas)
            estão posicionados DEPOIS do cabeçalho no canonical_text.
            Expandimos até o próximo artigo para incluí-los na busca.

            Estratégia em 2 camadas:
            1. Próximo artigo resolvido em `art_starts`
            2. Fallback regex: busca 'Art. Nº' no canonical_text após parent_end
            """
            # Camada 1: próximo artigo já resolvido
            for art_start, _ in art_starts:
                if art_start > parent_start:
                    return art_start
            # Camada 2: regex no canonical_text (cobre artigos não resolvidos)
            m = _RE_ART_HEADER.search(canonical_text, pos=parent_end)
            if m:
                return m.start()
            return len(canonical_text)

        for child in still_unresolved:
            child_text = child.text.strip()
            if not child_text:
                continue

            # Determina parent span_id
            parent_span_id = ""
            if child.parent_node_id:
                parts = child.parent_node_id.split("#", 1)
                if len(parts) == 2:
                    parent_span_id = parts[1]

            # Busca range do pai
            parent_start, parent_end = -1, -1
            if parent_span_id and parent_span_id in resolved:
                parent_start, parent_end = resolved[parent_span_id]
            elif parent_span_id:
                # Tenta artigo ancestral
                art_match = re.search(r"(\d{3})", parent_span_id)
                if art_match:
                    art_span = f"ART-{art_match.group(1)}"
                    if art_span in resolved:
                        parent_start, parent_end = resolved[art_span]

            if parent_start < 0 or parent_end <= parent_start:
                continue

            # Expande parent_end para cobrir filhos (§, incisos, alíneas)
            parent_end = _expanded_parent_end(parent_start, parent_end)

            # resolve_child_offsets() dentro do range do pai
            try:
                abs_start, abs_end = resolve_child_offsets(
                    canonical_text=canonical_text,
                    parent_start=parent_start,
                    parent_end=parent_end,
                    chunk_text=child_text,
                    document_id=document_id,
                    span_id=child.span_id,
                    device_type=child.device_type,
                )
                resolved[child.span_id] = (abs_start, abs_end)
                resolution_phases[child.span_id] = "C_parent"
                child_resolved += 1
                continue
            except OffsetResolutionError:
                pass

            # Fallback C2: whitespace normalizado
            normalized = " ".join(child_text.split())
            try:
                abs_start, abs_end = resolve_child_offsets(
                    canonical_text=canonical_text,
                    parent_start=parent_start,
                    parent_end=parent_end,
                    chunk_text=normalized,
                    document_id=document_id,
                    span_id=child.span_id,
                    device_type=child.device_type,
                )
                resolved[child.span_id] = (abs_start, abs_end)
                resolution_phases[child.span_id] = "C_parent_ws"
                child_resolved += 1
                continue
            except OffsetResolutionError:
                pass

            # Fallback C3: normalização agressiva dentro do range do pai
            parent_region = canonical_text[parent_start:parent_end]
            norm_parent, norm2orig_parent = normalize_with_offset_map(parent_region)
            norm_child = normalize_for_matching(child_text)
            if norm_child and norm_parent:
                pos = norm_parent.find(norm_child)
                if pos >= 0 and pos + len(norm_child) - 1 < len(norm2orig_parent):
                    abs_start = parent_start + norm2orig_parent[pos]
                    abs_end = parent_start + norm2orig_parent[pos + len(norm_child) - 1] + 1
                    resolved[child.span_id] = (abs_start, abs_end)
                    resolution_phases[child.span_id] = "C_normalized"
                    normalized_resolved += 1
                    logger.debug(f"Offset {child.span_id}: C3 normalized [{abs_start}:{abs_end}]")
                    continue

            # =============================================================
            # Fallback C4: prefix-anchored start + structural delimiter end
            # + similarity validation (SequenceMatcher)
            # =============================================================
            from difflib import SequenceMatcher
            c4_accepted = False
            if norm_child and norm_parent and len(norm_child) >= _C4_PREFIX_LEN:
                prefix = norm_child[:_C4_PREFIX_LEN]
                prefix_pos = norm_parent.find(prefix)
                if prefix_pos >= 0 and prefix_pos < len(norm2orig_parent):
                    # Ancora start no canonical_text
                    orig_rel_start = norm2orig_parent[prefix_pos]
                    abs_start = parent_start + orig_rel_start

                    # Busca próximo delimitador estrutural após start+20
                    search_from = abs_start + 20
                    delim_match = _RE_STRUCTURAL_DELIM.search(
                        canonical_text, pos=search_from, endpos=parent_end,
                    )
                    abs_end = delim_match.start() if delim_match else parent_end

                    # Trim trailing whitespace do snippet
                    candidate = canonical_text[abs_start:abs_end].rstrip()
                    abs_end = abs_start + len(candidate)

                    # Similaridade entre candidato normalizado e VLM text
                    norm_candidate = normalize_for_matching(candidate)
                    similarity = SequenceMatcher(
                        None, norm_candidate, norm_child,
                    ).ratio()

                    logger.info(
                        f"[{document_id}] {child.span_id} C4: "
                        f"prefix@{prefix_pos}, [{abs_start}:{abs_end}], "
                        f"sim={similarity:.3f}"
                    )

                    if similarity >= _C4_SIMILARITY_THRESHOLD:
                        resolved[child.span_id] = (abs_start, abs_end)
                        resolution_phases[child.span_id] = "C4_prefix_delim"
                        child_resolved += 1
                        c4_accepted = True
                    else:
                        # C4 rejeitado — log vlm_text + candidate para análise
                        logger.warning(
                            f"[{document_id}] {child.span_id} C4 REJECTED "
                            f"sim={similarity:.3f}<{_C4_SIMILARITY_THRESHOLD}"
                        )
                        logger.warning(
                            f"  vlm[:100]={norm_child[:100]!r}"
                        )
                        logger.warning(
                            f"  can[:100]={norm_candidate[:100]!r}"
                        )

            if c4_accepted:
                continue

            # Não resolvido em nenhuma fase — mantém sentinela
            logger.warning(
                f"[{document_id}] {child.span_id} ({child.device_type}) "
                f"não resolvido — offset sentinela"
            )

        if child_resolved > 0:
            logger.info(f"[{document_id}] Fase C (parent): {child_resolved} filhos resolvidos via range do pai")

        # =====================================================================
        # Aplica offsets resolvidos aos chunks
        # =====================================================================
        resolved_count = 0
        for chunk in chunks:
            if chunk.span_id in resolved:
                start, end = resolved[chunk.span_id]
                chunk.canonical_start = start
                chunk.canonical_end = end
                chunk.canonical_hash = canonical_hash
                resolved_count += 1
            else:
                # Sentinela: limpa hash para manter trio coerente [-1, -1, ""]
                chunk.canonical_hash = ""

        # =====================================================================
        # Sibling overlap trim (F2)
        # Chunks irmãos (mesmo parent_node_id) não devem ter offsets sobrepostos.
        # Se VLM retornou text contendo dispositivos adjacentes, os offsets
        # resolvidos podem se sobrepor. Trimmamos curr.canonical_end para
        # não ultrapassar next.canonical_start.
        # =====================================================================
        from collections import defaultdict
        sibling_groups: Dict[str, list] = defaultdict(list)
        for chunk in chunks:
            if chunk.canonical_start >= 0 and chunk.parent_node_id:
                sibling_groups[chunk.parent_node_id].append(chunk)

        overlap_trimmed = 0
        for parent_id, siblings in sibling_groups.items():
            siblings.sort(key=lambda c: c.canonical_start)
            for i in range(len(siblings) - 1):
                curr = siblings[i]
                nxt = siblings[i + 1]
                if curr.canonical_end > nxt.canonical_start:
                    new_end = nxt.canonical_start
                    if new_end <= curr.canonical_start:
                        # Trim degenerado — reseta para sentinela
                        curr.canonical_start = -1
                        curr.canonical_end = -1
                        curr.canonical_hash = ""
                        resolved_count -= 1
                        logger.warning(
                            f"[{document_id}] Overlap trim degenerado: "
                            f"{curr.span_id} resetado para sentinela"
                        )
                    else:
                        curr.canonical_end = new_end
                        overlap_trimmed += 1

        if overlap_trimmed > 0:
            logger.info(
                f"[{document_id}] Sibling overlap trim: {overlap_trimmed} chunks"
            )

        # --- Per-phase fallback rate instrumentation ---
        total = len(chunks) or 1
        sentinel_count = len(chunks) - resolved_count
        phase_counts: Dict[str, int] = {}
        for phase in resolution_phases.values():
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        phase_counts["sentinel"] = sentinel_count

        # Build rate string: "A_bbox=30(60.0%) B_find=8(16.0%) ..."
        rate_parts = []
        for phase_name in ["A_bbox", "B_find", "B_find_ws", "B_normalized",
                           "C_parent", "C_parent_ws", "C_normalized",
                           "C4_prefix_delim", "sentinel"]:
            count = phase_counts.get(phase_name, 0)
            if count > 0:
                pct = 100.0 * count / total
                rate_parts.append(f"{phase_name}={count}({pct:.1f}%)")

        logger.info(
            f"[{document_id}] VLM offsets resolvidos: {resolved_count}/{len(chunks)} "
            f"(bbox={bbox_resolved}, find={find_resolved}, normalized={normalized_resolved}, "
            f"parent={child_resolved}, sentinela={sentinel_count})"
        )
        logger.info(
            f"[{document_id}] Fallback rates: {' | '.join(rate_parts)}"
        )

        # Build resolution_map for debug (attached to chunks as metadata)
        resolution_map = {}
        for chunk in chunks:
            resolution_map[chunk.span_id] = {
                "canonical_start": chunk.canonical_start,
                "canonical_end": chunk.canonical_end,
                "phase": resolution_phases.get(chunk.span_id, "sentinel"),
                "confidence": chunk.confidence,
                "page_number": chunk.page_number,
            }
        # Attach summary rates to resolution_map
        resolution_map["_summary"] = {
            "total_chunks": len(chunks),
            "resolved": resolved_count,
            "phase_counts": phase_counts,
            "phase_rates": {k: round(v / total, 4) for k, v in phase_counts.items()},
        }

        # Store resolution_map for debug artifacts upload
        self._last_resolution_map = resolution_map

        return chunks

    @staticmethod
    def _extract_art_num_suffix(identifier: str):
        """Extrai número e sufixo de um identifier de artigo.
        'Art. 337-E' → ('337', '-E'), 'Art. 5º' → ('5', ''), 'Art. 1.048' → ('1048', '')
        """
        import re
        m = re.search(r"(\d+(?:\.\d+)*)(?:[º°o])?(-[A-Za-z]+)?", identifier or "")
        if not m:
            return "000", ""
        num = m.group(1).replace(".", "")
        suffix = m.group(2) or ""
        return num, suffix

    @staticmethod
    def _device_to_span_id(device) -> str:
        """Converte DeviceExtraction para span_id (ex: ART-005, PAR-005-1, ART-337-E)."""
        import re

        identifier = device.identifier.strip()
        dtype = device.device_type.lower()

        if dtype == "artigo":
            # "Art. 5º" -> "ART-005", "Art. 337-E" -> "ART-337-E"
            num, suffix = IngestionPipeline._extract_art_num_suffix(identifier)
            return f"ART-{num.zfill(3)}{suffix}"

        elif dtype == "paragrafo":
            # "§ 1º" -> precisa do pai para montar PAR-005-1 ou PAR-337-E-1
            match = re.search(r"(\d+)", identifier)
            num = match.group(1) if match else "0"
            # Tenta extrair número e sufixo do artigo pai
            parent_id = device.parent_identifier or ""
            parent_num, parent_suffix = IngestionPipeline._extract_art_num_suffix(parent_id)
            return f"PAR-{parent_num.zfill(3)}{parent_suffix}-{num}"

        elif dtype == "inciso":
            # "I" / "II" -> INC-005-1 ou INC-337-E-1
            parent_id = device.parent_identifier or ""
            parent_num, parent_suffix = IngestionPipeline._extract_art_num_suffix(parent_id)
            # Converte romano ou numérico
            inc_num = IngestionPipeline._roman_to_int(identifier.strip().rstrip(").-"))
            return f"INC-{parent_num.zfill(3)}{parent_suffix}-{inc_num}"

        elif dtype == "alinea":
            # "a)" -> ALI-005-3-a ou ALI-337-E-3-a
            parent_id = device.parent_identifier or ""
            parent_match = re.search(r"(\d+)", parent_id)
            if parent_match:
                parent_num = parent_match.group(1).zfill(3)
                # Check for suffix in grandparent context (via parent_id)
                _, parent_suffix = IngestionPipeline._extract_art_num_suffix(parent_id)
                parent_num = f"{parent_num}{parent_suffix}"
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
        num, suffix = IngestionPipeline._extract_art_num_suffix(identifier)

        # Detecta tipo do pai pelo identificador
        ident_lower = identifier.lower().strip()
        if ident_lower.startswith("art"):
            return f"ART-{num.zfill(3)}{suffix}"
        elif "§" in identifier or ident_lower.startswith("par"):
            par_match = re.search(r"(\d+)", identifier)
            par_num = par_match.group(1) if par_match else "0"
            return f"PAR-{num.zfill(3)}{suffix}-{par_num}"
        else:
            return f"ART-{num.zfill(3)}{suffix}"

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

    def _phase_artifacts_upload(
        self,
        pdf_content: bytes,
        canonical_text: str,
        canonical_hash: str,
        chunks: List[ProcessedChunk],
        request: IngestRequest,
        result: PipelineResult,
        debug_artifacts: Optional[list] = None,
    ) -> bool:
        """
        Upload de artefatos para a VPS.

        Envia PDF original, canonical.md e offsets.json.

        Se o upload falhar, continua o pipeline com warning.

        Returns:
            True sempre (nunca aborta o pipeline)
        """
        phase_start = time.perf_counter()

        try:
            uploader = self.artifacts_uploader
            if not uploader.is_configured():
                logger.info("Artifacts upload: uploader não configurado, pulando")
                result.phases.append({
                    "name": "artifacts_upload",
                    "duration_seconds": 0.0,
                    "output": "Skipped (uploader não configurado)",
                    "success": True,
                    "skipped": True,
                })
                return True

            logger.info("Artifacts upload iniciando...")

            from ..config import config as app_config
            from ..sinks.artifacts_uploader import (
                ArtifactMetadata,
                prepare_offsets_map,
                compute_sha256,
            )

            # Canonical text normalizado
            canonical_md = normalize_canonical_text(canonical_text)
            c_hash = compute_canonical_hash(canonical_md)

            # Constrói offsets_map a partir dos chunks com offsets resolvidos
            offsets_map: Dict[str, Tuple[int, int]] = {}
            for c in chunks:
                if c.canonical_start >= 0 and c.canonical_end > c.canonical_start:
                    offsets_map[c.span_id] = (c.canonical_start, c.canonical_end)

            offsets_json = prepare_offsets_map(offsets_map)

            metadata = ArtifactMetadata(
                document_id=request.document_id,
                tipo_documento=request.tipo_documento,
                numero=request.numero,
                ano=request.ano,
                sha256_source=result.document_hash,
                sha256_canonical_md=c_hash,
                canonical_hash=c_hash,
                ingest_run_id=result.ingest_run_id,
                pipeline_version=app_config.pipeline_version,
                document_version=str(int(datetime.utcnow().timestamp())),
            )

            # Debug artifacts (if enabled)
            vlm_debug_json = None
            resolution_map_json = None
            if app_config.debug_artifacts:
                import json as _json
                if debug_artifacts:
                    vlm_debug_json = _json.dumps(
                        debug_artifacts, ensure_ascii=False, indent=2,
                    ).encode("utf-8")
                if self._last_resolution_map:
                    resolution_map_json = _json.dumps(
                        self._last_resolution_map, ensure_ascii=False, indent=2,
                    ).encode("utf-8")

            upload_result = uploader.upload(
                pdf_content=pdf_content,
                canonical_md=canonical_md,
                offsets_json=offsets_json,
                metadata=metadata,
                vlm_debug_json=vlm_debug_json,
                resolution_map_json=resolution_map_json,
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
                logger.info(f"Artifacts upload concluído em {duration}s")
            else:
                result.phases.append({
                    "name": "artifacts_upload",
                    "duration_seconds": duration,
                    "output": f"WARNING: {upload_result.error}",
                    "retries": upload_result.retries,
                    "success": False,
                })
                logger.warning(
                    f"Artifacts upload FALHOU em {duration}s — "
                    f"continuando pipeline"
                )

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
            return True

    def _emit_regex_inspection_snapshot(
        self,
        classification_result: dict,
        canonical_text: str,
        canonical_hash: str,
        duration_ms: float,
        request: IngestRequest,
        pages_data: list,
    ) -> None:
        """
        Emite snapshot da classificação regex para o Redis (Inspector).
        Falha silenciosa — não aborta o pipeline.
        """
        import unicodedata
        from ..inspection.storage import InspectionStorage
        from ..inspection.models import (
            InspectionStage,
            InspectionMetadata,
            InspectionStatus,
            RegexClassificationArtifact,
            RegexDevice,
            RegexFilteredBlock,
            RegexUnclassifiedBlock,
            RegexClassificationStats,
            RegexIntegrityChecks,
            RegexOffsetCheck,
            PyMuPDFArtifact,
            PyMuPDFPageResult,
            PyMuPDFBlock,
            BBox,
        )

        storage = InspectionStorage()

        # --- Build regex artifact ---
        regex_devices = []
        for d in classification_result.get("devices", []):
            regex_devices.append(RegexDevice(
                span_id=d["span_id"],
                device_type=d["device_type"],
                identifier=d["identifier"] or "",
                parent_span_id=d["parent_span_id"] or "",
                children_span_ids=d.get("children_span_ids", []),
                hierarchy_depth=d["hierarchy_depth"],
                text=d["full_text"],
                text_preview=d["text_preview"],
                char_start=d["char_start"],
                char_end=d["char_end"],
                page_number=d["page_number"],
                bbox=d.get("bbox", []),
            ))

        filtered = [
            RegexFilteredBlock(
                block_index=f["block_index"],
                page_number=f["page_number"],
                filter_type=f["filter_type"],
                reason=f.get("reason", ""),
                text_preview=f.get("text_preview", ""),
            )
            for f in classification_result.get("filtered", [])
        ]

        unclassified = [
            RegexUnclassifiedBlock(
                block_index=u["block_index"],
                page_number=u["page_number"],
                reason=u.get("reason", ""),
                text_preview=u.get("text_preview", ""),
            )
            for u in classification_result.get("unclassified", [])
        ]

        raw_stats = classification_result.get("stats", {})
        stats = RegexClassificationStats(
            total_blocks=raw_stats.get("total_blocks", 0),
            devices=raw_stats.get("devices", 0),
            filtered=raw_stats.get("filtered", 0),
            unclassified=raw_stats.get("unclassified", 0),
            by_device_type=raw_stats.get("by_device_type", {}),
            by_filter_type=raw_stats.get("by_filter_type", {}),
            max_hierarchy_depth=raw_stats.get("max_hierarchy_depth", 0),
        )

        # --- Integrity checks ---
        from ..chunking.canonical_offsets import normalize_canonical_text as norm_ct

        # Check 1: offsets
        offset_details = []
        offsets_matches = 0
        for d in regex_devices:
            sliced = canonical_text[d.char_start:d.char_end] if d.char_start >= 0 else ""
            match = sliced == d.text
            if match:
                offsets_matches += 1
            offset_details.append(RegexOffsetCheck(
                span_id=d.span_id,
                page=d.page_number,
                char_start=d.char_start,
                char_end=d.char_end,
                match=match,
                expected_preview=d.text[:60],
                got_preview=sliced[:60],
            ))

        # Check 2: normalization idempotent
        norm_idempotent = canonical_text == norm_ct(canonical_text)

        # Check 3: no trailing spaces per line
        trailing_violations = sum(
            1 for line in canonical_text.split("\n")
            if line != line.rstrip()
        )

        # Check 4: unicode NFC
        is_nfc = unicodedata.is_normalized("NFC", canonical_text)

        # Check 5: trailing newline
        has_trailing_nl = canonical_text.endswith("\n") if canonical_text else False

        offsets_pass = offsets_matches == len(offset_details)
        all_pass = (
            offsets_pass
            and norm_idempotent
            and trailing_violations == 0
            and is_nfc
            and has_trailing_nl
        )

        checks = RegexIntegrityChecks(
            all_pass=all_pass,
            offsets_pass=offsets_pass,
            offsets_total=len(offset_details),
            offsets_matches=offsets_matches,
            offsets_details=offset_details,
            normalization_idempotent=norm_idempotent,
            no_trailing_spaces=trailing_violations == 0,
            trailing_space_violations=trailing_violations,
            unicode_nfc=is_nfc,
            trailing_newline=has_trailing_nl,
        )

        artifact = RegexClassificationArtifact(
            devices=regex_devices,
            filtered=filtered,
            unclassified=unclassified,
            stats=stats,
            checks=checks,
            canonical_text=canonical_text,
            canonical_hash=canonical_hash,
            canonical_length=len(canonical_text),
            duration_ms=duration_ms * 1000,
        )

        # --- Build PyMuPDF artifact (page images + blocks) ---
        pymupdf_pages = []
        for pg in pages_data:
            blocks = []
            for b in pg.blocks:
                blocks.append(PyMuPDFBlock(
                    block_index=b.block_index,
                    text=b.text[:200],
                    bbox=BBox(
                        x0=b.bbox_pdf[0], y0=b.bbox_pdf[1],
                        x1=b.bbox_pdf[2], y1=b.bbox_pdf[3],
                    ) if len(b.bbox_pdf) == 4 else BBox(x0=0, y0=0, x1=0, y1=0),
                    page=pg.page_number,
                ))
            pymupdf_pages.append(PyMuPDFPageResult(
                page_number=pg.page_number,
                width=pg.width,
                height=pg.height,
                blocks=blocks,
                image_base64=pg.image_base64,
            ))

        pymupdf_artifact = PyMuPDFArtifact(
            pages=pymupdf_pages,
            total_blocks=sum(len(p.blocks) for p in pages_data),
            total_pages=len(pages_data),
            total_chars=len(canonical_text),
        )

        # --- Save to Redis ---
        task_id = f"regex_{request.document_id}_{int(time.time())}"

        metadata = InspectionMetadata(
            inspection_id=task_id,
            document_id=request.document_id,
            tipo_documento=request.tipo_documento,
            numero=request.numero,
            ano=request.ano,
            total_pages=len(pages_data),
            started_at=datetime.utcnow().isoformat(),
            status=InspectionStatus.COMPLETED,
        )

        storage.save_metadata(task_id, metadata)
        storage.save_artifact(
            task_id, InspectionStage.REGEX_CLASSIFICATION,
            artifact.model_dump_json(),
        )
        storage.save_artifact(
            task_id, InspectionStage.PYMUPDF,
            pymupdf_artifact.model_dump_json(),
        )

        logger.info(
            f"Inspector snapshot emitted: {task_id} "
            f"({len(regex_devices)} devices, checks={'PASS' if all_pass else 'FAIL'})"
        )

    def _convert_pages_to_classifier_format(self, pages_data) -> list:
        """
        Converte PageData/BlockData (do PyMuPDFExtractor) para o formato dict
        que o regex classifier espera.
        """
        pages = []
        for page in pages_data:
            blocks = []
            for block in page.blocks:
                blocks.append({
                    "block_index": block.block_index,
                    "text": block.text,
                    "char_start": block.char_start,
                    "char_end": block.char_end,
                    "bbox": block.bbox_pdf,
                    "lines": block.lines,
                })
            pages.append({
                "page_number": page.page_number,
                "blocks": blocks,
            })
        return pages

    def _regex_to_processed_chunks(
        self,
        devices,
        canonical_text: str,
        canonical_hash: str,
        request: IngestRequest,
    ) -> List[ProcessedChunk]:
        """
        Converte List[ClassifiedDevice] -> List[ProcessedChunk].

        Offsets são nativos do PyMuPDF (nunca -1).
        """
        chunks = []

        # Build span_id → device map for subtree aggregation
        device_by_span = {d.span_id: d for d in devices}

        def _build_retrieval_text(article_device):
            """Concatena texto do artigo + toda a subárvore de filhos (recursivo)."""
            parts = [article_device.text or ""]

            def collect(span_id):
                d = device_by_span.get(span_id)
                if d:
                    parts.append(d.text or "")
                    for child_id in d.children_span_ids:
                        collect(child_id)

            for child_id in article_device.children_span_ids:
                collect(child_id)

            return "\n".join(parts)

        for device in devices:
            # node_id e chunk_id
            chunk_id = f"{request.document_id}#{device.span_id}"
            node_id = f"leis:{chunk_id}"

            # parent_node_id
            parent_node_id = ""
            if device.parent_span_id:
                parent_node_id = f"leis:{request.document_id}#{device.parent_span_id}"

            # chunk_level
            chunk_level = "article" if device.device_type == "article" else "device"

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
                span_id=device.span_id,
                device_type=device.device_type,
                chunk_level=chunk_level,
                text=device.text or "",
                parent_text="",
                retrieval_text=_build_retrieval_text(device) if chunk_level == "article" else (device.text or ""),
                document_id=request.document_id,
                tipo_documento=request.tipo_documento,
                numero=request.numero,
                ano=request.ano,
                article_number=str(device.article_number) if device.article_number else "",
                citations=citations,
                # PR13: offsets nativos (NUNCA -1)
                canonical_start=device.char_start,
                canonical_end=device.char_end,
                canonical_hash=canonical_hash,
                # Coordenadas
                page_number=device.page_number,
                bbox=device.bbox,
                bbox_img=[],
                img_width=0,
                img_height=0,
                confidence=1.0,  # regex determinístico
                # Cross-page
                is_cross_page=False,
                bbox_spans=[],
                # Campos Acórdão (vazios para LEI/DECRETO)
                colegiado="",
                processo="",
                relator="",
                data_sessao="",
                unidade_tecnica="",
            )
            chunks.append(pc)

        logger.info(f"Regex -> ProcessedChunk: {len(chunks)} chunks gerados")
        return chunks

    def _phase_pymupdf_regex_extraction(
        self,
        pdf_content: bytes,
        request: IngestRequest,
        result: PipelineResult,
        report_progress,
    ) -> None:
        """
        Pipeline PyMuPDF + Regex: extração estrutural sem GPU.

        1. PyMuPDFExtractor.extract_pages() → pages_data, canonical_text
        2. RegexClassifier → List[ClassifiedDevice]
        3. _regex_to_processed_chunks() → List[ProcessedChunk]
        4. OriginClassifier + Embeddings + Artifacts upload
        """
        phase_start = time.perf_counter()

        try:
            from ..config import config as app_config
            from ..extraction.pymupdf_extractor import PyMuPDFExtractor
            from ..extraction.regex_classifier import classify_to_devices

            # 1. Extração PyMuPDF (MESMO extrator do VLM path)
            extractor = PyMuPDFExtractor(dpi=app_config.vlm_page_dpi)
            pages_data, raw_canonical = extractor.extract_pages(pdf_content)

            report_progress("pymupdf_regex_extraction", 0.30)

            # 2. canonical_text = raw do extractor (já normalizado inline: NFC + rstrip + trailing \n)
            # Os offsets dos blocos são nativos a este texto — NÃO re-normalizar.
            canonical_text = raw_canonical
            normalized_check = normalize_canonical_text(raw_canonical)
            if canonical_text != normalized_check:
                logger.error(
                    f"[{request.document_id}] OFFSET DRIFT: extract_pages() retornou texto "
                    f"que difere de normalize_canonical_text() — offsets seriam inválidos! "
                    f"raw_len={len(canonical_text)} norm_len={len(normalized_check)}"
                )
                raise RuntimeError(
                    "extract_pages() output diverge de normalize_canonical_text(): "
                    "offsets nativos seriam inválidos. Verifique normalização inline do extractor."
                )
            canonical_hash = compute_canonical_hash(canonical_text)

            result.markdown_content = canonical_text
            result.canonical_hash = canonical_hash

            # 3. Drift detection
            from ..utils.drift_detector import DriftDetector
            drift_detector = DriftDetector()
            drift = drift_detector.check(
                document_id=request.document_id,
                pdf_hash=result.document_hash,
                pipeline_version=app_config.pipeline_version,
                current_canonical_hash=canonical_hash,
            )
            if drift.is_drifted:
                logger.warning(f"DRIFT DETECTADO: {drift.message}")
                result.quality_issues.append(
                    f"DRIFT: canonical_hash mudou para mesmo pdf_hash+pipeline_version "
                    f"(prev={drift.previous_canonical_hash[:16]}... "
                    f"curr={drift.current_canonical_hash[:16]}...)"
                )

            report_progress("pymupdf_regex_extraction", 0.40)

            # 4. Converte pages para formato do classifier
            pages_for_classifier = self._convert_pages_to_classifier_format(pages_data)

            # 5. Classificação regex
            from ..extraction.regex_classifier import classify_document as regex_classify_document
            classification_result = regex_classify_document(pages_for_classifier)
            devices = classify_to_devices(pages_for_classifier)
            logger.info(
                f"[{request.document_id}] RegexClassifier: {len(devices)} dispositivos"
            )

            report_progress("pymupdf_regex_extraction", 0.55)

            # Registra fase
            extract_duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "pymupdf_regex_extraction",
                "duration_seconds": extract_duration,
                "output": (
                    f"Extraídos {len(devices)} dispositivos de "
                    f"{len(pages_data)} páginas via PyMuPDF+Regex"
                ),
                "success": True,
                "method": "pymupdf+regex",
            })

            # === EMIT INSPECTION SNAPSHOT (Redis) ===
            try:
                self._emit_regex_inspection_snapshot(
                    classification_result, canonical_text, canonical_hash,
                    extract_duration, request, pages_data,
                )
            except Exception as e:
                logger.warning(f"Failed to emit inspector snapshot: {e}")

            # 6. Converte ClassifiedDevice -> ProcessedChunk
            chunks = self._regex_to_processed_chunks(
                devices, canonical_text, canonical_hash, request,
            )

            report_progress("pymupdf_regex_extraction", 0.65)

            # 7. Origin Classification
            from ..classification.origin_classifier import classify_document
            chunks = classify_document(chunks, canonical_text, request.document_id)
            external_count = sum(1 for c in chunks if c.origin_type == "external")
            if external_count > 0:
                logger.info(f"[{request.document_id}] OriginClassifier: {external_count}/{len(chunks)} external")

            # 7.5. Enriquecer retrieval_text para chunks external (proveniência)
            for chunk in chunks:
                if getattr(chunk, "is_external_material", False):
                    ref_name = getattr(chunk, "origin_reference_name", "") or ""
                    ref_id = getattr(chunk, "origin_reference", "") or ""
                    if ref_name:
                        prefix = f"{ref_name} ({ref_id})" if ref_id else ref_name
                        chunk.retrieval_text = f"{prefix}\n{chunk.retrieval_text}"

            report_progress("pymupdf_regex_extraction", 0.70)

            # 8. Embeddings (se não pular)
            if not request.skip_embeddings:
                report_progress("embedding", 0.70)
                for chunk in chunks:
                    text_for_embedding = chunk.retrieval_text or chunk.text
                    embed_result = self.embedder.encode([text_for_embedding])
                    chunk.dense_vector = embed_result.dense_embeddings[0]
                    chunk.sparse_vector = (
                        embed_result.sparse_embeddings[0]
                        if embed_result.sparse_embeddings
                        else {}
                    )
                report_progress("embedding", 0.88)

                result.phases.append({
                    "name": "embedding",
                    "duration_seconds": round(time.perf_counter() - phase_start - extract_duration, 2),
                    "output": f"Embeddings para {len(chunks)} chunks Regex",
                    "success": True,
                })

            # 9. Artifacts upload
            report_progress("artifacts_upload", 0.88)
            self._phase_artifacts_upload(
                pdf_content, canonical_text, canonical_hash,
                chunks, request, result,
            )
            report_progress("artifacts_upload", 0.94)

            # 10. Valida invariantes do contrato
            validate_chunk_invariants(chunks, request.document_id)

            # 11. Drift register + resultado
            drift_detector.register_run(
                document_id=request.document_id,
                pdf_hash=result.document_hash,
                pipeline_version=app_config.pipeline_version,
                canonical_hash=canonical_hash,
                ingest_run_id=result.ingest_run_id,
            )

            # 12. Manifesto de ingestão
            result.manifest = self._build_manifest(
                chunks, canonical_text, canonical_hash, request.document_id,
            )

            result.chunks = chunks
            result.status = IngestStatus.COMPLETED

            report_progress("completed", 1.0)

            logger.info(
                f"Pipeline PyMuPDF+Regex concluído: {len(chunks)} chunks, "
                f"{len(devices)} dispositivos, "
                f"manifest: {result.manifest.get('total_spans')} spans"
            )

        except Exception as e:
            logger.error(f"Erro no pipeline PyMuPDF+Regex: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(
                phase="pymupdf_regex",
                message=str(e),
            ))

    @staticmethod
    def _build_manifest(
        chunks: List[ProcessedChunk],
        canonical_text: str,
        canonical_hash: str,
        document_id: str,
    ) -> dict:
        """
        Gera manifesto de ingestão para reconciliação pela VPS.

        Inclui contagens, lista de span_ids, cobertura de offsets,
        e material externo detectado.
        """
        span_ids = []
        spans_by_type: Dict[str, int] = {}
        chunks_by_level: Dict[str, int] = {}
        article_numbers = []
        external_spans = []
        vehicle_articles = []
        target_documents_set: set = set()
        covered_intervals = []

        for c in chunks:
            sid = c.span_id or ""
            span_ids.append(sid)

            dt = c.device_type or ""
            spans_by_type[dt] = spans_by_type.get(dt, 0) + 1

            cl = c.chunk_level or ""
            chunks_by_level[cl] = chunks_by_level.get(cl, 0) + 1

            if dt == "article":
                # Extrai numero do artigo do span_id (ART-005 -> "5", ART-337-E -> "337-E")
                art_part = sid.replace("ART-", "", 1) if sid.startswith("ART-") else ""
                if art_part:
                    # Remove leading zeros: "005" -> "5", "337-E" -> "337-E"
                    parts = art_part.split("-", 1)
                    num_str = str(int(parts[0])) if parts[0].isdigit() else parts[0]
                    if len(parts) > 1:
                        num_str += f"-{parts[1]}"
                    article_numbers.append(num_str)

            if getattr(c, "origin_type", "self") == "external":
                external_spans.append(sid)
            if getattr(c, "origin_reason", "") and "veiculo" in getattr(c, "origin_reason", ""):
                vehicle_articles.append(sid)
            origin_ref = getattr(c, "origin_reference", "") or ""
            if origin_ref and origin_ref not in target_documents_set:
                target_documents_set.add(origin_ref)

            cs = getattr(c, "canonical_start", -1)
            ce = getattr(c, "canonical_end", -1)
            if cs >= 0 and ce > cs:
                covered_intervals.append((cs, ce))

        # Calcula cobertura de offsets (merge overlapping intervals)
        covered_chars = 0
        if covered_intervals:
            covered_intervals.sort()
            merged = [covered_intervals[0]]
            for start, end in covered_intervals[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            covered_chars = sum(end - start for start, end in merged)

        total_chars = len(canonical_text) if canonical_text else 0
        coverage_pct = round((covered_chars / total_chars * 100), 1) if total_chars > 0 else 0.0

        return {
            "document_id": document_id,
            "canonical_hash": canonical_hash,
            "canonical_length": total_chars,
            "total_spans": len(span_ids),
            "total_chunks": len(chunks),
            "spans_by_type": spans_by_type,
            "chunks_by_level": chunks_by_level,
            "span_ids": span_ids,
            "article_numbers": article_numbers,
            "offsets_coverage": {
                "total_chars": total_chars,
                "covered_chars": covered_chars,
                "coverage_pct": coverage_pct,
            },
            "external_material": {
                "count": len(external_spans),
                "spans": external_spans,
                "target_documents": sorted(target_documents_set),
                "vehicle_articles": vehicle_articles,
            },
        }

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

                # PR13: parent_node_id já vem pronto do ProcessedChunk
                parent_node_id = getattr(chunk, 'parent_node_id', '') or ''

                # Extrai citações para has_citations e citations_count
                citations = getattr(chunk, 'citations', []) or []
                has_citations = len(citations) > 0
                citations_count = len(citations)

                # Device type
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
                    "dense_vector": getattr(chunk, 'dense_vector', [0.0] * 1024),
                    "sparse_vector": getattr(chunk, 'sparse_vector', {}),
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


# Singleton
_pipeline: Optional[IngestionPipeline] = None


def get_pipeline() -> IngestionPipeline:
    """Retorna instancia singleton do pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline
