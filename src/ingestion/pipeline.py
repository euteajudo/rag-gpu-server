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
    1. node_id DEVE começar com "leis:" ou "acordaos:" e NÃO conter "@P"
    2. parent_node_id (quando não vazio) DEVE começar com prefixo válido e NÃO conter "@P"
    3. Filhos devem ter parent_node_id
    4. PR13 trio deve ser coerente (sentinela ou válido, nunca misturado)
    5. EVIDENCE CHUNKS DEVEM ter offsets válidos
       - Sentinela (-1,-1,"") é PROIBIDO para evidence
       - Evidence sem offset = snippet não determinístico = ABORT

    Raises:
        ContractViolationError: Se qualquer invariante for violada (aborta pipeline)
    """
    violations = []

    VALID_PREFIXES = ("leis:", "acordaos:")
    EVIDENCE_LEI = {"article", "paragraph", "inciso", "alinea"}
    EVIDENCE_ACORDAO = {"section", "paragraph", "item_dispositivo"}

    for i, chunk in enumerate(chunks):
        node_id = chunk.node_id or ""
        parent_node_id = chunk.parent_node_id or ""
        device_type = chunk.device_type or ""
        span_id = chunk.span_id or ""

        # Determina prefixo e evidence types
        if node_id.startswith("acordaos:"):
            prefix = "acordaos:"
            evidence_types = EVIDENCE_ACORDAO
        elif node_id.startswith("leis:"):
            prefix = "leis:"
            evidence_types = EVIDENCE_LEI
        else:
            violations.append(
                f"[{span_id}] node_id com prefixo desconhecido: '{node_id}'"
            )
            continue

        is_evidence = device_type in evidence_types

        # Invariante 2: node_id não pode conter @Pxx
        if "@P" in node_id:
            violations.append(
                f"[{span_id}] node_id contém @Pxx (proibido): '{node_id}'"
            )

        # Invariante 3: parent_node_id (quando não vazio) deve começar com prefixo válido
        if parent_node_id and not parent_node_id.startswith(prefix):
            violations.append(
                f"[{span_id}] parent_node_id sem prefixo '{prefix}': '{parent_node_id}'"
            )

        # Invariante 4: parent_node_id não pode conter @Pxx
        if "@P" in parent_node_id:
            violations.append(
                f"[{span_id}] parent_node_id contém @Pxx (proibido): '{parent_node_id}'"
            )

        # Invariante 5: filhos devem ter parent_node_id não vazio
        child_types_lei = ("paragraph", "inciso", "alinea")
        child_types_acordao = ("paragraph", "item_dispositivo")
        child_types = child_types_acordao if prefix == "acordaos:" else child_types_lei
        if device_type in child_types and not parent_node_id:
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
    # Snapshot de inspeção para a VPS salvar no PostgreSQL
    inspection_snapshot: dict = field(default_factory=dict)


class IngestionPipeline:
    """Pipeline de ingestao de documentos legais via VLM (PyMuPDF + Qwen3-VL)."""

    def __init__(self):
        self._embedder = None
        self._artifacts_uploader = None
        self._vlm_service = None

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

                if request.tipo_documento == "ACORDAO":
                    # === PIPELINE ACÓRDÃO ===
                    if extraction_mode == 'vlm':
                        logger.info(f"Pipeline Acórdão+VLM ativo para {request.document_id}")
                        report_progress("acordao_vlm_extraction", 0.10)
                        self._phase_acordao_vlm_extraction(
                            pdf_content, request, result, report_progress
                        )
                    else:
                        logger.info(f"Pipeline Acórdão+Regex ativo para {request.document_id}")
                        report_progress("acordao_extraction", 0.10)
                        self._phase_acordao_extraction(
                            pdf_content, request, result, report_progress
                        )
                elif extraction_mode == 'vlm':
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
        Pipeline VLM OCR: PyMuPDF (imagens) + Qwen3-VL (OCR) → mesmo regex.

        Entrada 2: a única diferença da Entrada 1 é DE ONDE vem o texto.
        Aqui o texto vem do OCR do Qwen3-VL em vez do PyMuPDF nativo.
        Todo o pipeline downstream (regex classifier, chunks, embeddings) é idêntico.
        """
        phase_start = time.perf_counter()

        try:
            from ..config import config as app_config
            from ..extraction.regex_classifier import classify_to_devices
            from ..extraction.regex_classifier import classify_document as regex_classify_document
            from ..extraction.vlm_ocr import validate_ocr_quality

            # 1. VLM OCR: PyMuPDF (imagens) + Qwen3-VL (texto por página)
            self.vlm_service.vlm_client.reset_client()
            pages_data, raw_canonical = asyncio.run(
                self.vlm_service.ocr_document(
                    pdf_bytes=pdf_content,
                    document_id=request.document_id,
                    progress_callback=report_progress,
                )
            )

            report_progress("vlm_extraction", 0.30)

            # 2. Idempotency check (offsets nativos devem sobreviver normalize)
            canonical_text = raw_canonical
            normalized_check = normalize_canonical_text(raw_canonical)
            if canonical_text != normalized_check:
                logger.error(
                    f"[{request.document_id}] OFFSET DRIFT: OCR canonical_text "
                    f"diverge de normalize_canonical_text() — offsets seriam inválidos! "
                    f"raw_len={len(canonical_text)} norm_len={len(normalized_check)}"
                )
                raise RuntimeError(
                    "VLM OCR canonical_text diverge de normalize_canonical_text(): "
                    "offsets nativos seriam inválidos."
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

            report_progress("vlm_extraction", 0.40)

            # 4. Converte pages para formato do classifier
            pages_for_classifier = self._convert_pages_to_classifier_format(pages_data)

            # 5. Classificação regex (MESMO classifier da Entrada 1)
            classification_result = regex_classify_document(pages_for_classifier)
            devices = classify_to_devices(pages_for_classifier)
            logger.info(
                f"[{request.document_id}] VLM OCR + RegexClassifier: {len(devices)} dispositivos"
            )

            report_progress("vlm_extraction", 0.55)

            # 5.1. Quality gate OCR
            ocr_warnings = validate_ocr_quality(
                devices, canonical_text, len(pages_data), request.document_id,
            )
            for w in ocr_warnings:
                logger.warning(f"[{request.document_id}] {w}")
                result.quality_issues.append(w)

            # Registra fase
            extract_duration = round(time.perf_counter() - phase_start, 2)
            result.phases.append({
                "name": "vlm_ocr_extraction",
                "duration_seconds": extract_duration,
                "output": (
                    f"Extraídos {len(devices)} dispositivos de "
                    f"{len(pages_data)} páginas via VLM OCR+Regex"
                ),
                "success": True,
                "method": "vlm_ocr+regex",
            })

            # 6. Emit inspection snapshot (Redis) — reutiliza formato regex
            try:
                result.inspection_snapshot = self._emit_regex_inspection_snapshot(
                    classification_result, canonical_text, canonical_hash,
                    extract_duration, request, pages_data,
                    extraction_source="vlm_ocr",
                ) or {}
            except Exception as e:
                logger.warning(f"Failed to emit inspector snapshot: {e}")

            # 7. Converte ClassifiedDevice -> ProcessedChunk (MESMO da Entrada 1)
            chunks = self._regex_to_processed_chunks(
                devices, canonical_text, canonical_hash, request,
            )

            report_progress("vlm_extraction", 0.65)

            # 8. Origin Classification
            from ..classification.origin_classifier import classify_document
            chunks = classify_document(chunks, canonical_text, request.document_id)
            external_count = sum(1 for c in chunks if c.origin_type == "external")
            if external_count > 0:
                logger.info(f"[{request.document_id}] OriginClassifier: {external_count}/{len(chunks)} external")

            # 8.5. Enriquecer retrieval_text para chunks external (proveniência)
            for chunk in chunks:
                if getattr(chunk, "is_external_material", False):
                    ref_name = getattr(chunk, "origin_reference_name", "") or ""
                    ref_id = getattr(chunk, "origin_reference", "") or ""
                    if ref_name:
                        prefix = f"{ref_name} ({ref_id})" if ref_id else ref_name
                        chunk.retrieval_text = f"{prefix}\n{chunk.retrieval_text}"

            report_progress("vlm_extraction", 0.70)

            # 9. Embeddings (se não pular)
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
                    "output": f"Embeddings para {len(chunks)} chunks VLM OCR",
                    "success": True,
                })

            # 10. Artifacts upload
            report_progress("artifacts_upload", 0.88)
            self._phase_artifacts_upload(
                pdf_content, canonical_text, canonical_hash,
                chunks, request, result,
            )
            report_progress("artifacts_upload", 0.94)

            # 11. Valida invariantes do contrato
            validate_chunk_invariants(chunks, request.document_id)

            # 12. Drift register + resultado
            drift_detector.register_run(
                document_id=request.document_id,
                pdf_hash=result.document_hash,
                pipeline_version=app_config.pipeline_version,
                canonical_hash=canonical_hash,
                ingest_run_id=result.ingest_run_id,
            )

            # 13. Manifesto de ingestão
            result.manifest = self._build_manifest(
                chunks, canonical_text, canonical_hash, request.document_id,
            )

            result.chunks = chunks
            result.status = IngestStatus.COMPLETED

            report_progress("completed", 1.0)

            logger.info(
                f"Pipeline VLM OCR concluído: {len(chunks)} chunks, "
                f"{len(devices)} dispositivos, "
                f"manifest: {result.manifest.get('total_spans')} spans"
            )

        except Exception as e:
            logger.error(f"Erro no pipeline VLM OCR: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(
                phase="vlm_ocr_extraction",
                message=str(e),
            ))

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
        extraction_source: str = "pymupdf_native",
    ) -> dict:
        """
        Emite snapshot da classificação regex para o Redis (Inspector).
        Falha silenciosa — não aborta o pipeline.
        Retorna dict do snapshot para inclusão no task.result.
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
            extraction_source=extraction_source,
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

        # --- Forward to VPS (fire-and-forget) ---
        try:
            from ..inspection.vps_forwarder import VpsInspectionForwarder
            forwarder = VpsInspectionForwarder()
            forwarder.forward_full_snapshot(
                task_id=task_id,
                metadata=metadata,
                stages={
                    InspectionStage.REGEX_CLASSIFICATION.value: artifact.model_dump_json(),
                    InspectionStage.PYMUPDF.value: pymupdf_artifact.model_dump_json(),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to forward inspection to VPS: {e}")

        # Return snapshot dict for inclusion in task.result
        from ..config import config as app_config
        return {
            "document_id": request.document_id,
            "run_id": task_id,
            "pipeline_version": getattr(app_config, 'pipeline_version', ''),
            "canonical_hash": canonical_hash,
            "total_pages": len(pages_data),
            "total_devices": len(regex_devices),
            "total_filtered": len(filtered),
            "unclassified_count": len(unclassified),
            "all_checks_pass": all_pass,
            "stages": {
                "pymupdf": pymupdf_artifact.model_dump(),
                "regex_classification": artifact.model_dump(),
            },
        }

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
                result.inspection_snapshot = self._emit_regex_inspection_snapshot(
                    classification_result, canonical_text, canonical_hash,
                    extract_duration, request, pages_data,
                ) or {}
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
        acordao_metadata: Optional[dict] = None,
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

        manifest = {
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

        # Acórdão-specific fields
        if acordao_metadata:
            section_types: Dict[str, int] = {}
            authority_levels: Dict[str, int] = {}
            for c in chunks:
                st = getattr(c, "section_type", "") or ""
                al = getattr(c, "authority_level", "") or ""
                if st:
                    section_types[st] = section_types.get(st, 0) + 1
                if al:
                    authority_levels[al] = authority_levels.get(al, 0) + 1
            manifest["section_types"] = section_types
            manifest["authority_levels"] = authority_levels
            manifest["metadata"] = acordao_metadata

        return manifest

    # =================================================================
    # ACÓRDÃO PIPELINE
    # =================================================================

    def _phase_acordao_extraction(
        self,
        pdf_content: bytes,
        request: IngestRequest,
        result: PipelineResult,
        report_progress,
    ) -> None:
        """Pipeline Acórdão: PyMuPDF + AcordaoParser."""
        phase_start = time.perf_counter()
        try:
            from ..config import config as app_config
            from ..extraction.pymupdf_extractor import PyMuPDFExtractor

            # 1. Extração PyMuPDF
            extractor = PyMuPDFExtractor(dpi=app_config.vlm_page_dpi)
            pages_data, raw_canonical = extractor.extract_pages(pdf_content)

            report_progress("acordao_extraction", 0.30)

            self._process_acordao(
                pages_data, raw_canonical, pdf_content,
                request, result, report_progress,
                phase_start, extraction_source="pymupdf_native",
            )

        except Exception as e:
            logger.error(f"Erro no pipeline Acórdão: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(
                phase="acordao_extraction",
                message=str(e),
            ))

    def _phase_acordao_vlm_extraction(
        self,
        pdf_content: bytes,
        request: IngestRequest,
        result: PipelineResult,
        report_progress,
    ) -> None:
        """Pipeline Acórdão + VLM OCR."""
        phase_start = time.perf_counter()
        try:
            # 1. VLM OCR
            self.vlm_service.vlm_client.reset_client()
            pages_data, raw_canonical = asyncio.run(
                self.vlm_service.ocr_document(
                    pdf_bytes=pdf_content,
                    document_id=request.document_id,
                    progress_callback=report_progress,
                )
            )

            report_progress("acordao_vlm_extraction", 0.30)

            self._process_acordao(
                pages_data, raw_canonical, pdf_content,
                request, result, report_progress,
                phase_start, extraction_source="vlm_ocr",
            )

        except Exception as e:
            logger.error(f"Erro no pipeline Acórdão+VLM: {e}", exc_info=True)
            result.status = IngestStatus.FAILED
            result.errors.append(IngestError(
                phase="acordao_vlm_extraction",
                message=str(e),
            ))

    def _process_acordao(
        self,
        pages_data: list,
        raw_canonical: str,
        pdf_content: bytes,
        request: IngestRequest,
        result: PipelineResult,
        report_progress,
        phase_start: float,
        extraction_source: str = "pymupdf_native",
    ) -> None:
        """
        Lógica compartilhada do pipeline de acórdãos (PyMuPDF e VLM).

        1. Idempotency check
        2. Drift detection
        3. Header parse
        4. AcordaoParser
        5. Inspection snapshot
        6. _acordao_to_processed_chunks
        7. OriginClassifier (simplificado: all self)
        8. Embeddings
        9. Artifacts
        10. Validate + Manifest
        """
        from ..config import config as app_config
        from ..extraction.acordao_header_parser import AcordaoHeaderParser
        from ..extraction.acordao_parser import AcordaoParser

        # 2. Idempotency check
        canonical_text = raw_canonical
        normalized_check = normalize_canonical_text(raw_canonical)
        if canonical_text != normalized_check:
            logger.error(
                f"[{request.document_id}] OFFSET DRIFT: canonical_text "
                f"diverge de normalize_canonical_text() — offsets inválidos! "
                f"raw_len={len(canonical_text)} norm_len={len(normalized_check)}"
            )
            raise RuntimeError(
                "canonical_text diverge de normalize_canonical_text(): "
                "offsets nativos seriam inválidos."
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

        report_progress("acordao_extraction", 0.40)

        # 4. Header parse
        header_parser = AcordaoHeaderParser()
        header_metadata = header_parser.parse_header(canonical_text)

        # 5. Build page_boundaries
        page_boundaries = []
        for pg in pages_data:
            if pg.blocks:
                pg_start = pg.blocks[0].char_start
                pg_end = pg.blocks[-1].char_end
            else:
                pg_start = 0
                pg_end = 0
            page_boundaries.append((pg_start, pg_end))

        # 6. AcordaoParser
        parser = AcordaoParser()
        acordao_devices = parser.parse(canonical_text, page_boundaries)
        logger.info(
            f"[{request.document_id}] AcordaoParser: {len(acordao_devices)} dispositivos"
        )

        report_progress("acordao_extraction", 0.55)

        # Registra fase
        extract_duration = round(time.perf_counter() - phase_start, 2)
        result.phases.append({
            "name": "acordao_extraction",
            "duration_seconds": extract_duration,
            "output": (
                f"Extraídos {len(acordao_devices)} dispositivos de "
                f"{len(pages_data)} páginas via AcordaoParser ({extraction_source})"
            ),
            "success": True,
            "method": extraction_source,
        })

        # 7. Inspection snapshot (reutiliza formato simplificado para acórdãos)
        try:
            self._emit_acordao_inspection_snapshot(
                acordao_devices, canonical_text, canonical_hash,
                extract_duration, request, pages_data,
                extraction_source=extraction_source,
            )
        except Exception as e:
            logger.warning(f"Failed to emit acordao inspector snapshot: {e}")

        # 8. Section chunking: AcordaoDevice → ParsedSection → AcordaoChunk → ProcessedChunk
        from ..extraction.acordao_chunker import AcordaoChunker, build_sections

        sections = build_sections(acordao_devices, canonical_text, header_metadata)
        chunker = AcordaoChunker()
        acordao_chunks = chunker.chunk(
            sections=sections,
            document_id=request.document_id,
            canonical_hash=canonical_hash,
            metadata={
                "numero": header_metadata.get("numero", request.numero),
                "ano": header_metadata.get("ano", str(request.ano)),
                "colegiado": header_metadata.get("colegiado", ""),
                "processo": header_metadata.get("processo", ""),
                "relator": header_metadata.get("relator", ""),
                "data_sessao": header_metadata.get("data_sessao", ""),
                "natureza": header_metadata.get("natureza", ""),
                "resultado": header_metadata.get("resultado", ""),
            },
        )
        chunks = self._acordao_to_processed_chunks(
            acordao_chunks, canonical_text, canonical_hash,
            request, header_metadata,
        )

        report_progress("acordao_extraction", 0.65)

        # 9. OriginClassifier: não se aplica a acórdãos (campos removidos do schema acordaos_v1)

        report_progress("acordao_extraction", 0.70)

        # 10. Embeddings
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
                "output": f"Embeddings para {len(chunks)} chunks Acórdão",
                "success": True,
            })

        # 11. Artifacts upload
        report_progress("artifacts_upload", 0.88)
        self._phase_artifacts_upload(
            pdf_content, canonical_text, canonical_hash,
            chunks, request, result,
        )
        report_progress("artifacts_upload", 0.94)

        # 12. Valida invariantes do contrato
        validate_chunk_invariants(chunks, request.document_id)

        # 13. Drift register
        drift_detector.register_run(
            document_id=request.document_id,
            pdf_hash=result.document_hash,
            pipeline_version=app_config.pipeline_version,
            canonical_hash=canonical_hash,
            ingest_run_id=result.ingest_run_id,
        )

        # 14. Manifesto de ingestão
        result.manifest = self._build_manifest(
            chunks, canonical_text, canonical_hash, request.document_id,
            acordao_metadata=header_metadata,
        )

        result.chunks = chunks
        result.status = IngestStatus.COMPLETED

        report_progress("completed", 1.0)

        logger.info(
            f"Pipeline Acórdão concluído: {len(chunks)} chunks, "
            f"{len(acordao_devices)} dispositivos, "
            f"manifest: {result.manifest.get('total_spans')} spans"
        )

    def _acordao_to_processed_chunks(
        self,
        acordao_chunks,
        canonical_text: str,
        canonical_hash: str,
        request: IngestRequest,
        header_metadata: dict,
    ) -> List[ProcessedChunk]:
        """
        Converte List[AcordaoChunk] -> List[ProcessedChunk].

        Prefixo "acordaos:" em node_id.
        """
        chunks = []

        colegiado = header_metadata.get("colegiado", getattr(request, "colegiado", "") or "")
        processo = header_metadata.get("processo", getattr(request, "processo", "") or "")
        relator = header_metadata.get("relator", getattr(request, "relator", "") or "")
        data_sessao = header_metadata.get("data_sessao", getattr(request, "data_sessao", "") or "")

        for ac in acordao_chunks:
            chunk_id = f"{request.document_id}#{ac.span_id}"
            node_id = f"acordaos:{chunk_id}"

            pc = ProcessedChunk(
                node_id=node_id,
                chunk_id=chunk_id,
                parent_node_id="",
                span_id=ac.span_id,
                device_type="section",
                chunk_level="section",
                text=ac.text or "",
                parent_text="",
                retrieval_text=ac.retrieval_text,
                document_id=request.document_id,
                tipo_documento=request.tipo_documento,
                numero=request.numero,
                ano=request.ano,
                article_number="",
                citations=[],
                canonical_start=ac.canonical_start,
                canonical_end=ac.canonical_end,
                canonical_hash=canonical_hash,
                page_number=ac.page_number,
                bbox=[],
                bbox_img=[],
                img_width=0,
                img_height=0,
                confidence=1.0,
                is_cross_page=False,
                bbox_spans=[],
                colegiado=colegiado,
                processo=processo,
                relator=relator,
                data_sessao=data_sessao,
                section_type=ac.section_type,
                authority_level=ac.authority_level,
                section_path=ac.section_path,
            )
            chunks.append(pc)

        logger.info(f"Acórdão -> ProcessedChunk: {len(chunks)} chunks gerados")
        return chunks

    def _emit_acordao_inspection_snapshot(
        self,
        devices,
        canonical_text: str,
        canonical_hash: str,
        duration_ms: float,
        request: IngestRequest,
        pages_data: list,
        extraction_source: str = "pymupdf_native",
    ) -> None:
        """
        Emite snapshot da classificação de acórdão para o Redis (Inspector).
        Reutiliza o modelo RegexClassificationArtifact com dados do AcordaoParser.
        """
        import unicodedata
        from ..inspection.storage import InspectionStorage
        from ..inspection.models import (
            InspectionStage,
            InspectionMetadata,
            InspectionStatus,
            RegexClassificationArtifact,
            RegexDevice,
            RegexClassificationStats,
            RegexIntegrityChecks,
            RegexOffsetCheck,
            PyMuPDFArtifact,
            PyMuPDFPageResult,
            PyMuPDFBlock,
            BBox,
        )

        storage = InspectionStorage()

        # Build regex devices from acordao devices
        regex_devices = []
        by_type: Dict[str, int] = {}
        for d in devices:
            by_type[d.device_type] = by_type.get(d.device_type, 0) + 1
            regex_devices.append(RegexDevice(
                span_id=d.span_id,
                device_type=d.device_type,
                identifier=d.identifier or "",
                parent_span_id=d.parent_span_id or "",
                children_span_ids=d.children_span_ids,
                hierarchy_depth=d.hierarchy_depth,
                text=d.text,
                text_preview=d.text_preview,
                char_start=d.char_start,
                char_end=d.char_end,
                page_number=d.page_number,
                bbox=d.bbox,
            ))

        stats = RegexClassificationStats(
            total_blocks=len(devices),
            devices=len(devices),
            filtered=0,
            unclassified=0,
            by_device_type=by_type,
            by_filter_type={},
            max_hierarchy_depth=max((d.hierarchy_depth for d in devices), default=0),
        )

        # Integrity checks
        from ..chunking.canonical_offsets import normalize_canonical_text as norm_ct

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

        norm_idempotent = canonical_text == norm_ct(canonical_text)
        trailing_violations = sum(
            1 for line in canonical_text.split("\n")
            if line != line.rstrip()
        )
        is_nfc = unicodedata.is_normalized("NFC", canonical_text)
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
            filtered=[],
            unclassified=[],
            stats=stats,
            checks=checks,
            canonical_text=canonical_text,
            canonical_hash=canonical_hash,
            canonical_length=len(canonical_text),
            duration_ms=duration_ms * 1000,
            extraction_source=extraction_source,
        )

        # PyMuPDF artifact
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

        # Save to Redis
        task_id = f"acordao_{request.document_id}_{int(time.time())}"

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
            f"Acordao inspector snapshot emitted: {task_id} "
            f"({len(regex_devices)} devices, checks={'PASS' if all_pass else 'FAIL'})"
        )

        # --- Forward to VPS (fire-and-forget) ---
        try:
            from ..inspection.vps_forwarder import VpsInspectionForwarder
            forwarder = VpsInspectionForwarder()
            forwarder.forward_full_snapshot(
                task_id=task_id,
                metadata=metadata,
                stages={
                    InspectionStage.REGEX_CLASSIFICATION.value: artifact.model_dump_json(),
                    InspectionStage.PYMUPDF.value: pymupdf_artifact.model_dump_json(),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to forward acordao inspection to VPS: {e}")

    def _phase_milvus_sink(self, materialized, request: IngestRequest, result: PipelineResult):
        """Fase 5.5: Inserir chunks no Milvus remoto (se MILVUS_HOST configurado)."""
        phase_start = time.perf_counter()
        try:
            from pymilvus import connections, Collection
            import json

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

                # Serializa citações para persistência no Milvus (Ledger Pattern)
                citations_json = json.dumps(citations, ensure_ascii=False) if citations else ""

                # Device type
                device_type = getattr(chunk, 'device_type', 'article')

                # Prepara aliases como JSON string
                aliases = getattr(chunk, 'aliases', []) or []
                aliases_str = json.dumps(aliases, ensure_ascii=False)

                # Bbox: schema usa 4 campos Float separados
                bbox = getattr(chunk, 'bbox', []) or []

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
                    "citations": citations_json,
                    # VLM: Campos do pipeline Qwen3-VL + PyMuPDF
                    "page_number": getattr(chunk, 'page_number', -1),
                    "bbox_x0": float(bbox[0]) if len(bbox) >= 4 else 0.0,
                    "bbox_y0": float(bbox[1]) if len(bbox) >= 4 else 0.0,
                    "bbox_x1": float(bbox[2]) if len(bbox) >= 4 else 0.0,
                    "bbox_y1": float(bbox[3]) if len(bbox) >= 4 else 0.0,
                    # Origin Classifier
                    "origin_type": getattr(chunk, 'origin_type', 'self'),
                    "origin_reference": getattr(chunk, 'origin_reference', ''),
                    "origin_reference_name": getattr(chunk, 'origin_reference_name', ''),
                    "is_external_material": getattr(chunk, 'is_external_material', False),
                    "origin_confidence": getattr(chunk, 'origin_confidence', 'high'),
                    "origin_reason": getattr(chunk, 'origin_reason', ''),
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
