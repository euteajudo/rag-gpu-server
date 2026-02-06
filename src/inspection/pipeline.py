"""
Pipeline dry-run de inspeção.

Executa as mesmas fases do pipeline de extração VLM, mas:
- NÃO indexa no Milvus
- NÃO gera embeddings
- Salva artefatos intermediários no Redis via InspectionStorage
- Reporta progresso via callback
- Gera artefatos visuais (imagens anotadas com bboxes coloridos)

Fases:
1. PyMuPDF  — blocos de texto com bboxes (determinístico)
2. VLM      — Qwen3-VL classifica dispositivos legais nas imagens
3. Reconciliação — matching PyMuPDF ↔ VLM, texto SEMPRE do PyMuPDF
4. Integridade   — validações no canonical_text
5. Chunks        — preview dos chunks que seriam criados na ingestão
"""

import asyncio
import hashlib
import logging
import re
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
    ReconciliationMatch,
    ReconciliationPageResult,
    ReconciliationStats,
    VLMArtifact,
    VLMElement,
    VLMPageResult,
)
from .page_renderer import PageRenderer
from .storage import InspectionStorage

logger = logging.getLogger(__name__)


# Mapeamento tipo VLM → device_type do chunk
_DEVICE_TYPE_MAP = {
    "artigo": "article",
    "paragrafo": "paragraph",
    "inciso": "inciso",
    "alinea": "alinea",
    "caput": "caput",
}

# Profundidade hierárquica por tipo
_DEVICE_DEPTH = {
    "artigo": 1,
    "paragrafo": 2,
    "inciso": 3,
    "alinea": 4,
    "caput": 2,
}


class InspectionPipeline:
    """
    Pipeline dry-run para inspeção de documentos.

    Executa fases de processamento VLM (PyMuPDF + Qwen3-VL),
    salva artefatos intermediários, mas NÃO indexa no Milvus.
    Usado para revisão humana antes da ingestão definitiva.
    """

    def __init__(self, storage: Optional[InspectionStorage] = None):
        self._storage = storage or InspectionStorage()

        # Lazy-loaded components
        self._vlm_client = None
        self._pymupdf_extractor = None

        # Dados compartilhados entre fases (dentro de um run)
        self._page_texts: list[str] = []

    # =========================================================================
    # Lazy properties
    # =========================================================================

    @property
    def vlm_client(self):
        if self._vlm_client is None:
            from ..config import config
            from ..extraction.vlm_client import VLMClient
            self._vlm_client = VLMClient(
                base_url=config.vllm_base_url,
                model=config.vllm_model,
                max_retries=config.vlm_max_retries,
            )
        return self._vlm_client

    @property
    def pymupdf_extractor(self):
        if self._pymupdf_extractor is None:
            from ..config import config
            from ..extraction.pymupdf_extractor import PyMuPDFExtractor
            self._pymupdf_extractor = PyMuPDFExtractor(dpi=config.vlm_page_dpi)
        return self._pymupdf_extractor

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
            report("pymupdf", 0.25)

            # Fase 2: VLM — Qwen3-VL extrai estrutura de cada página
            report("vlm", 0.27)
            vlm_artifact = self._phase_vlm(task_id, pdf_bytes, pymupdf_artifact)
            report("vlm", 0.55)

            # Fase 3: Reconciliação — matching PyMuPDF + VLM
            report("reconciliation", 0.57)
            recon_artifact = self._phase_reconciliation(
                task_id, pdf_bytes, pymupdf_artifact, vlm_artifact,
            )
            report("reconciliation", 0.70)

            # Fase 4: Integridade — valida a extração
            report("integrity", 0.72)
            integrity_artifact = self._phase_integrity(
                task_id, recon_artifact,
            )
            report("integrity", 0.80)

            # Fase 5: Chunks — preview dos chunks que seriam criados
            report("chunks", 0.82)
            chunks_artifact = self._phase_chunks_preview(
                task_id, recon_artifact, vlm_artifact,
                document_id, tipo_documento, numero, ano,
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
    # Fase 2: VLM — Extração de estrutura via Qwen3-VL
    # =========================================================================

    def _phase_vlm(
        self,
        task_id: str,
        pdf_bytes: bytes,
        pymupdf_artifact: PyMuPDFArtifact,
    ) -> VLMArtifact:
        """
        Envia imagem de cada página ao Qwen3-VL para extração de estrutura.

        O VLM recebe APENAS imagens e retorna classificação:
        tipo, identificador, bbox, hierarquia, confidence.
        O texto final vem SEMPRE do PyMuPDF (Fase 1/3).
        """
        phase_start = time.perf_counter()
        logger.info("Inspeção Fase 2: VLM — extraindo estrutura com Qwen3-VL...")

        # Extrai páginas com PyMuPDFExtractor (300 DPI para VLM)
        pages_data = self.pymupdf_extractor.extract_pages(pdf_bytes)

        # Armazena textos das páginas para Fase 3 (canonical_text)
        self._page_texts = [pd.text for pd in pages_data]

        # Envia cada página ao VLM (async de contexto sync)
        vlm_raw_results: list[tuple[int, dict]] = []  # (page_0idx, result)

        loop = asyncio.new_event_loop()
        try:
            for pd in pages_data:
                page_0idx = pd.page_number - 1  # PyMuPDFExtractor é 1-indexed
                logger.info(f"  VLM: processando página {pd.page_number}/{len(pages_data)}")
                try:
                    result = loop.run_until_complete(
                        self.vlm_client.extract_page(image_base64=pd.image_base64)
                    )
                    vlm_raw_results.append((page_0idx, result))
                    logger.info(
                        f"  VLM página {pd.page_number}: "
                        f"{len(result.get('devices', []))} dispositivos"
                    )
                except Exception as e:
                    logger.warning(f"  VLM falhou na página {pd.page_number}: {e}")
                    vlm_raw_results.append((page_0idx, {"devices": []}))
        finally:
            loop.close()

        # Converte para VLMElement + renderiza imagens anotadas
        vlm_pages: list[VLMPageResult] = []
        total_elements = 0
        max_depth = 0

        # Mapa global identifier → element_id (para resolver parent cross-page)
        identifier_to_element_id: dict[str, str] = {}
        all_elements: list[VLMElement] = []

        with PageRenderer(pdf_bytes) as renderer:
            for page_0idx, raw_result in vlm_raw_results:
                # Dimensões da página em PDF points
                if page_0idx < len(pymupdf_artifact.pages):
                    page_w = pymupdf_artifact.pages[page_0idx].width
                    page_h = pymupdf_artifact.pages[page_0idx].height
                else:
                    page_w, page_h = 595.0, 842.0  # A4 default

                elements: list[VLMElement] = []
                bboxes_for_render: list[tuple[BBox, str]] = []

                for i, device in enumerate(raw_result.get("devices", [])):
                    element_type = device.get("device_type", "unknown")
                    identifier = device.get("identifier", "")
                    element_id = f"{element_type}_{page_0idx}_{i}"

                    # Bbox: VLM retorna normalizado [0-1], converte para PDF points
                    bbox_raw = device.get("bbox", [])
                    bbox = None
                    if len(bbox_raw) == 4:
                        try:
                            bbox = BBox(
                                x0=float(bbox_raw[0]) * page_w,
                                y0=float(bbox_raw[1]) * page_h,
                                x1=float(bbox_raw[2]) * page_w,
                                y1=float(bbox_raw[3]) * page_h,
                            )
                            bboxes_for_render.append((bbox, "vlm"))
                        except (ValueError, TypeError):
                            pass

                    depth = _DEVICE_DEPTH.get(element_type, 0)
                    max_depth = max(max_depth, depth)

                    elem = VLMElement(
                        element_id=element_id,
                        element_type=element_type,
                        text=device.get("text", ""),
                        bbox=bbox,
                        confidence=float(device.get("confidence", 0.0)),
                        page=page_0idx,
                        parent_id=device.get("parent_identifier", "") or None,
                        children_ids=[],
                    )
                    elements.append(elem)
                    all_elements.append(elem)

                    if identifier:
                        identifier_to_element_id[identifier] = element_id

                # Renderiza página com bboxes VLM (verde)
                image_b64 = renderer.render_page_base64(page_0idx, bboxes_for_render)

                vlm_pages.append(VLMPageResult(
                    page_number=page_0idx,
                    elements=elements,
                    image_base64=image_b64,
                ))
                total_elements += len(elements)

        # Segunda passada: resolve parent_id (identifier → element_id)
        for elem in all_elements:
            if elem.parent_id and elem.parent_id in identifier_to_element_id:
                parent_eid = identifier_to_element_id[elem.parent_id]
                elem.parent_id = parent_eid
                # Atualiza children_ids do pai
                for parent_elem in all_elements:
                    if parent_elem.element_id == parent_eid:
                        if elem.element_id not in parent_elem.children_ids:
                            parent_elem.children_ids.append(elem.element_id)
                        break

        duration_ms = (time.perf_counter() - phase_start) * 1000
        artifact = VLMArtifact(
            pages=vlm_pages,
            total_elements=total_elements,
            total_pages=len(vlm_pages),
            hierarchy_depth=max_depth,
            duration_ms=duration_ms,
        )

        self._storage.save_artifact(
            task_id, InspectionStage.VLM, artifact.model_dump_json(),
        )
        logger.info(
            f"Inspeção Fase 2: VLM — {total_elements} elementos, "
            f"{len(vlm_pages)} páginas, depth={max_depth} em {duration_ms:.0f}ms"
        )
        return artifact

    # =========================================================================
    # Fase 3: Reconciliação — Matching PyMuPDF ↔ VLM
    # =========================================================================

    def _phase_reconciliation(
        self,
        task_id: str,
        pdf_bytes: bytes,
        pymupdf_artifact: PyMuPDFArtifact,
        vlm_artifact: VLMArtifact,
    ) -> ReconciliationArtifact:
        """
        Reconcilia blocos PyMuPDF com elementos VLM.

        - Texto final = SEMPRE do PyMuPDF (determinístico)
        - Classificação (tipo, hierarquia) = do VLM
        - Matching por IoU de bounding boxes
        - Constrói canonical_text a partir do PyMuPDF
        """
        phase_start = time.perf_counter()
        logger.info("Inspeção Fase 3: Reconciliação — matching PyMuPDF + VLM...")

        # 1. Constrói canonical_text a partir dos textos PyMuPDF (get_text("text"))
        from ..utils.canonical_utils import normalize_canonical_text, compute_canonical_hash

        raw_canonical = "\n".join(self._page_texts) if self._page_texts else ""
        canonical_text = normalize_canonical_text(raw_canonical)
        canonical_hash = compute_canonical_hash(canonical_text)

        logger.info(
            f"  canonical_text: {len(canonical_text)} chars, hash={canonical_hash[:16]}..."
        )

        # 2. Matching por página: PyMuPDF blocks ↔ VLM elements via bbox IoU
        recon_pages: list[ReconciliationPageResult] = []
        total_matches = 0
        exact_matches = 0
        partial_matches = 0
        conflicts = 0
        unmatched_pymupdf_count = 0
        unmatched_vlm_count = 0

        with PageRenderer(pdf_bytes) as renderer:
            for page_0idx in range(max(len(pymupdf_artifact.pages), len(vlm_artifact.pages))):
                # Blocos PyMuPDF desta página
                pymupdf_page = None
                if page_0idx < len(pymupdf_artifact.pages):
                    pymupdf_page = pymupdf_artifact.pages[page_0idx]
                blocks = pymupdf_page.blocks if pymupdf_page else []
                page_w = pymupdf_page.width if pymupdf_page else 595.0
                page_h = pymupdf_page.height if pymupdf_page else 842.0

                # Elementos VLM desta página
                vlm_page = None
                if page_0idx < len(vlm_artifact.pages):
                    vlm_page = vlm_artifact.pages[page_0idx]
                elements = vlm_page.elements if vlm_page else []

                matches: list[ReconciliationMatch] = []
                matched_block_indices: set[int] = set()
                matched_vlm_ids: set[str] = set()
                bboxes_for_render: list[tuple[BBox, str]] = []

                # Para cada elemento VLM, encontra melhor bloco PyMuPDF por IoU
                for elem in elements:
                    if elem.bbox is None:
                        # VLM não retornou bbox — sem match possível
                        matches.append(ReconciliationMatch(
                            pymupdf_block_index=-1,
                            vlm_element_id=elem.element_id,
                            match_quality="unmatched_vlm",
                            text_pymupdf="",
                            text_vlm=elem.text,
                            text_reconciled="",
                            bbox_overlap=0.0,
                            page=page_0idx,
                        ))
                        unmatched_vlm_count += 1
                        continue

                    best_iou = 0.0
                    best_block_idx = -1
                    best_block = None

                    for block in blocks:
                        if block.block_index in matched_block_indices:
                            continue
                        iou = _compute_bbox_iou(block.bbox, elem.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_block_idx = block.block_index
                            best_block = block

                    if best_iou >= 0.1 and best_block is not None:
                        # Match encontrado
                        matched_block_indices.add(best_block_idx)
                        matched_vlm_ids.add(elem.element_id)

                        if best_iou >= 0.5:
                            quality = "exact"
                            exact_matches += 1
                            bbox_color = "match"
                        else:
                            quality = "partial"
                            partial_matches += 1
                            bbox_color = "match"

                        total_matches += 1

                        # Texto reconciliado = SEMPRE do PyMuPDF
                        matches.append(ReconciliationMatch(
                            pymupdf_block_index=best_block_idx,
                            vlm_element_id=elem.element_id,
                            match_quality=quality,
                            text_pymupdf=best_block.text,
                            text_vlm=elem.text,
                            text_reconciled=best_block.text,
                            bbox_overlap=round(best_iou, 4),
                            page=page_0idx,
                        ))

                        # Usa bbox do match para renderização
                        if elem.bbox:
                            bboxes_for_render.append((elem.bbox, bbox_color))

                    else:
                        # VLM sem match no PyMuPDF
                        matches.append(ReconciliationMatch(
                            pymupdf_block_index=-1,
                            vlm_element_id=elem.element_id,
                            match_quality="unmatched_vlm",
                            text_pymupdf="",
                            text_vlm=elem.text,
                            text_reconciled="",
                            bbox_overlap=0.0,
                            page=page_0idx,
                        ))
                        unmatched_vlm_count += 1

                        if elem.bbox:
                            bboxes_for_render.append((elem.bbox, "conflict"))

                # Blocos PyMuPDF sem match no VLM
                for block in blocks:
                    if block.block_index not in matched_block_indices:
                        matches.append(ReconciliationMatch(
                            pymupdf_block_index=block.block_index,
                            vlm_element_id="",
                            match_quality="unmatched_pymupdf",
                            text_pymupdf=block.text,
                            text_vlm="",
                            text_reconciled=block.text,
                            bbox_overlap=0.0,
                            page=page_0idx,
                        ))
                        unmatched_pymupdf_count += 1
                        bboxes_for_render.append((block.bbox, "pymupdf"))

                # Renderiza página com bboxes de reconciliação
                image_b64 = ""
                if page_0idx < renderer.page_count:
                    image_b64 = renderer.render_page_base64(page_0idx, bboxes_for_render)

                recon_pages.append(ReconciliationPageResult(
                    page_number=page_0idx,
                    matches=matches,
                    image_base64=image_b64,
                ))

        # Estatísticas
        total_pymupdf = sum(len(p.blocks) for p in pymupdf_artifact.pages)
        total_vlm = vlm_artifact.total_elements
        stats = ReconciliationStats(
            total_matches=total_matches,
            exact_matches=exact_matches,
            partial_matches=partial_matches,
            conflicts=conflicts,
            unmatched_pymupdf=unmatched_pymupdf_count,
            unmatched_vlm=unmatched_vlm_count,
            coverage_pymupdf=round(total_matches / max(total_pymupdf, 1), 4),
            coverage_vlm=round(total_matches / max(total_vlm, 1), 4),
        )

        duration_ms = (time.perf_counter() - phase_start) * 1000
        artifact = ReconciliationArtifact(
            pages=recon_pages,
            stats=stats,
            canonical_text=canonical_text,
            duration_ms=duration_ms,
        )

        self._storage.save_artifact(
            task_id, InspectionStage.RECONCILIATION, artifact.model_dump_json(),
        )
        logger.info(
            f"Inspeção Fase 3: Reconciliação — {total_matches} matches "
            f"({exact_matches} exact, {partial_matches} partial), "
            f"{unmatched_pymupdf_count} blocos sem VLM, "
            f"{unmatched_vlm_count} VLM sem bloco, "
            f"{len(canonical_text)} chars canonical em {duration_ms:.0f}ms"
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

        # Check 5: Reconciliação tem matches (VLM funcionou)
        stats = recon_artifact.stats
        has_vlm_matches = stats.total_matches > 0
        checks.append(IntegrityCheck(
            check_name="vlm_matches",
            passed=has_vlm_matches,
            message=f"VLM: {stats.total_matches} matches ({stats.exact_matches} exact)" if has_vlm_matches
                    else "VLM não produziu matches — verifique se o vLLM está rodando",
            details={
                "total_matches": stats.total_matches,
                "exact_matches": stats.exact_matches,
                "partial_matches": stats.partial_matches,
                "coverage_vlm": stats.coverage_vlm,
            },
        ))

        if not has_articles:
            warnings.append("Nenhum artigo detectado — documento pode não ser legislação")
        if not has_vlm_matches:
            warnings.append("VLM sem matches — chunks serão gerados apenas com texto PyMuPDF")

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
        recon_artifact: ReconciliationArtifact,
        vlm_artifact: VLMArtifact,
        document_id: str,
        tipo_documento: str,
        numero: str,
        ano: int,
    ) -> ChunksPreviewArtifact:
        """
        Gera preview dos chunks que SERIAM criados na ingestão.

        Cada dispositivo VLM com match no PyMuPDF vira um chunk.
        Texto = do PyMuPDF (reconciliado). Classificação = do VLM.
        """
        phase_start = time.perf_counter()
        logger.info("Inspeção Fase 5: Chunks — gerando preview a partir do VLM...")

        chunks_preview: list[ChunkPreview] = []
        articles_count = 0
        paragraphs_count = 0
        incisos_count = 0
        alineas_count = 0
        max_depth = 0

        collection_prefix = "leis"

        # Coleta todos os matches com texto reconciliado + classificação VLM
        # Mapeia vlm_element_id → VLMElement para obter classificação
        vlm_element_map: dict[str, VLMElement] = {}
        for vlm_page in vlm_artifact.pages:
            for elem in vlm_page.elements:
                vlm_element_map[elem.element_id] = elem

        # Computa offsets de página no canonical_text para offsets aproximados
        page_start_offsets: list[int] = []
        offset = 0
        for page_text in self._page_texts:
            page_start_offsets.append(offset)
            offset += len(page_text) + 1  # +1 para "\n" entre páginas

        canonical_text = recon_artifact.canonical_text

        # Para cada match com VLM, gera um ChunkPreview
        # Agrupa por vlm_element_id (evita duplicatas)
        seen_element_ids: set[str] = set()
        all_chunk_data: list[dict] = []

        for recon_page in recon_artifact.pages:
            for match in recon_page.matches:
                # Só gera chunk para matches com VLM (tem classificação)
                if not match.vlm_element_id or match.vlm_element_id in seen_element_ids:
                    continue
                if match.match_quality in ("unmatched_vlm",):
                    continue
                if not match.text_reconciled.strip():
                    continue

                seen_element_ids.add(match.vlm_element_id)
                elem = vlm_element_map.get(match.vlm_element_id)
                if elem is None:
                    continue

                # Tipo e span_id
                device_type_vlm = elem.element_type.lower()
                device_type_chunk = _DEVICE_TYPE_MAP.get(device_type_vlm, device_type_vlm)
                identifier = _extract_identifier_from_element(elem)
                parent_identifier = _extract_parent_identifier(elem, vlm_element_map)

                span_id = _device_to_span_id(
                    device_type_vlm, identifier, parent_identifier,
                )
                node_id = f"{collection_prefix}:{document_id}#{span_id}"
                chunk_id = f"{document_id}#{span_id}"

                # Parent node_id
                parent_node_id = ""
                if parent_identifier:
                    parent_span_id = _parent_to_span_id(parent_identifier)
                    if parent_span_id:
                        parent_node_id = f"{collection_prefix}:{document_id}#{parent_span_id}"

                # Offsets canônicos (aproximado: busca texto no canonical)
                canonical_start = -1
                canonical_end = -1
                text_snippet = match.text_reconciled[:80]  # Primeiros 80 chars
                if text_snippet and canonical_text:
                    pos = canonical_text.find(text_snippet)
                    if pos >= 0:
                        canonical_start = pos
                        canonical_end = pos + len(match.text_reconciled)

                # Chunk level
                chunk_level = "article" if device_type_vlm == "artigo" else "device"

                # Depth
                depth = _DEVICE_DEPTH.get(device_type_vlm, 0)
                max_depth = max(max_depth, depth)

                all_chunk_data.append({
                    "span_id": span_id,
                    "node_id": node_id,
                    "chunk_id": chunk_id,
                    "parent_node_id": parent_node_id,
                    "device_type": device_type_chunk,
                    "chunk_level": chunk_level,
                    "text": match.text_reconciled,
                    "canonical_start": canonical_start,
                    "canonical_end": canonical_end,
                })

                # Conta por tipo
                if device_type_vlm == "artigo":
                    articles_count += 1
                elif device_type_vlm == "paragrafo":
                    paragraphs_count += 1
                elif device_type_vlm == "inciso":
                    incisos_count += 1
                elif device_type_vlm == "alinea":
                    alineas_count += 1

        # Calcula children_count para cada chunk
        chunk_id_set = {cd["chunk_id"] for cd in all_chunk_data}
        for cd in all_chunk_data:
            children_count = sum(
                1 for other in all_chunk_data
                if other["parent_node_id"] == cd["node_id"]
            )
            chunks_preview.append(ChunkPreview(
                node_id=cd["node_id"],
                chunk_id=cd["chunk_id"],
                parent_node_id=cd["parent_node_id"],
                span_id=cd["span_id"],
                device_type=cd["device_type"],
                chunk_level=cd["chunk_level"],
                text=cd["text"],
                canonical_start=cd["canonical_start"],
                canonical_end=cd["canonical_end"],
                children_count=children_count,
            ))

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


# =============================================================================
# Funções utilitárias (fora da classe)
# =============================================================================


def _compute_bbox_iou(bbox_a: BBox, bbox_b: BBox) -> float:
    """
    Calcula Intersection over Union entre dois bounding boxes.

    Ambos devem estar no mesmo espaço de coordenadas (PDF points).
    """
    x_left = max(bbox_a.x0, bbox_b.x0)
    y_top = max(bbox_a.y0, bbox_b.y0)
    x_right = min(bbox_a.x1, bbox_b.x1)
    y_bottom = min(bbox_a.y1, bbox_b.y1)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max((bbox_a.x1 - bbox_a.x0) * (bbox_a.y1 - bbox_a.y0), 1e-6)
    area_b = max((bbox_b.x1 - bbox_b.x0) * (bbox_b.y1 - bbox_b.y0), 1e-6)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _roman_to_int(s: str) -> int:
    """Converte numeral romano para inteiro."""
    roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    s = s.upper().strip()
    result = 0
    for i, c in enumerate(s):
        if c not in roman:
            try:
                return int(s)
            except ValueError:
                return 0
        if i + 1 < len(s) and roman.get(c, 0) < roman.get(s[i + 1], 0):
            result -= roman[c]
        else:
            result += roman[c]
    return result


def _extract_identifier_from_element(elem: VLMElement) -> str:
    """Extrai o identificador textual de um VLMElement."""
    # O texto do VLM geralmente começa com o identificador
    # Mas o element_id tem format "tipo_page_idx", não serve
    # Precisamos extrair do texto
    text = elem.text.strip()
    element_type = elem.element_type.lower()

    if element_type == "artigo":
        match = re.match(r'(Art\.?\s*\d+[º°]?)', text, re.IGNORECASE)
        if match:
            return match.group(1)
    elif element_type == "paragrafo":
        match = re.match(r'(§\s*\d+[º°]?)', text)
        if match:
            return match.group(1)
        if text.lower().startswith("parágrafo único"):
            return "Parágrafo único"
    elif element_type == "inciso":
        match = re.match(r'([IVXLC]+)\s*[-–—.]', text)
        if match:
            return match.group(1)
    elif element_type == "alinea":
        match = re.match(r'([a-z])\)', text)
        if match:
            return match.group(1) + ")"

    # Fallback: primeiros caracteres
    return text[:20] if text else ""


def _extract_parent_identifier(
    elem: VLMElement,
    vlm_element_map: dict[str, VLMElement],
) -> str:
    """Extrai o identificador textual do pai de um VLMElement."""
    if not elem.parent_id:
        return ""
    # parent_id pode ser um element_id (se resolvido) ou um identifier original
    parent = vlm_element_map.get(elem.parent_id)
    if parent:
        return _extract_identifier_from_element(parent)
    # Se não encontrou no mapa, parent_id pode ser o identifier original
    return elem.parent_id


def _device_to_span_id(
    device_type: str,
    identifier: str,
    parent_identifier: str,
) -> str:
    """
    Converte tipo + identificador VLM para span_id.

    Formato: ART-005, PAR-005-1, INC-005-III, ALI-005-III-a
    """
    dtype = device_type.lower()

    if dtype == "artigo":
        match = re.search(r"(\d+)", identifier)
        num = match.group(1) if match else "000"
        return f"ART-{num.zfill(3)}"

    elif dtype == "paragrafo":
        match = re.search(r"(\d+)", identifier)
        num = match.group(1) if match else "0"
        parent_match = re.search(r"(\d+)", parent_identifier)
        parent_num = parent_match.group(1) if parent_match else "000"
        return f"PAR-{parent_num.zfill(3)}-{num}"

    elif dtype == "inciso":
        parent_match = re.search(r"(\d+)", parent_identifier)
        parent_num = parent_match.group(1) if parent_match else "000"
        inc_num = _roman_to_int(identifier.strip().rstrip(").-"))
        return f"INC-{parent_num.zfill(3)}-{inc_num}"

    elif dtype == "alinea":
        parent_match = re.search(r"(\d+)", parent_identifier)
        if parent_match:
            parent_num = parent_match.group(1).zfill(3)
        else:
            parent_num = str(_roman_to_int(parent_identifier.strip()))
        letter = re.search(r"([a-z])", identifier.lower())
        letter_str = letter.group(1) if letter else "a"
        return f"ALI-{parent_num}-{letter_str}"

    return f"DEV-{identifier[:20]}"


def _parent_to_span_id(parent_identifier: str) -> str:
    """Converte parent_identifier para span_id do pai."""
    ident = parent_identifier.strip()
    if not ident:
        return ""

    ident_lower = ident.lower()
    match = re.search(r"(\d+)", ident)
    num = match.group(1) if match else "000"

    if ident_lower.startswith("art"):
        return f"ART-{num.zfill(3)}"
    elif "§" in ident or ident_lower.startswith("par"):
        return f"PAR-{num.zfill(3)}-{match.group(1) if match else '0'}"
    elif re.match(r'^[IVXLC]+$', ident):
        # Inciso romano — precisa do artigo pai, não disponível aqui
        return ""
    else:
        return f"ART-{num.zfill(3)}"
