"""
VLM OCR — Extração de texto via Qwen3-VL como OCR puro.

Entrada 2: PDF → PyMuPDF (imagens) → Qwen3-VL (OCR por página) → texto bruto
           → split em blocos sintéticos → mesmo regex classifier da Entrada 1.

A única variável entre Entrada 1 e 2 é DE ONDE vem o texto:
  - Entrada 1: PyMuPDF extrai texto nativo do PDF
  - Entrada 2: Qwen3-VL faz OCR das imagens das páginas
"""

import re
import logging
import unicodedata
from typing import List, Tuple

from .vlm_models import BlockData, PageData

logger = logging.getLogger(__name__)

# ============================================================================
# Prompts OCR
# ============================================================================

OCR_SYSTEM_PROMPT = (
    "Voce e um OCR de alta precisao para documentos legais brasileiros.\n"
    "Transcreva EXATAMENTE o texto visivel na imagem, preservando:\n"
    "- Quebras de linha entre paragrafos\n"
    "- Caracteres especiais (Art., §, º, etc.)\n"
    "- Numeracao e pontuacao exatas\n"
    "- Acentuacao correta\n"
    "NAO adicione comentarios ou formatacao. Retorne APENAS o texto."
)

OCR_PAGE_PROMPT = (
    "Transcreva todo o texto visivel nesta pagina de legislacao brasileira.\n"
    "Mantenha as quebras de linha entre paragrafos."
)

# ============================================================================
# Regex para detectar início de dispositivo legal (split points)
# ============================================================================

_RE_DEVICE_START = re.compile(
    r'(?:^|\n)\s*(?='
    r'Art\.\s*\d+'
    r'|§\s*\d+'
    r'|Parágrafo\s+[uú]nico'
    r'|[IVXLCDM]+\s*[-–—]\s'
    r'|[a-z]\)\s'
    r')',
    re.MULTILINE,
)


# ============================================================================
# split_ocr_into_blocks
# ============================================================================

def split_ocr_into_blocks(
    ocr_pages: List[Tuple[int, str]],
) -> Tuple[List[dict], str, List[Tuple[int, int, int]]]:
    """
    Divide texto OCR em blocos sintéticos para o regex classifier.

    Args:
        ocr_pages: [(page_number, ocr_text), ...]

    Returns:
        (blocks, canonical_text, page_boundaries)
        - blocks: List[dict] com block_index, text, char_start, char_end,
          page_number, bbox, lines
        - canonical_text: texto concatenado normalizado (NFC + rstrip + trailing \\n)
        - page_boundaries: [(page_number, char_start, char_end), ...]

    Invariante: canonical_text[b["char_start"]:b["char_end"]] == b["text"]
    para todo bloco b.
    """
    # 1. Normaliza e concatena páginas
    page_boundaries: List[Tuple[int, int, int]] = []
    parts: List[str] = []
    offset = 0

    for page_num, raw_text in ocr_pages:
        # NFC + rstrip por linha (match PyMuPDF extractor normalization)
        lines = []
        for line in raw_text.split("\n"):
            lines.append(unicodedata.normalize("NFC", line).rstrip())
        page_text = "\n".join(lines).strip()

        if not page_text:
            continue

        # Separador entre páginas (1 newline, como PyMuPDF extractor)
        if parts:
            parts.append("\n")
            offset += 1

        page_start = offset
        parts.append(page_text)
        offset += len(page_text)
        page_boundaries.append((page_num, page_start, offset))

    canonical_text = "".join(parts)

    # Trailing newline (mesma regra de normalize_canonical_text / PyMuPDF extractor)
    canonical_text = canonical_text.rstrip("\n")
    if canonical_text:
        canonical_text += "\n"

    # 2. Encontra posições de split (marcadores de dispositivo + linhas em branco)
    split_positions = {0}

    for m in _RE_DEVICE_START.finditer(canonical_text):
        pos = m.start()
        # Avança para o início do texto (pula o \n do match)
        if pos < len(canonical_text) and canonical_text[pos] == "\n":
            pos += 1
        split_positions.add(pos)

    # Linhas em branco como separadores adicionais
    for m in re.finditer(r'\n\n+', canonical_text):
        pos = m.end()
        if pos < len(canonical_text):
            split_positions.add(pos)

    split_positions_sorted = sorted(split_positions)

    # 3. Cria blocos
    blocks: List[dict] = []
    block_idx = 0

    for i, pos in enumerate(split_positions_sorted):
        end = (
            split_positions_sorted[i + 1]
            if i + 1 < len(split_positions_sorted)
            else len(canonical_text)
        )
        text = canonical_text[pos:end].rstrip("\n")
        if not text.strip():
            continue

        # Determina page_number pelo offset
        page_num = 0
        for pn, ps, pe in page_boundaries:
            if ps <= pos < pe:
                page_num = pn
                break

        blocks.append({
            "block_index": block_idx,
            "text": text,
            "char_start": pos,
            "char_end": pos + len(text),
            "bbox": [],       # OCR não tem bbox por bloco
            "lines": [],      # OCR não tem font/span data
            "page_number": page_num,
        })
        block_idx += 1

    logger.info(
        f"OCR split: {len(ocr_pages)} páginas → {len(blocks)} blocos, "
        f"{len(canonical_text)} chars canonical"
    )

    return blocks, canonical_text, page_boundaries


# ============================================================================
# ocr_to_pages_data
# ============================================================================

def ocr_to_pages_data(
    pymupdf_pages: List[PageData],
    blocks: List[dict],
    canonical_text: str,
    page_boundaries: List[Tuple[int, int, int]],
) -> List[PageData]:
    """
    Combina imagens PyMuPDF com blocos OCR sintéticos.
    Retorna List[PageData] no mesmo formato que PyMuPDFExtractor.extract_pages().
    """
    # Agrupa blocos por page_number
    blocks_by_page: dict[int, list] = {}
    for b in blocks:
        blocks_by_page.setdefault(b["page_number"], []).append(b)

    # Mapa page_number → boundary
    boundary_map = {pn: (ps, pe) for pn, ps, pe in page_boundaries}

    pages_data: List[PageData] = []
    for pymupdf_page in pymupdf_pages:
        pn = pymupdf_page.page_number
        page_blocks_raw = blocks_by_page.get(pn, [])
        ps, pe = boundary_map.get(pn, (0, 0))

        # Converte dicts para BlockData
        block_data_list: List[BlockData] = []
        for b in page_blocks_raw:
            block_data_list.append(BlockData(
                block_index=b["block_index"],
                char_start=b["char_start"],
                char_end=b["char_end"],
                bbox_pdf=[],
                text=b["text"],
                page_number=pn,
                lines=[],
            ))

        pages_data.append(PageData(
            page_number=pn,
            image_png=pymupdf_page.image_png,
            image_base64=pymupdf_page.image_base64,
            text=canonical_text[ps:pe] if ps < pe else "",
            width=pymupdf_page.width,
            height=pymupdf_page.height,
            img_width=pymupdf_page.img_width,
            img_height=pymupdf_page.img_height,
            blocks=block_data_list,
            char_start=ps,
            char_end=pe,
        ))

    return pages_data


# ============================================================================
# Quality gate para OCR
# ============================================================================

def validate_ocr_quality(
    devices,
    canonical_text: str,
    total_pages: int,
    document_id: str,
) -> List[str]:
    """
    Valida qualidade do OCR. Retorna lista de warnings (vazia = OK).

    Args:
        devices: List[ClassifiedDevice] do regex classifier
        canonical_text: texto canônico OCR
        total_pages: número de páginas do documento
        document_id: ID do documento para logs
    """
    warnings: List[str] = []

    # QG1: Ao menos 1 artigo encontrado
    article_count = sum(1 for d in devices if d.device_type == "article")
    if article_count == 0:
        warnings.append(
            f"VLM OCR quality: nenhum artigo encontrado em {total_pages} páginas "
            f"— texto pode estar corrompido ou prompt inadequado"
        )

    # QG2: Canonical text não suspeitamente curto
    chars_per_page = len(canonical_text) / max(total_pages, 1)
    if chars_per_page < 100:
        warnings.append(
            f"VLM OCR quality: apenas {chars_per_page:.0f} chars/página "
            f"(esperado >500) — OCR pode ter falhado"
        )

    # QG3: Proporção dispositivos/páginas razoável
    devices_per_page = len(devices) / max(total_pages, 1)
    if total_pages > 2 and devices_per_page < 1:
        warnings.append(
            f"VLM OCR quality: apenas {devices_per_page:.1f} dispositivos/página "
            f"— classificação pode estar incompleta"
        )

    return warnings
