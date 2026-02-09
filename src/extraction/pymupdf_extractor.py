"""
PyMuPDF Extractor - Extração determinística de texto e imagens de PDFs.

Usa PyMuPDF (fitz) para:
1. Renderizar páginas como PNG (para envio ao VLM)
2. Extrair blocos de texto via get_text("dict") com bboxes em PDF space
3. Construir canonical_text a partir dos blocos em reading order
4. Normalizar cada linha DURANTE a construção (NFC + rstrip) para que
   offsets sejam nativos ao canonical_text final
5. Calcular char_start/char_end DURANTE a concatenação (offsets nativos)
6. Coletar dimensões de cada página e do pixmap

O texto extraído pelo PyMuPDF é DETERMINÍSTICO: mesmo PDF + mesma versão
PyMuPDF = mesmo texto sempre. Isso garante idempotência nos offsets canônicos.

A normalização inline (NFC + rstrip por linha + trailing \\n) garante que o
canonical_text retornado já está no formato final, tornando a chamada
normalize_canonical_text() no pipeline uma operação idempotente (no-op).

Offsets são consequência natural da concatenação, não mapeamento posterior.
"""

import base64
import logging
import unicodedata

from .vlm_models import BlockData, PageData

logger = logging.getLogger(__name__)


class PyMuPDFExtractor:
    """Extrai páginas do PDF: imagens (para VLM) + blocos de texto com offsets."""

    def __init__(self, dpi: int = 300):
        """
        Args:
            dpi: Resolução para renderização de imagens (default 300 DPI).
        """
        self.dpi = dpi

    def extract_pages(self, pdf_bytes: bytes) -> tuple[list[PageData], str]:
        """
        Extrai dados de todas as páginas do PDF.

        Para cada página:
        - Renderiza como PNG no DPI configurado (para envio ao VLM)
        - Extrai blocos de texto via get_text("dict", sort=True) com bboxes
        - Concatena blocos em reading order calculando offsets incrementais
        - Coleta dimensões (width, height) em pontos PDF e pixmap em pixels

        Args:
            pdf_bytes: Conteúdo binário do PDF

        Returns:
            Tupla (pages, canonical_text):
            - pages: Lista de PageData com blocos e offsets (1-indexed)
            - canonical_text: Texto concatenado de todos os blocos

        Raises:
            RuntimeError: Se PyMuPDF não conseguir abrir o PDF
        """
        import fitz

        pages: list[PageData] = []
        canonical_parts: list[str] = []
        current_offset = 0

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise RuntimeError(f"PyMuPDF não conseguiu abrir o PDF: {e}") from e

        try:
            total_pages = len(doc)
            logger.info(f"PyMuPDF: extraindo {total_pages} páginas (DPI={self.dpi})")

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_number = page_idx + 1  # 1-indexed

                # Renderiza como PNG
                zoom = self.dpi / 72.0  # 72 DPI é o padrão do PDF
                matrix = fitz.Matrix(zoom, zoom)
                pixmap = page.get_pixmap(matrix=matrix)
                image_png = pixmap.tobytes("png")
                image_b64 = base64.b64encode(image_png).decode("ascii")

                # Dimensões da página em pontos PDF e do pixmap em pixels
                rect = page.rect
                page_width = rect.width
                page_height = rect.height
                img_width = pixmap.width
                img_height = pixmap.height

                # Extrai blocos com bbox via dict (reading order com sort=True)
                page_dict = page.get_text("dict", sort=True)
                raw_blocks = page_dict.get("blocks", [])

                # Processa apenas blocos de texto (type=0), ignora imagens (type=1)
                page_char_start = current_offset
                page_text_parts: list[str] = []
                block_data_list: list[BlockData] = []

                for blk_idx, block in enumerate(raw_blocks):
                    if block.get("type", 0) != 0:
                        continue  # skip image blocks

                    # Extrai texto de todas as linhas/spans do bloco
                    # NFC em cada span + rstrip em cada linha para que os offsets
                    # sejam computados contra o texto já normalizado (idêntico ao
                    # resultado de normalize_canonical_text()).
                    lines_text: list[str] = []
                    block_lines: list[dict] = []
                    for line in block.get("lines", []):
                        span_texts = []
                        line_spans = []
                        for span in line.get("spans", []):
                            span_text = unicodedata.normalize("NFC", span.get("text", ""))
                            span_texts.append(span_text)
                            line_spans.append({
                                "text": span_text,
                                "font": span.get("font", ""),
                                "size": round(span.get("size", 0), 1),
                                "flags": span.get("flags", 0),
                                "bbox": [round(c, 1) for c in span.get("bbox", [0, 0, 0, 0])],
                            })
                        lines_text.append("".join(span_texts).rstrip())
                        block_lines.append({
                            "bbox": [round(c, 1) for c in line.get("bbox", [0, 0, 0, 0])],
                            "spans": line_spans,
                        })

                    block_text = "\n".join(lines_text)
                    if not block_text.strip():
                        continue

                    # Separador newline entre blocos (gap de 1 char, não incluído no range do bloco)
                    if page_text_parts:
                        page_text_parts.append("\n")
                        current_offset += 1

                    block_char_start = current_offset
                    current_offset += len(block_text)
                    block_char_end = current_offset

                    page_text_parts.append(block_text)

                    # bbox do bloco já está em PDF points (72 DPI)
                    bbox_pdf = list(block.get("bbox", [0, 0, 0, 0]))

                    block_data_list.append(BlockData(
                        block_index=blk_idx,
                        char_start=block_char_start,
                        char_end=block_char_end,
                        bbox_pdf=bbox_pdf,
                        text=block_text,
                        page_number=page_number,
                        lines=block_lines,
                    ))

                page_text = "".join(page_text_parts)
                page_char_end = current_offset
                canonical_parts.append(page_text)

                # Separador entre páginas
                if page_idx < total_pages - 1:
                    canonical_parts.append("\n")
                    current_offset += 1

                pages.append(PageData(
                    page_number=page_number,
                    image_png=image_png,
                    image_base64=image_b64,
                    text=page_text,
                    width=page_width,
                    height=page_height,
                    img_width=img_width,
                    img_height=img_height,
                    blocks=block_data_list,
                    char_start=page_char_start,
                    char_end=page_char_end,
                ))

                logger.debug(
                    f"Página {page_number}/{total_pages}: "
                    f"{len(block_data_list)} blocos, {len(page_text)} chars, "
                    f"{len(image_png)} bytes PNG, "
                    f"{page_width:.0f}x{page_height:.0f} pts, "
                    f"{img_width}x{img_height} px"
                )

        finally:
            doc.close()

        canonical_text = "".join(canonical_parts)

        # Garante exatamente um \n no final (mesma regra de normalize_canonical_text)
        canonical_text = canonical_text.rstrip("\n")
        if canonical_text:
            canonical_text += "\n"

        total_blocks = sum(len(p.blocks) for p in pages)
        logger.info(
            f"PyMuPDF: {len(pages)} páginas, {total_blocks} blocos, "
            f"{len(canonical_text)} chars de canonical_text"
        )
        return pages, canonical_text
