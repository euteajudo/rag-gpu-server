"""
PyMuPDF Extractor - Extração determinística de texto e imagens de PDFs.

Usa PyMuPDF (fitz) para:
1. Renderizar páginas como PNG (para envio ao VLM)
2. Extrair texto nativo com get_text("text") (para canonical_text)
3. Coletar dimensões de cada página

O texto extraído pelo PyMuPDF é DETERMINÍSTICO: mesmo PDF + mesma versão
PyMuPDF = mesmo texto sempre. Isso garante idempotência nos offsets canônicos.
"""

import base64
import logging

from .vlm_models import PageData

logger = logging.getLogger(__name__)


class PyMuPDFExtractor:
    """Extrai páginas do PDF: imagens (para VLM) + texto (para canonical)."""

    def __init__(self, dpi: int = 300):
        """
        Args:
            dpi: Resolução para renderização de imagens (default 300 DPI).
        """
        self.dpi = dpi

    def extract_pages(self, pdf_bytes: bytes) -> list[PageData]:
        """
        Extrai dados de todas as páginas do PDF.

        Para cada página:
        - Renderiza como PNG no DPI configurado (para envio ao VLM)
        - Extrai texto nativo via get_text("text") (para canonical_text)
        - Coleta dimensões (width, height) em pontos PDF

        Args:
            pdf_bytes: Conteúdo binário do PDF

        Returns:
            Lista de PageData, uma por página (1-indexed)

        Raises:
            RuntimeError: Se PyMuPDF não conseguir abrir o PDF
        """
        import fitz

        pages: list[PageData] = []

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

                # Extrai texto nativo
                text = page.get_text("text")

                # Dimensões da página em pontos PDF
                rect = page.rect
                width = rect.width
                height = rect.height

                pages.append(PageData(
                    page_number=page_number,
                    image_png=image_png,
                    image_base64=image_b64,
                    text=text,
                    width=width,
                    height=height,
                ))

                logger.debug(
                    f"Página {page_number}/{total_pages}: "
                    f"{len(text)} chars, {len(image_png)} bytes PNG, "
                    f"{width:.0f}x{height:.0f} pts"
                )

        finally:
            doc.close()

        logger.info(
            f"PyMuPDF: {len(pages)} páginas extraídas, "
            f"total {sum(len(p.text) for p in pages)} chars de texto"
        )
        return pages
