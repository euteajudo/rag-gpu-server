"""
Renderizador de páginas PDF com bounding boxes coloridos.

Usa PyMuPDF (fitz) para:
1. Renderizar cada página do PDF como imagem PNG
2. Desenhar bboxes coloridos sobre os blocos detectados

Cores:
  - Azul:    blocos PyMuPDF
  - Verde:   elementos VLM
  - Amarelo: matches (PyMuPDF + VLM coincidentes)
  - Vermelho: conflitos (bbox mismatch)
"""

import base64
import logging
from typing import Optional

import fitz  # PyMuPDF

from .models import BBox

logger = logging.getLogger(__name__)

# Cores RGB para cada tipo de anotação
COLOR_PYMUPDF = (0.2, 0.4, 0.9)      # Azul
COLOR_VLM = (0.2, 0.8, 0.3)          # Verde
COLOR_MATCH = (0.9, 0.8, 0.1)        # Amarelo
COLOR_CONFLICT = (0.9, 0.2, 0.2)     # Vermelho

# DPI para renderização das páginas (150 = bom equilíbrio qualidade/tamanho)
RENDER_DPI = 150


class PageRenderer:
    """Renderiza páginas PDF com anotações visuais."""

    def __init__(self, pdf_bytes: bytes):
        """
        Args:
            pdf_bytes: Conteúdo do PDF em bytes.
        """
        self._doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    @property
    def page_count(self) -> int:
        return len(self._doc)

    def get_page_size(self, page_num: int) -> tuple[float, float]:
        """Retorna (width, height) da página em pontos."""
        page = self._doc[page_num]
        rect = page.rect
        return rect.width, rect.height

    def render_page_base64(
        self,
        page_num: int,
        bboxes: Optional[list[tuple[BBox, str]]] = None,
    ) -> str:
        """
        Renderiza uma página como PNG base64 com bboxes anotados.

        Args:
            page_num: Número da página (0-indexed).
            bboxes: Lista de (BBox, tipo) onde tipo é "pymupdf", "vlm", "match", "conflict".

        Returns:
            String base64 da imagem PNG.
        """
        page = self._doc[page_num]

        # Renderiza a página como pixmap
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Desenha bboxes se fornecidos
        if bboxes:
            # Escala para converter coordenadas de pontos para pixels
            scale = RENDER_DPI / 72.0

            for bbox, bbox_type in bboxes:
                color = self._get_color(bbox_type)
                # Converte coordenadas para pixels
                x0 = int(bbox.x0 * scale)
                y0 = int(bbox.y0 * scale)
                x1 = int(bbox.x1 * scale)
                y1 = int(bbox.y1 * scale)

                # Clamp dentro dos limites da página
                x0 = max(0, min(x0, pix.width - 1))
                y0 = max(0, min(y0, pix.height - 1))
                x1 = max(0, min(x1, pix.width - 1))
                y1 = max(0, min(y1, pix.height - 1))

                if x1 > x0 and y1 > y0:
                    # Desenha retângulo com linha semi-transparente
                    self._draw_rect_on_pixmap(pix, x0, y0, x1, y1, color)

        # Converte para PNG bytes
        png_bytes = pix.tobytes(output="png")
        return base64.b64encode(png_bytes).decode("ascii")

    def render_page_clean_base64(self, page_num: int) -> str:
        """Renderiza página sem anotações (imagem limpa)."""
        return self.render_page_base64(page_num, bboxes=None)

    def extract_blocks(self, page_num: int) -> list[dict]:
        """
        Extrai blocos de texto de uma página usando PyMuPDF.

        Returns:
            Lista de dicts com: text, bbox, font_size, is_bold
        """
        page = self._doc[page_num]
        blocks = []

        # Extrai blocos de texto (type 0 = text)
        raw_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block_idx, block in enumerate(raw_blocks.get("blocks", [])):
            if block.get("type") != 0:  # Pula blocos de imagem
                continue

            # Junta texto de todas as linhas do bloco
            lines_text = []
            max_font_size = 0.0
            has_bold = False

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    lines_text.append(span.get("text", ""))
                    font_size = span.get("size", 0.0)
                    if font_size > max_font_size:
                        max_font_size = font_size
                    flags = span.get("flags", 0)
                    if flags & 2 ** 4:  # Bit 4 = bold
                        has_bold = True

            text = " ".join(lines_text).strip()
            if not text:
                continue

            bbox = block.get("bbox", (0, 0, 0, 0))
            blocks.append({
                "block_index": block_idx,
                "text": text,
                "bbox": {
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                },
                "font_size": max_font_size,
                "is_bold": has_bold,
                "page": page_num,
            })

        return blocks

    def close(self) -> None:
        """Fecha o documento PDF."""
        if self._doc:
            self._doc.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # Helpers internos
    # =========================================================================

    @staticmethod
    def _get_color(bbox_type: str) -> tuple[float, float, float]:
        colors = {
            "pymupdf": COLOR_PYMUPDF,
            "vlm": COLOR_VLM,
            "match": COLOR_MATCH,
            "conflict": COLOR_CONFLICT,
        }
        return colors.get(bbox_type, COLOR_PYMUPDF)

    @staticmethod
    def _draw_rect_on_pixmap(
        pix: fitz.Pixmap,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[float, float, float],
        thickness: int = 2,
        alpha: float = 0.15,
    ) -> None:
        """
        Desenha um retângulo colorido no pixmap.

        Desenha borda sólida e preenchimento semi-transparente.
        """
        r = int(color[0] * 255)
        g = int(color[1] * 255)
        b = int(color[2] * 255)

        stride = pix.stride
        n = pix.n  # Número de canais (3 para RGB)
        samples = pix.samples_mv  # memoryview dos pixels

        # Preenchimento semi-transparente
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                idx = y * stride + x * n
                if idx + n <= len(samples):
                    old_r = samples[idx]
                    old_g = samples[idx + 1]
                    old_b = samples[idx + 2]
                    samples[idx] = int(old_r * (1 - alpha) + r * alpha)
                    samples[idx + 1] = int(old_g * (1 - alpha) + g * alpha)
                    samples[idx + 2] = int(old_b * (1 - alpha) + b * alpha)

        # Borda sólida
        for t in range(thickness):
            # Linhas horizontais (top e bottom)
            for x in range(x0, x1 + 1):
                for y_pos in [y0 + t, y1 - t]:
                    if 0 <= y_pos < pix.height:
                        idx = y_pos * stride + x * n
                        if idx + n <= len(samples):
                            samples[idx] = r
                            samples[idx + 1] = g
                            samples[idx + 2] = b

            # Linhas verticais (left e right)
            for y in range(y0, y1 + 1):
                for x_pos in [x0 + t, x1 - t]:
                    if 0 <= x_pos < pix.width:
                        idx = y * stride + x_pos * n
                        if idx + n <= len(samples):
                            samples[idx] = r
                            samples[idx + 1] = g
                            samples[idx + 2] = b
