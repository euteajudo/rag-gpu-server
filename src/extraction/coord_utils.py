"""
Coordinate Utilities — Conversão de coordenadas entre image space e PDF space.

O VLM (Qwen3-VL) retorna bboxes normalizadas [0-1] relativas à imagem renderizada.
O frontend precisa de coordenadas em PDF points (72 DPI) para highlight.

A conversão é linear porque PyMuPDF renderiza com Matrix(zoom, zoom) que escala
uniformemente: pixel = pdf_point × (dpi / 72). Logo a bbox normalizada converte:
  pdf_coord = norm_coord × page_dim_pts
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def image_bbox_to_pdf_bbox(
    bbox_norm: list[float],
    page_width_pts: float,
    page_height_pts: float,
) -> list[float]:
    """
    Converte bbox normalizada (image space 0-1) para PDF space (pontos 72 DPI).

    Args:
        bbox_norm: [x0, y0, x1, y1] normalizado 0-1 (image space)
        page_width_pts: largura da página em pontos PDF
        page_height_pts: altura da página em pontos PDF

    Returns:
        [x0, y0, x1, y1] em pontos PDF (72 DPI)
    """
    if len(bbox_norm) != 4:
        return []

    x0, y0, x1, y1 = bbox_norm
    return [
        x0 * page_width_pts,
        y0 * page_height_pts,
        x1 * page_width_pts,
        y1 * page_height_pts,
    ]


def validate_bbox_pdf(
    bbox_pdf: list[float],
    page_width_pts: float,
    page_height_pts: float,
    tolerance: float = 5.0,
) -> Optional[str]:
    """
    Valida uma bbox em PDF space.

    Returns:
        None se válida, string com motivo do erro se inválida.
    """
    if len(bbox_pdf) != 4:
        return f"bbox deve ter 4 elementos, tem {len(bbox_pdf)}"

    x0, y0, x1, y1 = bbox_pdf

    if x0 >= x1:
        return f"bbox degenerada: x0={x0:.1f} >= x1={x1:.1f}"
    if y0 >= y1:
        return f"bbox degenerada: y0={y0:.1f} >= y1={y1:.1f}"

    if x0 < -tolerance or y0 < -tolerance:
        return f"bbox fora da página (negativo): ({x0:.1f}, {y0:.1f})"
    if x1 > page_width_pts + tolerance:
        return f"bbox excede largura: x1={x1:.1f} > {page_width_pts:.1f}"
    if y1 > page_height_pts + tolerance:
        return f"bbox excede altura: y1={y1:.1f} > {page_height_pts:.1f}"

    return None


def compute_bbox_iou(a: list[float], b: list[float]) -> float:
    """
    Computa Intersection over Union (IoU) entre duas bboxes [x0, y0, x1, y1].

    Ambas devem estar no mesmo espaço de coordenadas (PDF points ou normalizado).

    Returns:
        IoU entre 0.0 e 1.0.
    """
    if len(a) != 4 or len(b) != 4:
        return 0.0

    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union
