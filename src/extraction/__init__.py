"""
Extraction - Pipeline VLM para extração de estrutura de documentos legais.

Módulos:
    pymupdf_extractor: Extração de texto determinístico + coordenadas via PyMuPDF
    vlm_client: Cliente multimodal para Qwen3-VL via vLLM
    vlm_service: Extração de estrutura hierárquica via Qwen3-VL
    vlm_models: Modelos Pydantic para o pipeline VLM
    vlm_prompts: Prompts para o Qwen3-VL
"""

from .vlm_models import DocumentExtraction, PageExtraction, DeviceExtraction, PageData, BlockData
from .vlm_client import VLMClient
from .vlm_service import VLMExtractionService
from .pymupdf_extractor import PyMuPDFExtractor
from .coord_utils import image_bbox_to_pdf_bbox, validate_bbox_pdf, compute_bbox_iou

__all__ = [
    "DocumentExtraction",
    "PageExtraction",
    "DeviceExtraction",
    "PageData",
    "BlockData",
    "VLMClient",
    "VLMExtractionService",
    "PyMuPDFExtractor",
    "image_bbox_to_pdf_bbox",
    "validate_bbox_pdf",
    "compute_bbox_iou",
]
