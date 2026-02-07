"""Módulo de Ingestão de PDFs — Pipeline VLM (PyMuPDF + Qwen3-VL)."""

from .models import (
    IngestRequest,
    IngestResponse,
    ProcessedChunk,
    IngestStatus,
    IngestError,
)
from .pipeline import IngestionPipeline, ExtractionMethod, PipelineResult
from .router import router as ingestion_router

__all__ = [
    "IngestRequest",
    "IngestResponse",
    "ProcessedChunk",
    "IngestStatus",
    "IngestError",
    "IngestionPipeline",
    "PipelineResult",
    "ExtractionMethod",
    "ingestion_router",
]
