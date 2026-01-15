"""
Módulo de Ingestão de PDFs com GPU.

Pipeline completo de processamento:
1. Docling (PDF → Markdown) - GPU accelerated
2. Validação de Qualidade (detecta texto corrompido)
3. OCR Fallback (se qualidade baixa)
4. SpanParser (Markdown → Spans)
5. ArticleOrchestrator (LLM extraction)
6. ChunkMaterializer (parent-child chunks)
7. Enriquecimento (context, thesis, questions)
8. Embeddings (BGE-M3)

Retorna chunks prontos para indexação no Milvus.
"""

from .models import (
    IngestRequest,
    IngestResponse,
    ProcessedChunk,
    IngestStatus,
    IngestError,
)
from .pipeline import IngestionPipeline, ExtractionMethod, PipelineResult
from .quality_validator import QualityValidator, QualityReport
from .router import router as ingestion_router

__all__ = [
    # Models
    "IngestRequest",
    "IngestResponse",
    "ProcessedChunk",
    "IngestStatus",
    "IngestError",
    # Pipeline
    "IngestionPipeline",
    "PipelineResult",
    "ExtractionMethod",
    # Quality Validation
    "QualityValidator",
    "QualityReport",
    # Router
    "ingestion_router",
]
