"""
Módulo de Ingestão de PDFs com GPU.

Pipeline completo de processamento:
1. Docling (PDF → Markdown) - GPU accelerated
2. SpanParser (Markdown → Spans)
3. ArticleOrchestrator (LLM extraction)
4. ChunkMaterializer (parent-child chunks)
5. Enriquecimento (context, thesis, questions)
6. Embeddings (BGE-M3)

Retorna chunks prontos para indexação no Milvus.
"""

from .models import (
    IngestRequest,
    IngestResponse,
    ProcessedChunk,
    IngestStatus,
    IngestError,
)
from .pipeline import IngestionPipeline
from .router import router as ingestion_router

__all__ = [
    "IngestRequest",
    "IngestResponse",
    "ProcessedChunk",
    "IngestStatus",
    "IngestError",
    "IngestionPipeline",
    "ingestion_router",
]
