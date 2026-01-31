"""
Módulo de orquestração do pipeline de ingestão.

PR3 v2.1 - Rebase: Usa SpanParser robusto + bridge para ChunkParts

Pipeline:
    PDF → Canonical → SpanParser → Bridge → Chunks → Embeddings → [Milvus, Neo4j]
"""

from .ingestion_runner import IngestionRunner, IngestionConfig, IngestionResult

__all__ = [
    "IngestionRunner",
    "IngestionConfig",
    "IngestionResult",
]
