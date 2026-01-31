"""
Módulo de embeddings.

PR3 v2 - Hard Reset RAG Architecture
"""

from .embedding_client import (
    EmbeddingClient,
    EmbeddingResult,
    EmbeddingConfig,
)

__all__ = [
    "EmbeddingClient",
    "EmbeddingResult",
    "EmbeddingConfig",
]
