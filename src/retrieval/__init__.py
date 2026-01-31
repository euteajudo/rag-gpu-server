"""
Módulo de construção de texto para retrieval.

PR3 v2 - Hard Reset RAG Architecture
PR3 v2.1 - Rebase: Adicionado suporte para ParsedDocument
"""

from .retrieval_text_builder import (
    # Classes originais (compatibilidade)
    RetrievalTextBuilder,
    ParentTextResolver,
    build_retrieval_text,
    # PR3 v2.1 - Classes para ParsedDocument
    RetrievalTextBuilderFromParsedDocument,
    ParentTextResolverFromParsedDocument,
    RetrievalContext,
)

__all__ = [
    # PR3 v2 - Classes originais
    "RetrievalTextBuilder",
    "ParentTextResolver",
    "build_retrieval_text",
    # PR3 v2.1 - Novas classes para ParsedDocument
    "RetrievalTextBuilderFromParsedDocument",
    "ParentTextResolverFromParsedDocument",
    "RetrievalContext",
]
