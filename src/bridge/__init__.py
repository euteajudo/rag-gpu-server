"""
Módulo Bridge: Conecta parsing/ robusto com spans/ físicos.

PR3 v2.1 - Rebase

Este módulo faz a ponte entre:
- parsing/span_parser.py: Extração determinística (SpanType, ParsedDocument)
- spans/splitter.py: Split físico com overlap (ChunkPart)

O fluxo é:
    Markdown → SpanParser → ParsedDocument → ParsedDocumentChunkPartsBuilder → ChunkPart[]
"""

from .parsed_document_chunkparts import (
    ParsedDocumentChunkPartsBuilder,
    build_chunk_parts,
    map_span_type_to_device_type,
    find_root_article_span_id,
)

__all__ = [
    "ParsedDocumentChunkPartsBuilder",
    "build_chunk_parts",
    "map_span_type_to_device_type",
    "find_root_article_span_id",
]
