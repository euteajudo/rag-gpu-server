"""
Módulo de tipos e split de spans.

PR3 v2.1 - Rebase

Este módulo contém:
- Tipos: Span, ChunkPart, DeviceType
- Split físico: split_text_with_offsets, split_span_to_parts

IMPORTANTE:
    Para EXTRAÇÃO de spans, use:
    - parsing/span_parser.py: Parser determinístico robusto
    - bridge/parsed_document_chunkparts.py: Converte ParsedDocument -> ChunkPart[]

    O span_extractor.py está DEPRECATED.
"""

from .span_types import Span, ChunkPart, DeviceType
from .splitter import (
    split_text_with_offsets,
    split_span_to_parts,
    calculate_part_count,
    MAX_TEXT_CHARS,
    OVERLAP_CHARS,
)

# Deprecated - importar apenas se necessário para retrocompatibilidade
# from .span_extractor import SpanExtractor, SpanExtractionResult, extract_spans

__all__ = [
    # Tipos
    "Span",
    "ChunkPart",
    "DeviceType",
    # Split físico (mantido)
    "split_text_with_offsets",
    "split_span_to_parts",
    "calculate_part_count",
    "MAX_TEXT_CHARS",
    "OVERLAP_CHARS",
]
