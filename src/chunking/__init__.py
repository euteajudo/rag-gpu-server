"""Módulo de Chunking — utilities compartilhados pelo pipeline VLM."""

from .citation_extractor import (
    CitationExtractor,
    extract_citations_from_chunk,
    NormativeReference,
)

from .canonical_offsets import (
    normalize_canonical_text,
    compute_canonical_hash,
    validate_offsets_hash,
    extract_snippet_by_offsets,
    resolve_child_offsets,
    OffsetResolutionError,
)

__all__ = [
    "CitationExtractor",
    "extract_citations_from_chunk",
    "NormativeReference",
    "normalize_canonical_text",
    "compute_canonical_hash",
    "validate_offsets_hash",
    "extract_snippet_by_offsets",
    "resolve_child_offsets",
    "OffsetResolutionError",
]
