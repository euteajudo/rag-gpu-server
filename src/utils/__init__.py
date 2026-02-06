"""
Utils - Funcoes utilitarias compartilhadas.
"""

from .normalization import normalize_document_id
from .canonical_utils import (
    normalize_canonical_text,
    compute_canonical_hash,
    validate_offsets_hash,
)

__all__ = [
    "normalize_document_id",
    "normalize_canonical_text",
    "compute_canonical_hash",
    "validate_offsets_hash",
]
