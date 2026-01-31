"""
Módulo de canonicalização e convenções de ID.

PR3 v2 - Hard Reset RAG Architecture
"""

from .id_conventions import (
    build_logical_node_id,
    build_node_id,
    build_chunk_id,
    build_parent_chunk_id,
    parse_logical_node_id,
    parse_node_id,
    get_prefix_for_document_type,
)
from .version import (
    get_pipeline_version,
    generate_ingest_run_id,
    SCHEMA_VERSION,
)
from .canonicalizer import Canonicalizer, CanonicalizationResult

__all__ = [
    # ID conventions
    "build_logical_node_id",
    "build_node_id",
    "build_chunk_id",
    "build_parent_chunk_id",
    "parse_logical_node_id",
    "parse_node_id",
    "get_prefix_for_document_type",
    # Versioning
    "get_pipeline_version",
    "generate_ingest_run_id",
    "SCHEMA_VERSION",
    # Canonicalization
    "Canonicalizer",
    "CanonicalizationResult",
]
