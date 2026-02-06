"""
Módulo de sinks (destinos de dados).

PR3 v2 - Hard Reset RAG Architecture
PR13/Etapa 4 - Artifacts Uploader
Inspection - Inspection Artifacts Uploader

Sinks disponíveis:
- MilvusWriter: Chunks físicos para busca vetorial
- Neo4jEdgeWriter: Relações lógicas no grafo
- ArtifactsUploader: Upload de evidências para VPS (ingestão)
- InspectionUploader: Upload de artefatos de inspeção para VPS
"""

from .milvus_writer import MilvusWriter, MilvusChunk
from .neo4j_writer import Neo4jEdgeWriter, EdgeCandidate, LegalNodePayload
from .artifacts_uploader import (
    ArtifactsUploader,
    ArtifactMetadata,
    ArtifactUploadResult,
    get_artifacts_uploader,
    prepare_offsets_map,
    compute_sha256,
)
from .inspection_uploader import (
    InspectionUploader,
    InspectionUploadResult,
    get_inspection_uploader,
)

__all__ = [
    # Milvus
    "MilvusWriter",
    "MilvusChunk",
    # Neo4j
    "Neo4jEdgeWriter",
    "EdgeCandidate",
    "LegalNodePayload",
    # Artifacts (PR13)
    "ArtifactsUploader",
    "ArtifactMetadata",
    "ArtifactUploadResult",
    "get_artifacts_uploader",
    "prepare_offsets_map",
    "compute_sha256",
    # Inspection
    "InspectionUploader",
    "InspectionUploadResult",
    "get_inspection_uploader",
]
