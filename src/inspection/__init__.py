"""
Módulo de inspeção do pipeline (Pipeline Inspector).

Dry-run com aprovação: processa o PDF, mostra resultados intermediários
para revisão humana, e ao aprovar, persiste artefatos no MinIO.
"""

from .models import (
    InspectionStage,
    InspectionStatus,
    InspectRequest,
    PyMuPDFBlock,
    PyMuPDFArtifact,
    VLMElement,
    VLMArtifact,
    ReconciliationMatch,
    ReconciliationArtifact,
    IntegrityCheck,
    IntegrityArtifact,
    ChunkPreview,
    ChunksPreviewArtifact,
    InspectionMetadata,
    ApprovalResult,
)
from .approval import ApprovalService
from .storage import InspectionStorage

__all__ = [
    "InspectionStage",
    "InspectionStatus",
    "InspectRequest",
    "PyMuPDFBlock",
    "PyMuPDFArtifact",
    "VLMElement",
    "VLMArtifact",
    "ReconciliationMatch",
    "ReconciliationArtifact",
    "IntegrityCheck",
    "IntegrityArtifact",
    "ChunkPreview",
    "ChunksPreviewArtifact",
    "InspectionMetadata",
    "ApprovalResult",
    "ApprovalService",
    "InspectionStorage",
]
