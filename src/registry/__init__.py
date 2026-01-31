"""
Módulo de registro de documentos (PostgreSQL).

PR3 v2 - Hard Reset RAG Architecture
"""

from .models import Document, DocumentStatus
from .document_registry import DocumentRegistryService

__all__ = [
    "Document",
    "DocumentStatus",
    "DocumentRegistryService",
]
