"""
Modelos SQLAlchemy para o registro de documentos.

PR3 v2 - Hard Reset RAG Architecture
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DocumentStatus(str, Enum):
    """Status do documento no pipeline de ingestão."""

    UPLOADED = "uploaded"  # PDF salvo no MinIO
    PROCESSED = "processed"  # Docling executou, canonical.md salvo
    EMBEDDED = "embedded"  # Embeddings gerados
    INDEXED = "indexed"  # Chunks upserted no Milvus
    GRAPH_SYNCED = "graph_synced"  # Edges upserted no Neo4j
    FAILED = "failed"  # Erro em alguma etapa


class Document(Base):
    """
    Registro de documento no PostgreSQL.

    Representa o estado de um documento no pipeline de ingestão.
    """

    __tablename__ = "documents"

    # Identificação
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(String(200), nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)

    # Status do pipeline
    status = Column(
        String(50),
        nullable=False,
        default=DocumentStatus.UPLOADED.value,
        index=True,
    )

    # Hashes para integridade
    sha256_source = Column(String(64), nullable=True)
    sha256_canonical_md = Column(String(64), nullable=True)

    # Referências ao MinIO
    minio_source_key = Column(Text, nullable=True)
    minio_canonical_key = Column(Text, nullable=True)

    # Métricas
    chunk_count = Column(Integer, nullable=True)
    edge_count = Column(Integer, nullable=True)

    # Rastreamento de execução
    ingest_run_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    pipeline_version = Column(String(50), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Erro (se status == FAILED)
    error_message = Column(Text, nullable=True)

    # Constraint de unicidade
    __table_args__ = (
        UniqueConstraint("document_id", "version", name="uq_document_version"),
    )

    def __repr__(self) -> str:
        return (
            f"<Document(id={self.id}, document_id={self.document_id}, "
            f"version={self.version}, status={self.status})>"
        )

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "id": str(self.id) if self.id else None,
            "document_id": self.document_id,
            "version": self.version,
            "status": self.status,
            "sha256_source": self.sha256_source,
            "sha256_canonical_md": self.sha256_canonical_md,
            "minio_source_key": self.minio_source_key,
            "minio_canonical_key": self.minio_canonical_key,
            "chunk_count": self.chunk_count,
            "edge_count": self.edge_count,
            "ingest_run_id": str(self.ingest_run_id) if self.ingest_run_id else None,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "error_message": self.error_message,
        }
