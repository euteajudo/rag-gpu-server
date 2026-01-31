"""
Serviço de registro de documentos.

PR3 v2 - Hard Reset RAG Architecture
"""

import os
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, Document, DocumentStatus


class DocumentRegistryService:
    """
    Serviço para gerenciar o registro de documentos no PostgreSQL.

    Responsável por:
    - Criar registros de documentos
    - Atualizar status durante o pipeline
    - Rastrear execuções de ingestão
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        echo: bool = False,
    ):
        """
        Inicializa o serviço.

        Args:
            database_url: URL de conexão PostgreSQL
                (default: env DATABASE_URL ou postgresql://rag:rag@localhost:5432/rag_legal)
            echo: Se True, loga queries SQL
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://rag:rag@localhost:5432/rag_legal",
        )

        self._engine = create_engine(self.database_url, echo=echo)
        self._session_factory = sessionmaker(bind=self._engine)

        # Cria as tabelas se não existirem
        Base.metadata.create_all(self._engine)

    def _get_session(self) -> Session:
        """Cria uma nova sessão."""
        return self._session_factory()

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def create_or_get_document(
        self,
        document_id: str,
        version: int = 1,
        ingest_run_id: Optional[str] = None,
        pipeline_version: Optional[str] = None,
    ) -> Document:
        """
        Cria ou obtém um documento existente.

        Se o documento já existir com o mesmo (document_id, version),
        retorna o registro existente.

        Args:
            document_id: ID do documento (ex: "LEI-14133-2021")
            version: Versão do documento (default: 1)
            ingest_run_id: UUID da execução de ingestão
            pipeline_version: Versão do pipeline

        Returns:
            Documento criado ou existente
        """
        with self._get_session() as session:
            # Tenta encontrar existente
            doc = (
                session.query(Document)
                .filter(
                    Document.document_id == document_id,
                    Document.version == version,
                )
                .first()
            )

            if doc:
                # Atualiza metadados se fornecidos
                if ingest_run_id:
                    doc.ingest_run_id = uuid.UUID(ingest_run_id)
                if pipeline_version:
                    doc.pipeline_version = pipeline_version
                doc.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(doc)
                return self._detach(doc)

            # Cria novo
            doc = Document(
                document_id=document_id,
                version=version,
                status=DocumentStatus.UPLOADED.value,
                ingest_run_id=uuid.UUID(ingest_run_id) if ingest_run_id else None,
                pipeline_version=pipeline_version,
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return self._detach(doc)

    def get(
        self,
        document_id: str,
        version: int = 1,
    ) -> Optional[Document]:
        """
        Obtém um documento pelo ID e versão.

        Args:
            document_id: ID do documento
            version: Versão do documento

        Returns:
            Documento ou None se não existir
        """
        with self._get_session() as session:
            doc = (
                session.query(Document)
                .filter(
                    Document.document_id == document_id,
                    Document.version == version,
                )
                .first()
            )
            return self._detach(doc) if doc else None

    def get_by_id(self, id: str) -> Optional[Document]:
        """
        Obtém um documento pelo UUID.

        Args:
            id: UUID do documento

        Returns:
            Documento ou None se não existir
        """
        with self._get_session() as session:
            doc = session.query(Document).filter(Document.id == uuid.UUID(id)).first()
            return self._detach(doc) if doc else None

    def update_status(
        self,
        document_id: str,
        version: int,
        status: DocumentStatus,
        **kwargs,
    ) -> Optional[Document]:
        """
        Atualiza o status de um documento.

        Args:
            document_id: ID do documento
            version: Versão do documento
            status: Novo status
            **kwargs: Campos adicionais para atualizar
                - sha256_source: Hash do PDF fonte
                - sha256_canonical_md: Hash do markdown
                - minio_source_key: Key do PDF no MinIO
                - minio_canonical_key: Key do markdown no MinIO
                - chunk_count: Quantidade de chunks
                - edge_count: Quantidade de edges

        Returns:
            Documento atualizado ou None se não existir
        """
        with self._get_session() as session:
            doc = (
                session.query(Document)
                .filter(
                    Document.document_id == document_id,
                    Document.version == version,
                )
                .first()
            )

            if not doc:
                return None

            doc.status = status.value
            doc.updated_at = datetime.utcnow()

            # Atualiza campos adicionais
            for key, value in kwargs.items():
                if hasattr(doc, key):
                    setattr(doc, key, value)

            session.commit()
            session.refresh(doc)
            return self._detach(doc)

    def mark_failed(
        self,
        document_id: str,
        version: int,
        error_message: str,
    ) -> Optional[Document]:
        """
        Marca um documento como falho.

        Args:
            document_id: ID do documento
            version: Versão do documento
            error_message: Mensagem de erro

        Returns:
            Documento atualizado ou None se não existir
        """
        with self._get_session() as session:
            doc = (
                session.query(Document)
                .filter(
                    Document.document_id == document_id,
                    Document.version == version,
                )
                .first()
            )

            if not doc:
                return None

            doc.status = DocumentStatus.FAILED.value
            doc.error_message = error_message
            doc.updated_at = datetime.utcnow()

            session.commit()
            session.refresh(doc)
            return self._detach(doc)

    # =========================================================================
    # Query Operations
    # =========================================================================

    def list_by_status(
        self,
        status: DocumentStatus,
        limit: int = 100,
    ) -> list[Document]:
        """
        Lista documentos por status.

        Args:
            status: Status a filtrar
            limit: Limite de resultados

        Returns:
            Lista de documentos
        """
        with self._get_session() as session:
            docs = (
                session.query(Document)
                .filter(Document.status == status.value)
                .order_by(Document.updated_at.desc())
                .limit(limit)
                .all()
            )
            return [self._detach(doc) for doc in docs]

    def list_by_ingest_run(self, ingest_run_id: str) -> list[Document]:
        """
        Lista documentos de uma execução de ingestão.

        Args:
            ingest_run_id: UUID da execução

        Returns:
            Lista de documentos
        """
        with self._get_session() as session:
            docs = (
                session.query(Document)
                .filter(Document.ingest_run_id == uuid.UUID(ingest_run_id))
                .all()
            )
            return [self._detach(doc) for doc in docs]

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _detach(doc: Document) -> Document:
        """
        Cria uma cópia do documento desanexada da sessão.

        Isso permite usar o objeto após a sessão ser fechada.
        """
        if doc is None:
            return None

        detached = Document(
            id=doc.id,
            document_id=doc.document_id,
            version=doc.version,
            status=doc.status,
            sha256_source=doc.sha256_source,
            sha256_canonical_md=doc.sha256_canonical_md,
            minio_source_key=doc.minio_source_key,
            minio_canonical_key=doc.minio_canonical_key,
            chunk_count=doc.chunk_count,
            edge_count=doc.edge_count,
            ingest_run_id=doc.ingest_run_id,
            pipeline_version=doc.pipeline_version,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            error_message=doc.error_message,
        )
        return detached
