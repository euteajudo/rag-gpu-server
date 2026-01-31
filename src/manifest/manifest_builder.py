"""
Construtor de manifest para rastreabilidade.

PR3 v2 - Hard Reset RAG Architecture

O manifest.json é armazenado no MinIO e contém
metadados completos sobre a ingestão de um documento.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class IngestManifest:
    """
    Manifest de ingestão de um documento.

    Armazena todos os metadados para rastreabilidade completa.
    """

    # Identificação
    document_id: str
    ingest_run_id: str
    pipeline_version: str
    schema_version: str = "2.0.0"

    # Timestamps
    started_at: str = ""  # ISO format
    completed_at: str = ""  # ISO format
    duration_seconds: float = 0.0

    # Hashes para integridade
    sha256_source: str = ""  # Hash do PDF original
    sha256_canonical: str = ""  # Hash do markdown canônico

    # Métricas de extração
    source_bytes: int = 0
    canonical_chars: int = 0
    page_count: int = 0

    # Métricas de spans
    span_count: int = 0
    article_count: int = 0
    paragraph_count: int = 0
    inciso_count: int = 0
    alinea_count: int = 0

    # Métricas de chunks
    chunk_count: int = 0  # Chunks físicos no Milvus
    split_count: int = 0  # Quantos spans foram divididos

    # Métricas de Neo4j
    node_count: int = 0  # Nós lógicos
    edge_count: int = 0  # Relações CITA

    # Status
    status: str = "pending"  # pending, success, failed
    error_message: Optional[str] = None

    # Referências no MinIO
    minio_source_key: str = ""
    minio_canonical_key: str = ""
    minio_manifest_key: str = ""

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Converte para JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "IngestManifest":
        """Cria instância a partir de dicionário."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "IngestManifest":
        """Cria instância a partir de JSON."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ManifestBuilder:
    """
    Construtor de manifest para um documento.

    Acumula métricas durante o pipeline e gera o manifest final.
    """

    def __init__(
        self,
        document_id: str,
        ingest_run_id: str,
        pipeline_version: str,
        schema_version: str = "2.0.0",
    ):
        """
        Inicializa o builder.

        Args:
            document_id: ID do documento
            ingest_run_id: UUID da execução
            pipeline_version: Versão do pipeline (git SHA)
            schema_version: Versão do schema
        """
        self.manifest = IngestManifest(
            document_id=document_id,
            ingest_run_id=ingest_run_id,
            pipeline_version=pipeline_version,
            schema_version=schema_version,
        )
        self._start_time: Optional[datetime] = None

    def start(self) -> None:
        """Marca início da ingestão."""
        self._start_time = datetime.utcnow()
        self.manifest.started_at = self._start_time.isoformat() + "Z"
        self.manifest.status = "running"
        logger.debug(f"Ingestão iniciada: {self.manifest.document_id}")

    def set_source_info(
        self,
        sha256: str,
        size_bytes: int,
        minio_key: str,
    ) -> None:
        """
        Define informações do arquivo fonte (PDF).

        Args:
            sha256: Hash do arquivo
            size_bytes: Tamanho em bytes
            minio_key: Chave no MinIO
        """
        self.manifest.sha256_source = sha256
        self.manifest.source_bytes = size_bytes
        self.manifest.minio_source_key = minio_key

    def set_canonical_info(
        self,
        sha256: str,
        char_count: int,
        page_count: int,
        minio_key: str,
    ) -> None:
        """
        Define informações do markdown canônico.

        Args:
            sha256: Hash do markdown
            char_count: Número de caracteres
            page_count: Número de páginas
            minio_key: Chave no MinIO
        """
        self.manifest.sha256_canonical = sha256
        self.manifest.canonical_chars = char_count
        self.manifest.page_count = page_count
        self.manifest.minio_canonical_key = minio_key

    def set_span_metrics(
        self,
        span_count: int,
        article_count: int,
        paragraph_count: int = 0,
        inciso_count: int = 0,
        alinea_count: int = 0,
    ) -> None:
        """
        Define métricas de extração de spans.

        Args:
            span_count: Total de spans
            article_count: Número de artigos
            paragraph_count: Número de parágrafos
            inciso_count: Número de incisos
            alinea_count: Número de alíneas
        """
        self.manifest.span_count = span_count
        self.manifest.article_count = article_count
        self.manifest.paragraph_count = paragraph_count
        self.manifest.inciso_count = inciso_count
        self.manifest.alinea_count = alinea_count

    def set_chunk_metrics(
        self,
        chunk_count: int,
        split_count: int = 0,
    ) -> None:
        """
        Define métricas de chunks físicos.

        Args:
            chunk_count: Total de chunks no Milvus
            split_count: Quantos spans foram divididos
        """
        self.manifest.chunk_count = chunk_count
        self.manifest.split_count = split_count

    def set_graph_metrics(
        self,
        node_count: int,
        edge_count: int,
    ) -> None:
        """
        Define métricas do grafo Neo4j.

        Args:
            node_count: Número de nós lógicos
            edge_count: Número de relações CITA
        """
        self.manifest.node_count = node_count
        self.manifest.edge_count = edge_count

    def set_manifest_key(self, minio_key: str) -> None:
        """Define a chave do manifest no MinIO."""
        self.manifest.minio_manifest_key = minio_key

    def complete(self) -> IngestManifest:
        """
        Finaliza a ingestão com sucesso.

        Returns:
            Manifest completo
        """
        now = datetime.utcnow()
        self.manifest.completed_at = now.isoformat() + "Z"
        self.manifest.status = "success"

        if self._start_time:
            self.manifest.duration_seconds = (now - self._start_time).total_seconds()

        logger.info(
            f"Ingestão concluída: {self.manifest.document_id} "
            f"({self.manifest.chunk_count} chunks, {self.manifest.edge_count} edges)"
        )

        return self.manifest

    def fail(self, error_message: str) -> IngestManifest:
        """
        Marca a ingestão como falha.

        Args:
            error_message: Mensagem de erro

        Returns:
            Manifest com status de falha
        """
        now = datetime.utcnow()
        self.manifest.completed_at = now.isoformat() + "Z"
        self.manifest.status = "failed"
        self.manifest.error_message = error_message

        if self._start_time:
            self.manifest.duration_seconds = (now - self._start_time).total_seconds()

        logger.error(
            f"Ingestão falhou: {self.manifest.document_id} - {error_message}"
        )

        return self.manifest

    def build(self) -> IngestManifest:
        """
        Retorna o manifest atual (sem marcar como completo).

        Returns:
            Manifest no estado atual
        """
        return self.manifest
