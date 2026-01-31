"""
Writer para Milvus - chunks físicos.

PR3 v2 - Hard Reset RAG Architecture
PR3 v2.1 - Patches de robustez (device_type enum, document_version, article_number_int)

Milvus armazena chunks físicos (com @Pxx suffix) para busca vetorial.
NÃO armazena citations[] - relações vão para Neo4j.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """
    Tipos de dispositivos legais padronizados.

    Evita variações como "inciso" vs "INCISO" vs "Inciso".
    """
    ART = "ART"  # Artigo
    PAR = "PAR"  # Parágrafo
    INC = "INC"  # Inciso
    ALI = "ALI"  # Alínea
    UNKNOWN = "UNKNOWN"  # Fallback

    @classmethod
    def from_string(cls, value: str) -> "DeviceType":
        """
        Converte string para DeviceType padronizado.

        Args:
            value: String como "article", "paragraph", "inciso", "alinea", etc.

        Returns:
            DeviceType correspondente ou UNKNOWN
        """
        if not value:
            return cls.UNKNOWN

        normalized = value.lower().strip()

        mapping = {
            "art": cls.ART,
            "article": cls.ART,
            "artigo": cls.ART,
            "par": cls.PAR,
            "paragraph": cls.PAR,
            "paragrafo": cls.PAR,
            "parágrafo": cls.PAR,
            "inc": cls.INC,
            "inciso": cls.INC,
            "ali": cls.ALI,
            "alinea": cls.ALI,
            "alínea": cls.ALI,
        }

        return mapping.get(normalized, cls.UNKNOWN)


@dataclass
class MilvusChunk:
    """
    Chunk físico para inserção no Milvus.

    IMPORTANTE:
    - node_id é o PK (logical_node_id@Pxx)
    - NÃO inclui citations[] - isso vai para Neo4j
    - text é a fonte de verdade para LLM
    - retrieval_text é para embeddings/busca

    PR3 v2.1 Patches:
    - device_type padronizado (ART|PAR|INC|ALI)
    - document_version para custódia
    - article_number_int para ordenação/filtros
    - parent_chunk_id validado como artigo@P00
    """

    # IDs (obrigatórios)
    node_id: str  # PK: logical_node_id@Pxx
    logical_node_id: str  # Sem @Pxx, usado para join com Neo4j
    chunk_id: str  # document_id#span_id@Pxx
    parent_chunk_id: Optional[str]  # document_id#ART-xxx@P00 (sempre artigo) ou None

    # Informações de split
    part_index: int  # 0, 1, 2, ...
    part_total: int  # Total de partes deste span

    # Texto
    text: str  # Fonte de verdade - LLM só vê isso
    retrieval_text: str  # Para embeddings e busca
    parent_text: Optional[str]  # Contexto do parent

    # Vetores
    dense_vector: list[float] = field(default_factory=list)
    sparse_vector: dict[int, float] = field(default_factory=dict)

    # Metadados
    document_id: str = ""
    span_id: str = ""
    device_type: str = ""  # Será convertido para DeviceType (ART|PAR|INC|ALI)
    article_number: Optional[str] = None  # Texto: "5", "10-A", etc.
    article_number_int: Optional[int] = None  # Int para ordenação: 5, 10, etc.
    document_type: Optional[str] = None  # LEI, DECRETO, IN

    # Custódia e Proveniência
    document_version: Optional[str] = None  # Versão do documento (hash ou timestamp)
    ingest_run_id: Optional[str] = None
    pipeline_version: Optional[str] = None
    schema_version: str = "2.1.0"  # Atualizado para v2.1

    # Posição no documento original
    char_start: int = 0
    char_end: int = 0

    def __post_init__(self):
        """Validações e normalizações pós-inicialização."""
        # Normaliza device_type para enum padronizado
        if self.device_type:
            normalized = DeviceType.from_string(self.device_type)
            self.device_type = normalized.value

        # Extrai article_number_int do article_number se possível
        if self.article_number and self.article_number_int is None:
            match = re.match(r"(\d+)", self.article_number)
            if match:
                self.article_number_int = int(match.group(1))

        # Valida parent_chunk_id (deve ser artigo@P00 ou vazio para artigos)
        if self.parent_chunk_id:
            # Artigos não devem ter parent
            if self.device_type == DeviceType.ART.value:
                logger.warning(
                    f"Artigo {self.node_id} não deveria ter parent_chunk_id. "
                    f"Limpando valor: {self.parent_chunk_id}"
                )
                self.parent_chunk_id = None
            # Parent deve ser sempre @P00 (primeira parte do artigo)
            elif not self.parent_chunk_id.endswith("@P00"):
                logger.warning(
                    f"parent_chunk_id deve ser artigo@P00, recebido: {self.parent_chunk_id}"
                )


class MilvusWriter:
    """
    Writer para inserção de chunks no Milvus.

    Usa a collection configurada (padrão: leis_v4).
    """

    DEFAULT_COLLECTION = "leis_v4"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 19530,
        collection_name: Optional[str] = None,
    ):
        """
        Inicializa o writer.

        Args:
            host: Host do Milvus
            port: Porta do Milvus
            collection_name: Nome da collection
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        self._collection = None
        self._connected = False

    def _ensure_connected(self) -> None:
        """Conecta ao Milvus se necessário."""
        if self._connected:
            return

        try:
            from pymilvus import connections, Collection

            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
            )

            self._collection = Collection(self.collection_name)
            self._collection.load()
            self._connected = True

            logger.info(f"Conectado ao Milvus: {self.collection_name}")

        except Exception as e:
            logger.error(f"Erro ao conectar ao Milvus: {e}")
            raise

    def upsert(self, chunk: MilvusChunk) -> bool:
        """
        Insere ou atualiza um chunk no Milvus.

        Args:
            chunk: Chunk para inserir

        Returns:
            True se sucesso
        """
        self._ensure_connected()

        try:
            # Prepara dados para inserção
            data = self._chunk_to_dict(chunk)

            # Delete existente (se houver)
            self._collection.delete(expr=f'node_id == "{chunk.node_id}"')

            # Insert novo
            self._collection.insert([data])

            logger.debug(f"Chunk inserido: {chunk.node_id}")
            return True

        except Exception as e:
            logger.error(f"Erro ao inserir chunk {chunk.node_id}: {e}")
            return False

    def upsert_batch(self, chunks: list[MilvusChunk]) -> int:
        """
        Insere múltiplos chunks em batch.

        Args:
            chunks: Lista de chunks

        Returns:
            Número de chunks inseridos com sucesso
        """
        self._ensure_connected()

        if not chunks:
            return 0

        try:
            # Prepara dados
            data_list = [self._chunk_to_dict(c) for c in chunks]

            # Delete existentes
            node_ids = [c.node_id for c in chunks]
            ids_str = ", ".join([f'"{nid}"' for nid in node_ids])
            self._collection.delete(expr=f"node_id in [{ids_str}]")

            # Insert batch
            self._collection.insert(data_list)

            logger.info(f"Batch de {len(chunks)} chunks inserido")
            return len(chunks)

        except Exception as e:
            logger.error(f"Erro ao inserir batch: {e}")
            return 0

    def _chunk_to_dict(self, chunk: MilvusChunk) -> dict:
        """
        Converte MilvusChunk para dicionário para inserção.

        Args:
            chunk: Chunk para converter

        Returns:
            Dicionário com campos do Milvus
        """
        return {
            # IDs
            "node_id": chunk.node_id,
            "logical_node_id": chunk.logical_node_id,
            "chunk_id": chunk.chunk_id,
            "parent_chunk_id": chunk.parent_chunk_id or "",

            # Split info
            "part_index": chunk.part_index,
            "part_total": chunk.part_total,

            # Texto
            "text": chunk.text,
            "retrieval_text": chunk.retrieval_text,
            "parent_text": chunk.parent_text or "",

            # Vetores
            "dense_vector": chunk.dense_vector,
            "sparse_vector": chunk.sparse_vector,

            # Metadados
            "document_id": chunk.document_id,
            "span_id": chunk.span_id,
            "device_type": chunk.device_type,  # Já normalizado para ART|PAR|INC|ALI
            "article_number": chunk.article_number or "",
            "article_number_int": chunk.article_number_int if chunk.article_number_int is not None else 0,
            "document_type": chunk.document_type or "",

            # Custódia e Proveniência (PR3 v2.1)
            "document_version": chunk.document_version or "",
            "ingest_run_id": str(chunk.ingest_run_id) if chunk.ingest_run_id else "",
            "pipeline_version": chunk.pipeline_version or "",
            "schema_version": chunk.schema_version,

            # Posição
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        }

    def flush(self) -> None:
        """Força flush dos dados para disco."""
        if self._connected and self._collection:
            self._collection.flush()
            logger.debug("Flush executado")

    def close(self) -> None:
        """Fecha conexão com Milvus."""
        if self._connected:
            from pymilvus import connections
            connections.disconnect("default")
            self._connected = False
            logger.info("Conexão Milvus fechada")

    def __enter__(self):
        """Context manager entry."""
        self._ensure_connected()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
