"""
AcordaoChunker - Materializa chunks de acordaos para indexacao.

Transforma spans do AcordaoSpanParser em chunks prontos para o Milvus.

Estrutura de Chunks:
===================

    acordao_id: AC-2724-2025-P
    span_id: REL-001
    chunk_id: AC-2724-2025-P#REL-001
    node_id: acordaos:AC-2724-2025-P#REL-001

Hierarquia Parent-Child:
=======================

    SUMARIO (raiz)
    REL-001, REL-002... (raiz)
    VOTO-001, VOTO-002... (raiz)
    ACORDAO (raiz)
    └── ACORDAO-9-1, ACORDAO-9-2... (filhos de ACORDAO)

Uso:
====

    from parsing.acordao_span_parser import AcordaoSpanParser
    from chunking.acordao_chunker import AcordaoChunker

    parser = AcordaoSpanParser()
    acordao = parser.parse(markdown)

    chunker = AcordaoChunker()
    chunks = chunker.materialize(acordao)

    for chunk in chunks:
        print(f"{chunk.node_id}: {chunk.text[:50]}...")

@author: VectorGov
@version: 1.0.0
@since: 22/01/2025
"""

import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ..parsing.acordao_models import ParsedAcordao, AcordaoSpan, AcordaoSpanType


@dataclass
class AcordaoChunkMetadata:
    """Metadados de um chunk de acordao."""
    schema_version: str = "1.0.0"
    extractor_version: str = "1.0.0"
    ingestion_timestamp: str = ""
    document_hash: str = ""


@dataclass
class MaterializedAcordaoChunk:
    """
    Chunk de acordao pronto para indexacao no Milvus.

    Todos os campos mapeiam diretamente para o schema acordaos_v1.
    """
    # IDs
    node_id: str  # acordaos:{acordao_id}#{span_id}
    chunk_id: str  # {acordao_id}#{span_id}
    acordao_id: str  # AC-2724-2025-P
    span_id: str  # REL-001, ACORDAO-9-1, etc.

    # Hierarquia
    device_type: str  # sumario, relatorio, voto, acordao, deliberacao
    parent_chunk_id: str  # vazio ou chunk_id do pai

    # Metadados do acordao
    numero: int
    ano: int
    colegiado: str
    processo: str
    relator: str
    data_sessao: str
    unidade_tecnica: str
    codigo_eletronico: str

    # Conteudo
    text: str
    enriched_text: str = ""
    context_header: str = ""
    thesis_text: str = ""
    thesis_type: str = ""
    synthetic_questions: str = ""

    # Citacoes
    citations: str = "[]"  # JSON

    # Aliases
    aliases: str = "[]"
    sparse_source: str = ""

    # Proveniencia
    metadata: AcordaoChunkMetadata = field(default_factory=AcordaoChunkMetadata)

    # Vetores (preenchidos depois pelo embedder)
    _dense_vector: Optional[List[float]] = None
    _thesis_vector: Optional[List[float]] = None
    _sparse_vector: Optional[Dict[int, float]] = None

    def to_milvus_row(self) -> Dict[str, Any]:
        """Converte para formato de insercao no Milvus."""
        return {
            "node_id": self.node_id,
            "chunk_id": self.chunk_id,
            "acordao_id": self.acordao_id,
            "span_id": self.span_id,
            "device_type": self.device_type,
            "parent_chunk_id": self.parent_chunk_id,
            "numero": self.numero,
            "ano": self.ano,
            "colegiado": self.colegiado,
            "processo": self.processo,
            "relator": self.relator,
            "data_sessao": self.data_sessao,
            "unidade_tecnica": self.unidade_tecnica,
            "codigo_eletronico": self.codigo_eletronico,
            "text": self.text,
            "enriched_text": self.enriched_text or self.text,
            "context_header": self.context_header,
            "thesis_text": self.thesis_text,
            "thesis_type": self.thesis_type,
            "synthetic_questions": self.synthetic_questions,
            "citations": self.citations,
            "aliases": self.aliases,
            "sparse_source": self.sparse_source or self.text,
            "schema_version": self.metadata.schema_version,
            "extractor_version": self.metadata.extractor_version,
            "ingestion_timestamp": self.metadata.ingestion_timestamp,
            "document_hash": self.metadata.document_hash,
            "dense_vector": self._dense_vector or [0.0] * 1024,
            "thesis_vector": self._thesis_vector or [0.0] * 1024,
            "sparse_vector": self._sparse_vector or {},
        }


class AcordaoChunker:
    """
    Materializa chunks de acordaos para indexacao.

    Transforma ParsedAcordao em lista de MaterializedAcordaoChunk.
    """

    def __init__(self, schema_version: str = "1.0.0", extractor_version: str = "1.0.0"):
        """Inicializa o chunker."""
        self.schema_version = schema_version
        self.extractor_version = extractor_version

    def materialize(
        self,
        acordao: ParsedAcordao,
        document_hash: Optional[str] = None
    ) -> List[MaterializedAcordaoChunk]:
        """
        Materializa todos os spans de um acordao em chunks.

        Args:
            acordao: Acordao parseado
            document_hash: Hash SHA-256 do PDF (opcional)

        Returns:
            Lista de chunks prontos para indexacao
        """
        chunks = []

        # Calcula hash se nao fornecido
        if not document_hash:
            document_hash = hashlib.sha256(acordao.source_text.encode()).hexdigest()

        # Metadados comuns
        metadata = AcordaoChunkMetadata(
            schema_version=self.schema_version,
            extractor_version=self.extractor_version,
            ingestion_timestamp=datetime.now().isoformat(),
            document_hash=document_hash,
        )

        # Processa cada span
        for span in acordao.spans:
            chunk = self._create_chunk(acordao, span, metadata)
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        acordao: ParsedAcordao,
        span: AcordaoSpan,
        metadata: AcordaoChunkMetadata
    ) -> MaterializedAcordaoChunk:
        """Cria chunk a partir de um span."""
        acordao_id = acordao.acordao_id
        chunk_id = f"{acordao_id}#{span.span_id}"
        node_id = f"acordaos:{chunk_id}"

        # Determina parent_chunk_id
        parent_chunk_id = ""
        if span.parent_id:
            parent_chunk_id = f"{acordao_id}#{span.parent_id}"

        # Mapeia span_type para device_type
        device_type = self._map_device_type(span.span_type)

        return MaterializedAcordaoChunk(
            node_id=node_id,
            chunk_id=chunk_id,
            acordao_id=acordao_id,
            span_id=span.span_id,
            device_type=device_type,
            parent_chunk_id=parent_chunk_id,
            numero=acordao.metadata.numero,
            ano=acordao.metadata.ano,
            colegiado=acordao.metadata.colegiado,
            processo=acordao.metadata.processo,
            relator=acordao.metadata.relator,
            data_sessao=acordao.metadata.data_sessao,
            unidade_tecnica=acordao.metadata.unidade_tecnica,
            codigo_eletronico=acordao.metadata.codigo_eletronico,
            text=span.text,
            metadata=metadata,
        )

    def _map_device_type(self, span_type: AcordaoSpanType) -> str:
        """Mapeia AcordaoSpanType para device_type string."""
        mapping = {
            AcordaoSpanType.HEADER: "header",
            AcordaoSpanType.SUMARIO: "sumario",
            AcordaoSpanType.RELATORIO: "relatorio",
            AcordaoSpanType.VOTO: "voto",
            AcordaoSpanType.ACORDAO: "acordao",
            AcordaoSpanType.DELIBERACAO: "deliberacao",
        }
        return mapping.get(span_type, "outro")


def materialize_acordao(
    acordao: ParsedAcordao,
    document_hash: Optional[str] = None
) -> List[MaterializedAcordaoChunk]:
    """
    Funcao de conveniencia para materializar acordao.

    Args:
        acordao: Acordao parseado
        document_hash: Hash do PDF

    Returns:
        Lista de chunks
    """
    chunker = AcordaoChunker()
    return chunker.materialize(acordao, document_hash)
