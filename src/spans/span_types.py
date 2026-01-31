"""
Tipos de dados para spans e chunks.

PR3 v2 - Hard Reset RAG Architecture

Hierarquia:
- Span: Unidade lógica extraída do markdown (artigo, parágrafo, inciso)
- ChunkPart: Parte física de um Span (pode haver N partes se texto > 8000 chars)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DeviceType(str, Enum):
    """Tipo de dispositivo legal."""

    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    INCISO = "inciso"
    ALINEA = "alinea"
    CAPUT = "caput"
    EMENTA = "ementa"
    PREAMBULO = "preambulo"
    UNKNOWN = "unknown"


@dataclass
class Span:
    """
    Unidade lógica extraída do markdown canônico.

    Representa um dispositivo legal (artigo, parágrafo, inciso, etc.)
    no nível lógico, antes de ser dividido em partes físicas.
    """

    # Identificação lógica
    logical_node_id: str  # leis:LEI-14133-2021#ART-005
    document_id: str  # LEI-14133-2021
    span_id: str  # ART-005, PAR-005-1, INC-005-I

    # Hierarquia
    parent_span_id: Optional[str]  # ART-005 (para PAR-005-1)

    # Tipo
    device_type: DeviceType  # article, paragraph, inciso, alinea

    # Conteúdo
    text: str  # Texto completo do dispositivo

    # Metadados opcionais
    article_number: Optional[str] = None  # "5" para ART-005
    document_type: Optional[str] = None  # LEI, DECRETO, IN

    # Posição no markdown original (para debug/rastreamento)
    source_start: Optional[int] = None
    source_end: Optional[int] = None

    def __post_init__(self):
        """Valida campos obrigatórios."""
        if not self.logical_node_id:
            raise ValueError("logical_node_id é obrigatório")
        if not self.document_id:
            raise ValueError("document_id é obrigatório")
        if not self.span_id:
            raise ValueError("span_id é obrigatório")


@dataclass
class ChunkPart:
    """
    Parte física de um Span.

    Quando um Span tem texto > MAX_TEXT_CHARS (8000), ele é dividido
    em múltiplas partes com overlap de 200 caracteres.
    """

    # IDs físicos
    node_id: str  # logical_node_id@P00 (PK no Milvus)
    logical_node_id: str  # leis:LEI-14133-2021#ART-005
    chunk_id: str  # LEI-14133-2021#ART-005@P00
    parent_chunk_id: Optional[str]  # LEI-14133-2021#ART-004@P00 ou None

    # Informações de split
    part_index: int  # 0, 1, 2, ...
    part_total: int  # Total de partes deste span

    # Conteúdo desta parte
    text: str  # Texto desta parte (com overlap se não for primeira/última)
    char_start: int  # Posição inicial no texto original do Span
    char_end: int  # Posição final no texto original do Span

    # Campos herdados do Span pai
    document_id: str
    span_id: str
    device_type: DeviceType
    article_number: Optional[str] = None
    document_type: Optional[str] = None

    # Campos calculados posteriormente (embeddings, etc.)
    retrieval_text: Optional[str] = None
    parent_text: Optional[str] = None
    dense_vector: Optional[list[float]] = None
    sparse_vector: Optional[dict[int, float]] = None

    # Citações detectadas nesta parte
    has_citations: bool = False
    citations_count: int = 0

    def __post_init__(self):
        """Valida campos obrigatórios."""
        if not self.node_id:
            raise ValueError("node_id é obrigatório")
        if self.part_index < 0:
            raise ValueError("part_index deve ser >= 0")
        if self.part_total < 1:
            raise ValueError("part_total deve ser >= 1")
        if self.part_index >= self.part_total:
            raise ValueError("part_index deve ser < part_total")

    @property
    def is_split(self) -> bool:
        """Retorna True se este span foi dividido em múltiplas partes."""
        return self.part_total > 1

    @property
    def is_first_part(self) -> bool:
        """Retorna True se esta é a primeira parte."""
        return self.part_index == 0

    @property
    def is_last_part(self) -> bool:
        """Retorna True se esta é a última parte."""
        return self.part_index == self.part_total - 1
