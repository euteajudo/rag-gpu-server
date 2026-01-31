"""
Convenções de IDs para o sistema RAG.

PR3 v2 - Hard Reset RAG Architecture

Formato dos IDs:
- logical_node_id: {prefix}:{document_id}#{span_id}
  Exemplo: leis:LEI-14133-2021#ART-005

- node_id: {logical_node_id}@P{part_index:02d}
  Exemplo: leis:LEI-14133-2021#ART-005@P00

- chunk_id: {document_id}#{span_id}@P{part_index:02d}
  Exemplo: LEI-14133-2021#ART-005@P00

- parent_chunk_id: {document_id}#{parent_span_id}@P00 ou None
  Exemplo: LEI-14133-2021#ART-005@P00 (para um parágrafo do Art. 5)
"""

import re
from typing import Optional, Tuple

# Mapeamento de tipo de documento para prefixo canônico
DOCUMENT_TYPE_PREFIXES = {
    "LEI": "leis",
    "DECRETO": "leis",
    "INSTRUCAO_NORMATIVA": "leis",
    "IN": "leis",
    "PORTARIA": "leis",
    "RESOLUCAO": "leis",
    "ACORDAO": "acordaos",
    "TCU": "tcu",
    "KB_CARD": "kb",
}

# Regex para parsear IDs
LOGICAL_NODE_ID_PATTERN = re.compile(
    r"^(?P<prefix>[a-z_]+):(?P<document_id>[A-Z0-9\-\.]+)#(?P<span_id>[A-Z0-9\-_]+)$"
)
NODE_ID_PATTERN = re.compile(
    r"^(?P<logical_node_id>.+)@P(?P<part_index>\d{2})$"
)


def get_prefix_for_document_type(document_type: str) -> str:
    """
    Retorna o prefixo canônico para um tipo de documento.

    Args:
        document_type: Tipo do documento (ex: "LEI", "DECRETO", "ACORDAO")

    Returns:
        Prefixo canônico (ex: "leis", "acordaos", "tcu")
    """
    doc_type_upper = document_type.upper().replace(" ", "_")
    return DOCUMENT_TYPE_PREFIXES.get(doc_type_upper, "leis")


def build_logical_node_id(
    prefix: str,
    document_id: str,
    span_id: str,
) -> str:
    """
    Constrói o logical_node_id (ID lógico do node no Neo4j).

    Args:
        prefix: Prefixo do namespace (ex: "leis", "acordaos")
        document_id: ID do documento (ex: "LEI-14133-2021")
        span_id: ID do span (ex: "ART-005", "PAR-005-1")

    Returns:
        logical_node_id no formato {prefix}:{document_id}#{span_id}
        Exemplo: "leis:LEI-14133-2021#ART-005"
    """
    return f"{prefix}:{document_id}#{span_id}"


def build_node_id(
    logical_node_id: str,
    part_index: int,
) -> str:
    """
    Constrói o node_id (PK física no Milvus).

    Args:
        logical_node_id: ID lógico (ex: "leis:LEI-14133-2021#ART-005")
        part_index: Índice da parte (0, 1, 2, ...)

    Returns:
        node_id no formato {logical_node_id}@P{part_index:02d}
        Exemplo: "leis:LEI-14133-2021#ART-005@P00"
    """
    return f"{logical_node_id}@P{part_index:02d}"


def build_chunk_id(
    document_id: str,
    span_id: str,
    part_index: int,
) -> str:
    """
    Constrói o chunk_id (ID legível do chunk).

    Args:
        document_id: ID do documento (ex: "LEI-14133-2021")
        span_id: ID do span (ex: "ART-005")
        part_index: Índice da parte (0, 1, 2, ...)

    Returns:
        chunk_id no formato {document_id}#{span_id}@P{part_index:02d}
        Exemplo: "LEI-14133-2021#ART-005@P00"
    """
    return f"{document_id}#{span_id}@P{part_index:02d}"


def build_parent_chunk_id(
    document_id: str,
    parent_span_id: Optional[str],
) -> Optional[str]:
    """
    Constrói o parent_chunk_id para chunks filhos.

    Args:
        document_id: ID do documento
        parent_span_id: ID do span pai (None se for root)

    Returns:
        parent_chunk_id no formato {document_id}#{parent_span_id}@P00
        ou None se não tiver pai

    Note:
        Parent sempre referencia a parte 0 (@P00) do span pai,
        pois a hierarquia é entre spans lógicos, não partes físicas.
    """
    if parent_span_id is None:
        return None
    return f"{document_id}#{parent_span_id}@P00"


def parse_logical_node_id(logical_node_id: str) -> Optional[Tuple[str, str, str]]:
    """
    Faz parse de um logical_node_id.

    Args:
        logical_node_id: ID lógico (ex: "leis:LEI-14133-2021#ART-005")

    Returns:
        Tupla (prefix, document_id, span_id) ou None se inválido
    """
    match = LOGICAL_NODE_ID_PATTERN.match(logical_node_id)
    if not match:
        return None
    return (
        match.group("prefix"),
        match.group("document_id"),
        match.group("span_id"),
    )


def parse_node_id(node_id: str) -> Optional[Tuple[str, int]]:
    """
    Faz parse de um node_id físico.

    Args:
        node_id: ID físico (ex: "leis:LEI-14133-2021#ART-005@P00")

    Returns:
        Tupla (logical_node_id, part_index) ou None se inválido
    """
    match = NODE_ID_PATTERN.match(node_id)
    if not match:
        return None
    return (
        match.group("logical_node_id"),
        int(match.group("part_index")),
    )


def extract_logical_from_node_id(node_id: str) -> Optional[str]:
    """
    Extrai o logical_node_id de um node_id físico.

    Args:
        node_id: ID físico (ex: "leis:LEI-14133-2021#ART-005@P00")

    Returns:
        logical_node_id (ex: "leis:LEI-14133-2021#ART-005") ou None se inválido
    """
    parsed = parse_node_id(node_id)
    if parsed is None:
        return None
    return parsed[0]


def is_valid_logical_node_id(value: str) -> bool:
    """Verifica se é um logical_node_id válido."""
    return LOGICAL_NODE_ID_PATTERN.match(value) is not None


def is_valid_node_id(value: str) -> bool:
    """Verifica se é um node_id físico válido."""
    return NODE_ID_PATTERN.match(value) is not None
