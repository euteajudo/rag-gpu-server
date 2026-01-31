"""
Funções para split de texto com overlap.

PR3 v2 - Hard Reset RAG Architecture
"""

from typing import List, Tuple

# Constantes de split
MAX_TEXT_CHARS = 8000  # Tamanho máximo de texto por parte
OVERLAP_CHARS = 200  # Overlap entre partes consecutivas


def split_text_with_offsets(
    text: str,
    max_chars: int = MAX_TEXT_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> List[Tuple[str, int, int]]:
    """
    Divide texto em partes com overlap, retornando offsets.

    Se o texto for menor ou igual a max_chars, retorna uma única parte.
    Caso contrário, divide em partes de max_chars com overlap entre elas.

    Args:
        text: Texto a ser dividido
        max_chars: Tamanho máximo de cada parte (default: 8000)
        overlap: Sobreposição entre partes consecutivas (default: 200)

    Returns:
        Lista de tuplas (texto_parte, char_start, char_end)
        - char_start: índice do primeiro caractere (inclusivo)
        - char_end: índice do último caractere (exclusivo)

    Examples:
        >>> parts = split_text_with_offsets("abc" * 3000, max_chars=5000, overlap=100)
        >>> len(parts)
        2
        >>> parts[0][1], parts[0][2]  # char_start, char_end da primeira parte
        (0, 5000)
    """
    if not text:
        return []

    text_len = len(text)

    # Caso simples: texto cabe inteiro
    if text_len <= max_chars:
        return [(text, 0, text_len)]

    parts: List[Tuple[str, int, int]] = []
    start = 0

    while start < text_len:
        # Calcula fim desta parte
        end = min(start + max_chars, text_len)

        # Tenta quebrar em espaço (para não cortar palavras)
        if end < text_len:
            # Procura último espaço antes do limite
            space_pos = text.rfind(" ", start, end)
            if space_pos > start + max_chars // 2:  # Só se não perder muito texto
                end = space_pos + 1  # Inclui o espaço

        # Extrai parte
        part_text = text[start:end]
        parts.append((part_text, start, end))

        # Próxima parte começa com overlap
        if end >= text_len:
            break

        # Move start, mantendo overlap
        start = end - overlap
        if start < 0:
            start = 0

    return parts


def calculate_part_count(text_len: int, max_chars: int = MAX_TEXT_CHARS) -> int:
    """
    Calcula quantas partes um texto terá após o split.

    Args:
        text_len: Tamanho do texto em caracteres
        max_chars: Tamanho máximo de cada parte

    Returns:
        Número de partes
    """
    if text_len <= max_chars:
        return 1

    # Fórmula considerando overlap
    effective_chars = max_chars - OVERLAP_CHARS
    if effective_chars <= 0:
        return text_len // max_chars + (1 if text_len % max_chars else 0)

    return (text_len - 1) // effective_chars + 1


def split_span_to_parts(
    text: str,
    logical_node_id: str,
    document_id: str,
    span_id: str,
    parent_span_id: str | None,
    device_type: str,
    article_number: str | None = None,
    document_type: str | None = None,
    max_chars: int = MAX_TEXT_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list:
    """
    Divide um span em ChunkParts.

    Conveniência que combina split_text_with_offsets com criação de ChunkParts.

    Args:
        text: Texto do span
        logical_node_id: ID lógico do span
        document_id: ID do documento
        span_id: ID do span
        parent_span_id: ID do span pai (ou None)
        device_type: Tipo do dispositivo
        article_number: Número do artigo (opcional)
        document_type: Tipo do documento (opcional)
        max_chars: Tamanho máximo por parte
        overlap: Sobreposição entre partes

    Returns:
        Lista de ChunkPart
    """
    from .span_types import ChunkPart, DeviceType
    from ..canonical import build_node_id, build_chunk_id, build_parent_chunk_id

    parts = split_text_with_offsets(text, max_chars, overlap)
    part_total = len(parts)

    chunk_parts = []
    for part_index, (part_text, char_start, char_end) in enumerate(parts):
        chunk_part = ChunkPart(
            node_id=build_node_id(logical_node_id, part_index),
            logical_node_id=logical_node_id,
            chunk_id=build_chunk_id(document_id, span_id, part_index),
            parent_chunk_id=build_parent_chunk_id(document_id, parent_span_id),
            part_index=part_index,
            part_total=part_total,
            text=part_text,
            char_start=char_start,
            char_end=char_end,
            document_id=document_id,
            span_id=span_id,
            device_type=DeviceType(device_type) if isinstance(device_type, str) else device_type,
            article_number=article_number,
            document_type=document_type,
        )
        chunk_parts.append(chunk_part)

    return chunk_parts
