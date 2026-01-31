# -*- coding: utf-8 -*-
"""
Canonical Offsets - Extração de offsets do ParsedDocument (PR13).

Este módulo extrai os offsets (start_pos, end_pos) de cada span
do ParsedDocument, permitindo slicing puro sem find().

Princípio PR13:
==============
    Quando canonical_hash == hash_atual E start/end >= 0:
        → usa slicing puro: canonical_text[start:end]
    Caso contrário:
        → fallback best-effort via find()

Uso:
====
    from chunking.canonical_offsets import extract_offsets_from_parsed_doc

    # Após SpanParser
    parsed_doc = span_parser.parse(markdown)

    # Extrai offsets
    offsets_map, canonical_hash = extract_offsets_from_parsed_doc(parsed_doc)

    # Passa para ChunkMaterializer
    materializer = ChunkMaterializer(
        document_id=...,
        offsets_map=offsets_map,
        canonical_hash=canonical_hash,
    )
"""

import hashlib
import logging
import unicodedata
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def normalize_canonical_text(text: str) -> str:
    """
    Normaliza texto canônico para garantir determinismo.

    Regras aplicadas (em ordem):
    1. Unicode NFC normalization
    2. Normaliza line endings para LF (\\n)
    3. Remove trailing whitespace de cada linha
    4. Garante exatamente um \\n no final

    IMPORTANTE: Esta função DEVE ser usada sempre que o canonical
    é construído ou comparado, para garantir byte-a-byte igualdade.

    Args:
        text: Texto a normalizar

    Returns:
        Texto normalizado (determinístico)
    """
    if not text:
        return ""

    # 1. Unicode NFC (Canonical Decomposition, followed by Canonical Composition)
    text = unicodedata.normalize("NFC", text)

    # 2. Normaliza line endings: CRLF -> LF, CR -> LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Remove trailing whitespace de cada linha
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 4. Garante exatamente um \n no final (se houver conteúdo)
    text = text.rstrip("\n")
    if text:
        text += "\n"

    return text


def compute_canonical_hash(canonical_text: str) -> str:
    """
    Computa hash SHA256 do texto canônico.

    O hash é usado para detectar mismatch entre offsets armazenados
    e o canonical atual (anti-mismatch).

    Args:
        canonical_text: Texto canônico (já normalizado)

    Returns:
        Hash SHA256 como string hexadecimal (64 chars)
    """
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def extract_offsets_from_parsed_doc(
    parsed_doc,  # ParsedDocument
) -> Tuple[Dict[str, Tuple[int, int]], str]:
    """
    Extrai offsets de todos os spans de um ParsedDocument.

    Cada span do ParsedDocument tem start_pos e end_pos que indicam
    a posição no texto fonte. Esta função coleta esses offsets
    e computa o hash do source_text para validação anti-mismatch.

    Args:
        parsed_doc: ParsedDocument com spans parseados

    Returns:
        Tupla (offsets_map, canonical_hash) onde:
        - offsets_map: dict[span_id, (start_pos, end_pos)]
        - canonical_hash: SHA256 do source_text normalizado
    """
    offsets_map: Dict[str, Tuple[int, int]] = {}

    # Extrai offsets de cada span
    for span in parsed_doc.spans:
        # Verifica se o span tem offsets válidos
        # NOTA: Não usar "or -1" pois start_pos=0 é válido (início do documento)
        start = getattr(span, 'start_pos', -1)
        if start is None:
            start = -1
        end = getattr(span, 'end_pos', -1)
        if end is None:
            end = -1

        if start >= 0 and end > start:
            offsets_map[span.span_id] = (start, end)
        else:
            # Log span sem offsets válidos (não crítico)
            logger.debug(f"Span {span.span_id} sem offsets válidos: start={start}, end={end}")

    # Computa hash do source_text normalizado
    source_text = getattr(parsed_doc, 'source_text', '') or ''
    normalized_text = normalize_canonical_text(source_text)
    canonical_hash = compute_canonical_hash(normalized_text) if normalized_text else ""

    logger.info(
        f"Extraídos offsets de {len(offsets_map)} spans. "
        f"Hash: {canonical_hash[:16]}..."
    )

    return offsets_map, canonical_hash


def validate_offsets_hash(
    stored_hash: str,
    current_canonical_text: str,
) -> bool:
    """
    Valida se o hash armazenado confere com o texto canônico atual.

    Se o hash não confere, significa que o canonical_text mudou
    e os offsets armazenados são inválidos (devem usar fallback find()).

    Args:
        stored_hash: Hash armazenado no chunk
        current_canonical_text: Texto canônico atual (será normalizado)

    Returns:
        True se hash confere (offsets válidos), False caso contrário
    """
    if not stored_hash:
        return False

    normalized = normalize_canonical_text(current_canonical_text)
    current_hash = compute_canonical_hash(normalized)

    return stored_hash == current_hash


def extract_snippet_by_offsets(
    canonical_text: str,
    start: int,
    end: int,
    stored_hash: str,
) -> Tuple[str, bool]:
    """
    Extrai snippet usando offsets (zero fallback find).

    Esta é a função principal do PR13: usa slicing puro quando
    os offsets são válidos e o hash confere.

    Args:
        canonical_text: Texto canônico completo
        start: Offset início
        end: Offset fim
        stored_hash: Hash armazenado para validação

    Returns:
        Tupla (snippet, used_offsets) onde:
        - snippet: Texto extraído
        - used_offsets: True se usou slicing puro, False se fallback
    """
    # Verifica se pode usar slicing puro
    if start >= 0 and end > start and stored_hash:
        if validate_offsets_hash(stored_hash, canonical_text):
            # PR13: slicing puro (zero find)
            snippet = canonical_text[start:end]
            return snippet, True
        else:
            logger.warning(
                f"Hash mismatch: offsets inválidos. "
                f"stored_hash={stored_hash[:16]}..."
            )

    # Fallback: não pode usar offsets
    return "", False
