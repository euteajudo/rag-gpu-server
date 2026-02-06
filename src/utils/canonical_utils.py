# -*- coding: utf-8 -*-
"""
Canonical Utils - Funções utilitárias para texto canônico.

Funções de normalização e hash determinístico do texto canônico,
usadas por múltiplos módulos do pipeline (canonical_offsets, reconciliator,
pymupdf_extractor, etc.).

Extraído de chunking/canonical_offsets.py (PR13) para reutilização
no novo pipeline VLM (Fase 0 da migração SpanParser → Qwen3-VL).
"""

import hashlib
import unicodedata


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
