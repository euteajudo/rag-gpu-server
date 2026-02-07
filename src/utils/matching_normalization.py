# -*- coding: utf-8 -*-
"""
Matching Normalization — Normalização agressiva para matching VLM ↔ PyMuPDF.

O VLM lê texto de imagens renderizadas (OCR-like), enquanto o canonical_text
vem do PyMuPDF (nativo). Divergências por hifenização, ligaturas, Unicode,
quebras de linha causam falhas no matching de offsets.

Este módulo fornece:
- normalize_for_matching(text) → texto normalizado para comparação
- normalize_with_offset_map(text) → (texto_normalizado, norm2orig[])

NORMALIZATION_VERSION deve ser incrementada ao mudar regras (invalida cache).
"""

import re
import unicodedata

# Incrementar ao mudar qualquer regra de normalização
NORMALIZATION_VERSION = 1

# Tabela OCR: caracteres que o VLM pode ler diferente do PyMuPDF
_OCR_REPLACEMENTS = {
    "\u00AD": "",        # soft hyphen → remove
    "\u2013": "-",       # en-dash → hyphen
    "\u2014": "-",       # em-dash → hyphen
    "\u2015": "-",       # horizontal bar → hyphen
    "\u2010": "-",       # hyphen (Unicode) → ASCII hyphen
    "\u2011": "-",       # non-breaking hyphen → ASCII hyphen
    "\u2012": "-",       # figure dash → ASCII hyphen
    "\u201C": '"',       # left double quotation mark
    "\u201D": '"',       # right double quotation mark
    "\u201E": '"',       # double low-9 quotation mark
    "\u201F": '"',       # double high-reversed-9 quotation mark
    "\u2018": "'",       # left single quotation mark
    "\u2019": "'",       # right single quotation mark
    "\u201A": "'",       # single low-9 quotation mark
    "\u201B": "'",       # single high-reversed-9 quotation mark
    "\u00AB": '"',       # left-pointing double angle quotation mark «
    "\u00BB": '"',       # right-pointing double angle quotation mark »
    "\u2039": "'",       # single left-pointing angle quotation mark ‹
    "\u203A": "'",       # single right-pointing angle quotation mark ›
    "\u00BA": "o",       # masculine ordinal indicator º
    "\u00AA": "a",       # feminine ordinal indicator ª
    "\u00B0": "o",       # degree sign ° (often confused with º)
    "\u2026": "...",     # ellipsis → three dots
    "\u00A0": " ",       # non-breaking space → space
    "\u2002": " ",       # en space
    "\u2003": " ",       # em space
    "\u2009": " ",       # thin space
    "\u200A": " ",       # hair space
    "\u200B": "",        # zero-width space → remove
    "\uFEFF": "",        # BOM → remove
}

# Padrão para hífens de quebra de linha: hífen seguido de whitespace + newline
_HYPHEN_BREAK_RE = re.compile(r"-\s*\n\s*")

# Padrão para colapso de whitespace
_MULTI_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_matching(text: str) -> str:
    """
    Normaliza texto agressivamente para matching VLM ↔ PyMuPDF.

    Etapas:
    1. NFKC (ligaturas ﬁ→fi, ﬀ→ff, etc.)
    2. Tabela OCR (soft hyphen, dashes, aspas tipográficas, etc.)
    3. Remove hífens de quebra de linha: "-\\n" → ""
    4. Colapsa todo whitespace → single space
    5. Strip final

    Args:
        text: Texto a normalizar

    Returns:
        Texto normalizado para comparação
    """
    if not text:
        return ""

    # 1. NFKC — decomposes ligatures, normalizes compatibility chars
    result = unicodedata.normalize("NFKC", text)

    # 2. Tabela OCR
    for old, new in _OCR_REPLACEMENTS.items():
        result = result.replace(old, new)

    # 3. Remove hífens de quebra de linha
    result = _HYPHEN_BREAK_RE.sub("", result)

    # 4. Colapsa whitespace
    result = _MULTI_WHITESPACE_RE.sub(" ", result)

    # 5. Strip
    result = result.strip()

    return result


def normalize_with_offset_map(text: str) -> tuple[str, list[int]]:
    """
    Normaliza texto e retorna mapeamento de posições normalizado → original.

    Retorna (norm_text, norm2orig) onde norm2orig[i] é o índice no texto
    original do caractere i do texto normalizado.

    Regras de mapeamento:
    - Expansões NFKC/ligaturas (1 char orig → N chars norm):
      todos os N chars apontam para o mesmo índice original
    - Colapso whitespace (N chars orig → 1 char norm):
      o char normalizado aponta para o primeiro char do bloco
    - Remoção (soft hyphen, ZWS, hífens de quebra):
      não emite chars, pula índices
    - Substituição 1:1 (dashes, quotes):
      mapeia diretamente

    Args:
        text: Texto original a normalizar

    Returns:
        Tupla (norm_text, norm2orig) onde:
        - norm_text: texto normalizado
        - norm2orig: lista de índices no texto original
    """
    if not text:
        return ("", [])

    # Phase 1: NFKC com mapeamento char-a-char
    # Processamos cada char original, aplicando NFKC e registrando a expansão
    nfkc_chars: list[str] = []
    nfkc_map: list[int] = []  # nfkc_map[i] = índice no texto original

    for orig_idx, ch in enumerate(text):
        nfkc = unicodedata.normalize("NFKC", ch)
        for expanded_ch in nfkc:
            nfkc_chars.append(expanded_ch)
            nfkc_map.append(orig_idx)

    # Phase 2: Tabela OCR + remoção de hífens de quebra + colapso whitespace
    # Trabalhamos sobre nfkc_chars/nfkc_map
    nfkc_text = "".join(nfkc_chars)

    # Primeiro: marca posições que devem ser removidas por hífens de quebra
    # Padrão: "-\s*\n\s*" → remove tudo
    hyphen_break_positions: set[int] = set()
    for m in _HYPHEN_BREAK_RE.finditer(nfkc_text):
        for pos in range(m.start(), m.end()):
            hyphen_break_positions.add(pos)

    # Agora processamos char por char aplicando OCR replacements,
    # removendo hífens de quebra e colapsando whitespace
    norm_chars: list[str] = []
    norm2orig: list[int] = []
    in_whitespace = False

    i = 0
    while i < len(nfkc_chars):
        if i in hyphen_break_positions:
            # Skip: parte de um hífen de quebra de linha
            i += 1
            continue

        ch = nfkc_chars[i]
        orig_idx = nfkc_map[i]

        # Tabela OCR
        if ch in _OCR_REPLACEMENTS:
            replacement = _OCR_REPLACEMENTS[ch]
            if not replacement:
                # Remoção (soft hyphen, ZWS, etc.)
                i += 1
                continue
            # Substituição (pode ser multi-char, ex: ellipsis → "...")
            for rc in replacement:
                if rc == " " or rc in ("\t", "\n", "\r"):
                    if not in_whitespace:
                        norm_chars.append(" ")
                        norm2orig.append(orig_idx)
                        in_whitespace = True
                else:
                    in_whitespace = False
                    norm_chars.append(rc)
                    norm2orig.append(orig_idx)
            i += 1
            continue

        # Whitespace collapsing
        if ch in (" ", "\t", "\n", "\r", "\x0b", "\x0c"):
            if not in_whitespace:
                norm_chars.append(" ")
                norm2orig.append(orig_idx)
                in_whitespace = True
            i += 1
            continue

        # Normal character
        in_whitespace = False
        norm_chars.append(ch)
        norm2orig.append(orig_idx)
        i += 1

    # Strip leading/trailing spaces
    norm_text = "".join(norm_chars)
    stripped = norm_text.strip()

    if not stripped:
        return ("", [])

    # Compute strip offsets
    leading = len(norm_text) - len(norm_text.lstrip())
    trailing = len(norm_text) - len(norm_text.rstrip())
    end_idx = len(norm2orig) - trailing if trailing > 0 else len(norm2orig)

    final_map = norm2orig[leading:end_idx]

    return (stripped, final_map)
