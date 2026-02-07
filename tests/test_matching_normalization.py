# -*- coding: utf-8 -*-
"""
Testes unitários para matching_normalization.py.

Cobre normalização agressiva para matching VLM ↔ PyMuPDF:
- Soft hyphen, ligaturas, dashes, aspas tipográficas
- Hífens de quebra de linha
- Colapso de whitespace
- Offset map round-trip
"""

import pytest
import sys
sys.path.insert(0, '/workspace/rag-gpu-server')

from src.utils.matching_normalization import (
    normalize_for_matching,
    normalize_with_offset_map,
    NORMALIZATION_VERSION,
)


class TestNormalizeForMatching:
    """Testes para normalize_for_matching()."""

    def test_empty_string(self):
        assert normalize_for_matching("") == ""

    def test_soft_hyphen_removed(self):
        """Soft hyphen U+00AD deve ser removido."""
        assert normalize_for_matching("obriga\u00ADção") == "obrigação"

    def test_ligature_fi(self):
        """Ligadura ﬁ deve virar fi."""
        assert normalize_for_matching("o\uFB01cial") == "oficial"

    def test_ligature_ff(self):
        """Ligadura ﬀ deve virar ff."""
        assert normalize_for_matching("e\uFB00ective") == "effective"

    def test_ligature_fl(self):
        """Ligadura ﬂ deve virar fl."""
        assert normalize_for_matching("\uFB02uxo") == "fluxo"

    def test_ligature_ffi(self):
        """Ligadura ﬃ deve virar ffi."""
        assert normalize_for_matching("o\uFB03cial") == "official"

    def test_en_dash(self):
        """En-dash U+2013 deve virar hífen ASCII."""
        assert normalize_for_matching("Art. 1\u2013 texto") == "Art. 1- texto"

    def test_em_dash(self):
        """Em-dash U+2014 deve virar hífen ASCII."""
        assert normalize_for_matching("Art. 1\u2014 texto") == "Art. 1- texto"

    def test_line_break_hyphen(self):
        """Hífen de quebra de linha deve ser removido, juntando palavras."""
        assert normalize_for_matching("adminis-\ntração") == "administração"

    def test_line_break_hyphen_with_spaces(self):
        """Hífen de quebra com espaços ao redor."""
        assert normalize_for_matching("adminis- \n tração") == "administração"

    def test_typographic_double_quotes(self):
        """Aspas tipográficas duplas U+201C/U+201D devem virar aspas ASCII."""
        assert normalize_for_matching("\u201Ctexto\u201D") == '"texto"'

    def test_typographic_single_quotes(self):
        """Aspas tipográficas simples U+2018/U+2019 devem virar apóstrofo."""
        assert normalize_for_matching("\u2018texto\u2019") == "'texto'"

    def test_guillemets(self):
        """Guillemets « » devem virar aspas ASCII."""
        assert normalize_for_matching("\u00ABtexto\u00BB") == '"texto"'

    def test_ordinal_indicators(self):
        """Indicadores ordinais º/ª devem virar o/a."""
        assert normalize_for_matching("Art. 5\u00BA") == "Art. 5o"
        assert normalize_for_matching("1\u00AA vez") == "1a vez"

    def test_degree_sign(self):
        """Sinal de grau ° deve virar o."""
        assert normalize_for_matching("Art. 5\u00B0") == "Art. 5o"

    def test_ellipsis(self):
        """Reticências Unicode U+2026 devem virar três pontos."""
        assert normalize_for_matching("texto\u2026") == "texto..."

    def test_whitespace_collapse(self):
        """Múltiplos espaços/tabs devem colapsar para um espaço."""
        assert normalize_for_matching("a   b\t\tc") == "a b c"

    def test_non_breaking_space(self):
        """Non-breaking space U+00A0 deve virar espaço normal."""
        assert normalize_for_matching("a\u00A0b") == "a b"

    def test_zero_width_space_removed(self):
        """Zero-width space U+200B deve ser removido."""
        assert normalize_for_matching("a\u200Bb") == "ab"

    def test_combined_normalizations(self):
        """Múltiplas normalizações combinadas."""
        text = "O o\uFB01cial adminis-\ntrou o Art. 5\u00BA\u2013\u201Ctexto\u201D"
        expected = 'O oficial administrou o Art. 5o-"texto"'
        assert normalize_for_matching(text) == expected

    def test_strip_whitespace(self):
        """Texto com espaços nas bordas deve ser trimado."""
        assert normalize_for_matching("  texto  ") == "texto"

    def test_newlines_collapsed(self):
        """Newlines devem ser colapsados para espaço."""
        assert normalize_for_matching("linha1\nlinha2\nlinha3") == "linha1 linha2 linha3"

    def test_normalization_version_exists(self):
        """NORMALIZATION_VERSION deve existir e ser inteiro positivo."""
        assert isinstance(NORMALIZATION_VERSION, int)
        assert NORMALIZATION_VERSION >= 1


class TestNormalizeWithOffsetMap:
    """Testes para normalize_with_offset_map()."""

    def test_empty_string(self):
        norm, offmap = normalize_with_offset_map("")
        assert norm == ""
        assert offmap == []

    def test_identity(self):
        """Texto ASCII simples: mapeamento 1:1."""
        norm, offmap = normalize_with_offset_map("abc")
        assert norm == "abc"
        assert offmap == [0, 1, 2]

    def test_soft_hyphen_offset(self):
        """Soft hyphen removido: índices ajustados."""
        text = "obriga\u00ADção"
        norm, offmap = normalize_with_offset_map(text)
        assert norm == "obrigação"
        # O soft hyphen (índice 6) é removido; "ção" começa do índice 7
        assert offmap[0] == 0  # 'o'
        assert offmap[5] == 5  # 'a'
        assert offmap[6] == 7  # 'ç' (pula o soft hyphen no idx 6)

    def test_ligature_fi_offset(self):
        """Ligadura ﬁ expande para 2 chars, ambos apontam para mesmo índice."""
        text = "o\uFB01cial"  # o + ﬁ + c + i + a + l
        norm, offmap = normalize_with_offset_map(text)
        assert norm == "oficial"
        # ﬁ está no índice 1 do original
        # Expandida para "fi" no normalizado (índices 1 e 2)
        assert offmap[1] == 1  # 'f' (de ﬁ)
        assert offmap[2] == 1  # 'i' (de ﬁ)

    def test_whitespace_collapse_offset(self):
        """Múltiplos espaços colapsados: mapeia para primeiro do bloco."""
        text = "a  b"
        norm, offmap = normalize_with_offset_map(text)
        assert norm == "a b"
        assert offmap[0] == 0  # 'a'
        assert offmap[1] == 1  # ' ' (primeiro espaço do bloco)
        assert offmap[2] == 3  # 'b'

    def test_offset_map_round_trip(self):
        """Normaliza texto, encontra substring, mapeia de volta para original."""
        original = "O Art. 5\u00BA estabelece que o o\uFB01cial deve..."
        norm, offmap = normalize_with_offset_map(original)

        # Encontra "oficial" no normalizado
        search = "oficial"
        pos = norm.find(search)
        assert pos >= 0

        # Mapeia de volta para o original
        orig_start = offmap[pos]
        orig_end = offmap[pos + len(search) - 1] + 1

        # O trecho original deve conter o texto (possivelmente com ligadura)
        orig_slice = original[orig_start:orig_end]
        assert "o\uFB01cial" in orig_slice or "oficial" in orig_slice

    def test_offset_map_en_dash(self):
        """En-dash substituído: mapeamento 1:1."""
        text = "Art. 1\u2013 texto"
        norm, offmap = normalize_with_offset_map(text)
        assert "-" in norm
        # O en-dash está no índice 6 do original
        dash_pos = norm.find("-")
        assert offmap[dash_pos] == 6

    def test_line_break_hyphen_offset(self):
        """Hífen de quebra removido: índices pulam os chars removidos."""
        text = "adminis-\ntração"
        norm, offmap = normalize_with_offset_map(text)
        assert norm == "administração"
        # "adminis" = indices 0-6 no original
        # "-\n" = indices 7-8 (removidos)
        # "tração" = indices 9-14
        assert offmap[0] == 0  # 'a'
        assert offmap[6] == 6  # 's'
        assert offmap[7] == 9  # 't' (pula hífen e newline)

    def test_combined_offset_round_trip(self):
        """Round-trip complexo com múltiplas normalizações."""
        original = "O o\uFB01cial adminis-\ntrou conforme Art. 5\u00BA"
        norm, offmap = normalize_with_offset_map(original)

        # Verifica que todo índice no offmap aponta para posição válida
        for i, orig_idx in enumerate(offmap):
            assert 0 <= orig_idx < len(original), (
                f"offmap[{i}]={orig_idx} fora do range [0, {len(original)})"
            )

        # Encontra "administrou" no normalizado
        search = "administrou"
        pos = norm.find(search)
        assert pos >= 0

        orig_start = offmap[pos]
        orig_end = offmap[pos + len(search) - 1] + 1
        assert orig_start >= 0
        assert orig_end <= len(original)

    def test_ellipsis_offset(self):
        """Reticências Unicode expandem para 3 chars, todos apontam mesmo índice."""
        text = "fim\u2026"
        norm, offmap = normalize_with_offset_map(text)
        assert norm == "fim..."
        # \u2026 está no índice 3; NFKC expande para "..." (3 chars)
        # Todos os 3 pontos devem apontar para índice 3
        assert offmap[3] == 3
        assert offmap[4] == 3
        assert offmap[5] == 3

    def test_only_whitespace(self):
        """Texto com apenas whitespace retorna vazio."""
        norm, offmap = normalize_with_offset_map("   \n\t  ")
        assert norm == ""
        assert offmap == []

    def test_typographic_quotes_offset(self):
        """Aspas tipográficas: mapeamento 1:1."""
        text = "\u201Ctexto\u201D"
        norm, offmap = normalize_with_offset_map(text)
        assert norm == '"texto"'
        assert offmap[0] == 0  # " (de U+201C)
        assert offmap[6] == 6  # " (de U+201D)
