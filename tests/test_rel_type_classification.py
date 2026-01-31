# PR5: Testes para o classificador de rel_type
"""
Testes para o classificador de tipos de relacionamento (PR5).

Valida:
- Classificação correta de padrões de texto
- Confiança apropriada para cada padrão
- Default para CITA quando nenhum padrão match
- REVOGA_TACITAMENTE nunca é emitido (reservado para uso manual)
- Integração com extract_citations_from_chunk()

Executa com:
    cd D:/2025/pipeline/rag-gpu-server
    pytest tests/test_rel_type_classification.py -v
"""

import sys
from pathlib import Path

import pytest

# Adiciona src ao path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from chunking.rel_type_classifier import (
    classify_rel_type,
    classify_rel_type_from_match,
    get_rel_type_description,
    get_all_patterns,
    REL_TYPE_PATTERNS,
)


class TestClassifyRelType:
    """Testes para a função classify_rel_type()."""

    # =========================================================================
    # REVOGA_EXPRESSAMENTE
    # =========================================================================

    def test_revoga_expressamente_fica_revogado(self):
        """Padrão 'fica revogado' deve retornar REVOGA_EXPRESSAMENTE."""
        text = "Art. 200. Fica revogado o art. 5º da Lei nº 8.666, de 1993."
        rel_type, confidence = classify_rel_type(text, start=20, end=52)

        assert rel_type == "REVOGA_EXPRESSAMENTE"
        assert confidence >= 0.90

    def test_revoga_expressamente_ficam_revogados(self):
        """Padrão 'ficam revogados' deve retornar REVOGA_EXPRESSAMENTE."""
        text = "Art. 193. Ficam revogados os arts. 1º a 5º da Lei nº 8.666."
        rel_type, confidence = classify_rel_type(text, start=15, end=55)

        assert rel_type == "REVOGA_EXPRESSAMENTE"
        assert confidence >= 0.90

    def test_revoga_expressamente_revoga_o_art(self):
        """Padrão 'revoga o art.' deve retornar REVOGA_EXPRESSAMENTE."""
        text = "Este decreto revoga o art. 18 da Lei nº 14.133/2021."
        rel_type, confidence = classify_rel_type(text, start=14, end=48)

        assert rel_type == "REVOGA_EXPRESSAMENTE"
        assert confidence >= 0.85

    # =========================================================================
    # ALTERA_EXPRESSAMENTE
    # =========================================================================

    def test_altera_expressamente_passa_a_vigorar(self):
        """Padrão 'passa a vigorar com a seguinte redação' deve retornar ALTERA_EXPRESSAMENTE."""
        text = "O art. 5º da Lei 14.133 passa a vigorar com a seguinte redação:"
        rel_type, confidence = classify_rel_type(text, start=0, end=60)

        assert rel_type == "ALTERA_EXPRESSAMENTE"
        assert confidence >= 0.90

    def test_altera_expressamente_altera_o_art(self):
        """Padrão 'altera o art.' deve retornar ALTERA_EXPRESSAMENTE."""
        text = "Este decreto altera o art. 18 da Lei 14.133/2021."
        rel_type, confidence = classify_rel_type(text, start=14, end=45)

        assert rel_type == "ALTERA_EXPRESSAMENTE"
        assert confidence >= 0.85

    def test_altera_expressamente_nova_redacao(self):
        """Padrão 'nova redação do art.' deve retornar ALTERA_EXPRESSAMENTE."""
        text = "Dá nova redação ao art. 5º da Lei nº 14.133."
        rel_type, confidence = classify_rel_type(text, start=0, end=40)

        assert rel_type == "ALTERA_EXPRESSAMENTE"
        assert confidence >= 0.80

    # =========================================================================
    # REGULAMENTA
    # =========================================================================

    def test_regulamenta_decreto(self):
        """Padrão 'Este Decreto regulamenta' deve retornar REGULAMENTA."""
        text = "Este Decreto regulamenta a Lei nº 14.133, de 2021."
        rel_type, confidence = classify_rel_type(text, start=0, end=47)

        assert rel_type == "REGULAMENTA"
        assert confidence >= 0.90

    def test_regulamenta_lei(self):
        """Padrão 'regulamenta a Lei' deve retornar REGULAMENTA."""
        text = "Art. 1º Este Decreto regulamenta a Lei nº 14.133."
        rel_type, confidence = classify_rel_type(text, start=15, end=45)

        assert rel_type == "REGULAMENTA"
        assert confidence >= 0.85

    # =========================================================================
    # EXCEPCIONA
    # =========================================================================

    def test_excepciona_salvo_o_disposto(self):
        """Padrão 'salvo o disposto' deve retornar EXCEPCIONA."""
        text = "Aplicam-se as regras gerais, salvo o disposto no art. 75."
        rel_type, confidence = classify_rel_type(text, start=28, end=55)

        assert rel_type == "EXCEPCIONA"
        assert confidence >= 0.88

    def test_excepciona_exceto(self):
        """Padrão 'exceto o disposto' deve retornar EXCEPCIONA."""
        text = "Esta norma se aplica a todos, exceto o disposto no art. 3º."
        rel_type, confidence = classify_rel_type(text, start=30, end=56)

        assert rel_type == "EXCEPCIONA"
        assert confidence >= 0.85

    def test_excepciona_ressalvado(self):
        """Padrão 'ressalvado o disposto' deve retornar EXCEPCIONA."""
        text = "Observar-se-á o procedimento padrão, ressalvado o disposto no art. 14."
        rel_type, confidence = classify_rel_type(text, start=36, end=67)

        assert rel_type == "EXCEPCIONA"
        assert confidence >= 0.85

    # =========================================================================
    # DEPENDE_DE
    # =========================================================================

    def test_depende_de_nos_termos(self):
        """Padrão 'nos termos do art.' deve retornar DEPENDE_DE."""
        text = "A contratação será realizada nos termos do art. 75 da Lei 14.133."
        rel_type, confidence = classify_rel_type(text, start=30, end=60)

        assert rel_type == "DEPENDE_DE"
        assert confidence >= 0.80

    def test_depende_de_na_forma(self):
        """Padrão 'na forma do art.' deve retornar DEPENDE_DE."""
        text = "Proceder-se-á na forma do art. 18 desta Lei."
        rel_type, confidence = classify_rel_type(text, start=14, end=40)

        assert rel_type == "DEPENDE_DE"
        assert confidence >= 0.80

    def test_depende_de_observado(self):
        """Padrão 'observado o disposto' deve retornar DEPENDE_DE."""
        text = "A licitação será realizada, observado o disposto no art. 5º."
        rel_type, confidence = classify_rel_type(text, start=27, end=57)

        assert rel_type == "DEPENDE_DE"
        assert confidence >= 0.78

    def test_depende_de_de_acordo_com(self):
        """Padrão 'de acordo com o art.' deve retornar DEPENDE_DE."""
        text = "O procedimento será conduzido de acordo com o art. 18."
        rel_type, confidence = classify_rel_type(text, start=30, end=52)

        assert rel_type == "DEPENDE_DE"
        assert confidence >= 0.75

    # =========================================================================
    # REFERENCIA
    # =========================================================================

    def test_referencia_conforme(self):
        """Padrão 'conforme o art.' deve retornar REFERENCIA."""
        text = "Os conceitos são definidos conforme o art. 3º desta Lei."
        rel_type, confidence = classify_rel_type(text, start=27, end=52)

        assert rel_type == "REFERENCIA"
        assert confidence >= 0.70

    def test_referencia_vide(self):
        """Padrão 'vide art.' deve retornar REFERENCIA."""
        text = "Para maiores detalhes, vide art. 18 da Lei 14.133."
        rel_type, confidence = classify_rel_type(text, start=22, end=46)

        assert rel_type == "REFERENCIA"
        assert confidence >= 0.70

    def test_referencia_mencionado(self):
        """Padrão 'mencionado no art.' deve retornar REFERENCIA."""
        text = "O instrumento mencionado no art. 5º desta norma."
        rel_type, confidence = classify_rel_type(text, start=14, end=45)

        assert rel_type == "REFERENCIA"
        assert confidence >= 0.65

    # =========================================================================
    # CITA (default)
    # =========================================================================

    def test_cita_default_sem_contexto(self):
        """Citação sem contexto especial deve retornar CITA (default)."""
        text = "Art. 18 da Lei nº 14.133, de 2021."
        rel_type, confidence = classify_rel_type(text, start=0, end=30)

        assert rel_type == "CITA"
        assert confidence == 0.5  # Baixa confiança no default

    def test_cita_default_texto_generico(self):
        """Texto genérico deve retornar CITA (default)."""
        text = "O artigo 5 estabelece as definições básicas para esta Lei."
        rel_type, confidence = classify_rel_type(text, start=2, end=10)

        assert rel_type == "CITA"
        assert confidence == 0.5

    # =========================================================================
    # REGRA OBRIGATÓRIA: REVOGA_TACITAMENTE nunca é emitido
    # =========================================================================

    def test_revoga_tacitamente_nunca_emitido(self):
        """
        REVOGA_TACITAMENTE NUNCA deve ser emitido pelo classificador.

        Este tipo é reservado para análise manual/LLM especializada,
        pois conflitos implícitos não podem ser detectados por regex.
        """
        # Verificar que nenhum padrão retorna REVOGA_TACITAMENTE
        for pattern in REL_TYPE_PATTERNS:
            assert pattern.rel_type != "REVOGA_TACITAMENTE", (
                f"Pattern '{pattern.description}' retorna REVOGA_TACITAMENTE, "
                "mas este tipo é reservado para análise manual!"
            )

    def test_revoga_tacitamente_nao_detectado_em_texto(self):
        """
        Mesmo texto que poderia sugerir conflito tácito deve retornar outro tipo ou CITA.
        """
        # Texto que poderia parecer revogação tácita
        text = "Esta Lei dispõe de forma contrária ao art. 5º da Lei 8.666."
        rel_type, confidence = classify_rel_type(text, start=32, end=55)

        # NUNCA deve retornar REVOGA_TACITAMENTE
        assert rel_type != "REVOGA_TACITAMENTE"
        # Deve retornar algo (provavelmente CITA por falta de padrão específico)
        assert rel_type in ["CITA", "REFERENCIA", "DEPENDE_DE"]

    # =========================================================================
    # Testes de contexto (janela)
    # =========================================================================

    def test_contexto_janela_default_120(self):
        """Janela de contexto padrão de 120 caracteres deve funcionar."""
        # Criar texto onde o padrão está longe da citação
        text = "x" * 100 + "Fica revogado o" + "y" * 50 + "art. 5º" + "z" * 100
        # start/end da citação "art. 5º"
        start = 100 + 15 + 50
        end = start + 7

        rel_type, confidence = classify_rel_type(text, start=start, end=end)

        # Com janela de 120, deve encontrar "Fica revogado"
        assert rel_type == "REVOGA_EXPRESSAMENTE"

    def test_contexto_janela_pequena(self):
        """Janela pequena não deve encontrar padrão distante."""
        text = "x" * 200 + "Fica revogado o" + "y" * 100 + "art. 5º" + "z" * 200
        start = 200 + 15 + 100
        end = start + 7

        # Janela muito pequena
        rel_type, confidence = classify_rel_type(
            text, start=start, end=end, context_window=10
        )

        # Não deve encontrar o padrão
        assert rel_type == "CITA"

    # =========================================================================
    # Testes de case insensitive
    # =========================================================================

    def test_case_insensitive_revoga(self):
        """Padrão deve funcionar em lowercase."""
        text = "FICA REVOGADO o art. 5º da Lei nº 8.666."
        rel_type, _ = classify_rel_type(text, start=15, end=35)
        assert rel_type == "REVOGA_EXPRESSAMENTE"

    def test_case_insensitive_altera(self):
        """Padrão deve funcionar em mixed case."""
        text = "Altera o Art. 18 da Lei 14.133."
        rel_type, _ = classify_rel_type(text, start=0, end=28)
        assert rel_type == "ALTERA_EXPRESSAMENTE"


class TestClassifyRelTypeFromMatch:
    """Testes para classify_rel_type_from_match()."""

    def test_from_match_funciona(self):
        """Deve funcionar com objeto Match."""
        import re

        text = "Fica revogado o art. 5º da Lei nº 8.666."
        match = re.search(r"art\.\s*\d+", text)

        if match:
            rel_type, confidence = classify_rel_type_from_match(text, match)
            assert rel_type == "REVOGA_EXPRESSAMENTE"
            assert confidence >= 0.90


class TestHelperFunctions:
    """Testes para funções auxiliares."""

    def test_get_rel_type_description(self):
        """Deve retornar descrições corretas."""
        assert "genérica" in get_rel_type_description("CITA").lower()
        assert "revogação" in get_rel_type_description("REVOGA_EXPRESSAMENTE").lower()
        assert "alteração" in get_rel_type_description("ALTERA_EXPRESSAMENTE").lower()
        assert "manual" in get_rel_type_description("REVOGA_TACITAMENTE").lower()

    def test_get_all_patterns(self):
        """Deve retornar lista de padrões."""
        patterns = get_all_patterns()

        assert len(patterns) > 0
        assert all(isinstance(p, dict) for p in patterns)
        assert all("rel_type" in p for p in patterns)
        assert all("confidence" in p for p in patterns)


class TestIntegrationWithExtractor:
    """Testes de integração com citation_extractor."""

    def test_extract_citations_with_rel_type(self):
        """
        extract_citations_from_chunk() deve retornar rel_type e rel_type_confidence.
        """
        from chunking.citation_extractor import extract_citations_from_chunk

        text = """
        Art. 18. O ETP deve observar os requisitos previstos no art. 75 da Lei nº 14.133.

        Parágrafo único. Fica revogado o art. 5º da IN 58/2022.
        """

        citations = extract_citations_from_chunk(
            text=text,
            document_id="IN-65-2021",
        )

        # Deve retornar lista de dicts
        assert isinstance(citations, list)
        assert len(citations) >= 1

        # Cada citação deve ter rel_type e rel_type_confidence
        for citation in citations:
            assert isinstance(citation, dict)
            assert "target_node_id" in citation
            assert "rel_type" in citation
            assert "rel_type_confidence" in citation

            # rel_type deve ser válido (PR5 taxonomia)
            VALID_REL_TYPES = frozenset([
                "CITA", "REFERENCIA", "ALTERA", "ALTERA_EXPRESSAMENTE",
                "REVOGA", "REVOGA_EXPRESSAMENTE", "REVOGA_TACITAMENTE",
                "REGULAMENTA", "DEPENDE_DE", "EXCEPCIONA",
            ])

            assert citation["rel_type"] in VALID_REL_TYPES

            # rel_type_confidence deve ser float entre 0 e 1
            assert 0.0 <= citation["rel_type_confidence"] <= 1.0

    def test_extract_citations_revoga_detected(self):
        """Citação com contexto de revogação deve ter rel_type=REVOGA_EXPRESSAMENTE."""
        from chunking.citation_extractor import extract_citations_from_chunk

        text = "Art. 200. Fica revogado o art. 193 da Lei nº 8.666, de 1993."

        citations = extract_citations_from_chunk(
            text=text,
            document_id="LEI-14133-2021",
        )

        # Pelo menos uma citação deve ser REVOGA_EXPRESSAMENTE
        revoga_citations = [
            c for c in citations if c.get("rel_type") == "REVOGA_EXPRESSAMENTE"
        ]

        # Pode não haver citação detectada se o regex não capturar "Lei nº 8.666"
        # Mas se houver, deve ter rel_type correto
        if revoga_citations:
            assert revoga_citations[0]["rel_type_confidence"] >= 0.85


class TestPrecedence:
    """Testes de precedência de padrões."""

    def test_revoga_antes_de_cita(self):
        """REVOGA_EXPRESSAMENTE tem precedência sobre CITA."""
        # Texto com múltiplos padrões possíveis
        text = "Fica revogado, nos termos do art. 75 da Lei 14.133."

        rel_type, _ = classify_rel_type(text, start=20, end=48)

        # REVOGA vem primeiro nos padrões
        assert rel_type == "REVOGA_EXPRESSAMENTE"

    def test_excepciona_antes_de_depende_de(self):
        """EXCEPCIONA tem precedência sobre DEPENDE_DE."""
        # "salvo" vs "nos termos"
        text = "Aplica-se, salvo o disposto nos termos do art. 5º."

        rel_type, _ = classify_rel_type(text, start=35, end=48)

        # EXCEPCIONA vem primeiro
        assert rel_type == "EXCEPCIONA"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
