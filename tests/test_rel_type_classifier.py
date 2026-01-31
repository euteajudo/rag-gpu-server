# -*- coding: utf-8 -*-
"""
Testes unitários para o classificador de tipos de relacionamento.

PR5 - Camada 1: Classificador isolado.

Este módulo testa a função classify_rel_type() que classifica citações
em tipos semânticos específicos baseado no contexto ao redor.

@author: Equipe VectorGov
@since: 30/01/2025
"""

import sys
from pathlib import Path

# Adiciona src ao path para imports (evita __init__.py problemático)
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import direto do módulo para evitar circular imports no __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rel_type_classifier",
    src_path / "chunking" / "rel_type_classifier.py"
)
rel_type_classifier_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rel_type_classifier_module)
classify_rel_type = rel_type_classifier_module.classify_rel_type


# =============================================================================
# TESTES POSITIVOS - Um para cada rel_type
# =============================================================================


class TestRevogaExpressamente:
    """Testes para REVOGA_EXPRESSAMENTE."""

    def test_fica_revogado(self):
        """'Fica revogado o art. X' deve retornar REVOGA_EXPRESSAMENTE."""
        text = "Art. 200. Fica revogado o art. 5º da Lei 14.133/2021."
        # Citação: "art. 5º da Lei 14.133/2021" posições aproximadas
        start = 27
        end = 51

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "REVOGA_EXPRESSAMENTE"
        assert confidence >= 0.90

    def test_ficam_revogados(self):
        """'Ficam revogados os arts.' deve retornar REVOGA_EXPRESSAMENTE."""
        text = "Art. 201. Ficam revogados os arts. 10 e 11 da Lei 8.666/1993."
        start = 30
        end = 58

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "REVOGA_EXPRESSAMENTE"
        assert confidence >= 0.90

    def test_revoga_o_art(self):
        """'revoga o art.' deve retornar REVOGA_EXPRESSAMENTE."""
        text = "Esta lei revoga o art. 3º e o § 2º do art. 5º da Lei anterior."
        start = 17
        end = 60

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "REVOGA_EXPRESSAMENTE"
        assert confidence >= 0.85


class TestAlteraExpressamente:
    """Testes para ALTERA_EXPRESSAMENTE."""

    def test_passa_a_vigorar_com_seguinte_redacao(self):
        """'passa a vigorar com a seguinte redação' retorna ALTERA_EXPRESSAMENTE."""
        text = "O art. 5º da Lei 14.133/2021 passa a vigorar com a seguinte redação:"
        start = 2
        end = 28

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "ALTERA_EXPRESSAMENTE"
        assert confidence >= 0.90

    def test_altera_o_art(self):
        """'altera o art.' deve retornar ALTERA_EXPRESSAMENTE."""
        text = "Esta Lei altera o art. 75 da Lei 14.133, de 2021."
        start = 17
        end = 47

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "ALTERA_EXPRESSAMENTE"
        assert confidence >= 0.85

    def test_nova_redacao_do_art(self):
        """'nova redação do art.' deve retornar ALTERA_EXPRESSAMENTE."""
        text = "Dá nova redação ao art. 18 da Lei 14.133/2021."
        start = 19
        end = 44

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "ALTERA_EXPRESSAMENTE"
        assert confidence >= 0.80


class TestRegulamenta:
    """Testes para REGULAMENTA."""

    def test_regulamenta_a_lei(self):
        """'regulamenta a Lei' deve retornar REGULAMENTA."""
        text = "Este Decreto regulamenta a Lei 14.133, de 2021."
        start = 26
        end = 45

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "REGULAMENTA"
        assert confidence >= 0.85

    def test_este_decreto_regulamenta(self):
        """'Este Decreto regulamenta...' deve retornar REGULAMENTA."""
        text = "Este Decreto regulamenta o disposto nos arts. 1º a 10 da Lei 8.666."
        start = 44
        end = 65

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "REGULAMENTA"
        assert confidence >= 0.90


class TestExcepciona:
    """Testes para EXCEPCIONA."""

    def test_salvo_o_disposto(self):
        """'salvo o disposto no art.' deve retornar EXCEPCIONA."""
        text = "Aplica-se a todos os contratos, salvo o disposto no art. 75."
        start = 52
        end = 59

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "EXCEPCIONA"
        assert confidence >= 0.90

    def test_exceto_nos_casos(self):
        """'exceto nos casos do art.' deve retornar EXCEPCIONA."""
        text = "Válido para todas as modalidades, exceto nos casos do art. 28."
        start = 55
        end = 61

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "EXCEPCIONA"
        assert confidence >= 0.85

    def test_ressalvado_o_disposto(self):
        """'ressalvado o disposto' deve retornar EXCEPCIONA."""
        text = "Sem prejuízo do contraditório, ressalvado o disposto no art. 10."
        start = 56
        end = 63

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "EXCEPCIONA"
        assert confidence >= 0.85


class TestDependeDe:
    """Testes para DEPENDE_DE."""

    def test_nos_termos_do_art(self):
        """'nos termos do art.' deve retornar DEPENDE_DE."""
        text = "O contrato será rescindido nos termos do art. 137 da Lei 14.133."
        start = 40
        end = 62

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "DEPENDE_DE"
        assert confidence >= 0.80

    def test_na_forma_da_lei(self):
        """'na forma da Lei' deve retornar DEPENDE_DE."""
        text = "A licitação será realizada na forma da Lei 14.133/2021."
        start = 39
        end = 53

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "DEPENDE_DE"
        assert confidence >= 0.80

    def test_observado_o_disposto(self):
        """'observado o disposto' deve retornar DEPENDE_DE."""
        text = "Será permitida a subcontratação, observado o disposto no art. 122."
        start = 57
        end = 65

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "DEPENDE_DE"
        assert confidence >= 0.80


class TestReferencia:
    """Testes para REFERENCIA."""

    def test_conforme_o_art(self):
        """'conforme o art.' deve retornar REFERENCIA."""
        text = "O procedimento, conforme o art. 25, deve ser público."
        start = 26
        end = 32

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "REFERENCIA"
        assert confidence >= 0.70

    def test_vide_art(self):
        """'vide art.' deve retornar REFERENCIA."""
        text = "Para maiores informações, vide art. 100 desta Lei."
        start = 31
        end = 48

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "REFERENCIA"
        assert confidence >= 0.70


class TestCitaDefault:
    """Testes para CITA (default)."""

    def test_citacao_simples(self):
        """Citação sem contexto especial deve retornar CITA."""
        text = "O art. 5º estabelece os princípios gerais."
        start = 2
        end = 9

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "CITA"
        assert confidence == 0.5

    def test_mencao_generica(self):
        """Menção genérica deve retornar CITA com baixa confiança."""
        text = "O dispositivo do art. 10 trata de licitações."
        start = 17
        end = 24

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "CITA"
        assert confidence == 0.5


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Testes de casos extremos (edge cases)."""

    def test_texto_vazio(self):
        """Texto vazio deve retornar CITA com confiança 0."""
        text = ""
        start = 0
        end = 0

        rel_type, confidence = classify_rel_type(text, start, end)

        assert rel_type == "CITA"
        assert confidence == 0.0

    def test_citacao_no_inicio_do_texto(self):
        """Citação no início (sem contexto antes) deve funcionar."""
        # Citação está logo no início, janela de contexto anterior é zero
        text = "Art. 5º fica revogado pela nova Lei."
        start = 0
        end = 7

        rel_type, confidence = classify_rel_type(text, start, end)

        # Deve detectar "fica revogado" no contexto após
        assert rel_type == "REVOGA_EXPRESSAMENTE"
        assert confidence >= 0.90


# =============================================================================
# TESTES DE PRECEDÊNCIA
# =============================================================================


class TestPrecedencia:
    """Testes de precedência entre padrões."""

    def test_revoga_tem_precedencia_sobre_altera(self):
        """REVOGA_EXPRESSAMENTE tem precedência sobre ALTERA_EXPRESSAMENTE."""
        # Texto ambíguo que poderia casar com ambos
        text = "Fica revogado e alterado o art. 5º da Lei 14.133."
        start = 27
        end = 47

        rel_type, confidence = classify_rel_type(text, start, end)

        # REVOGA vem primeiro nos padrões
        assert rel_type == "REVOGA_EXPRESSAMENTE"

    def test_excepciona_tem_precedencia_sobre_depende(self):
        """EXCEPCIONA tem precedência sobre DEPENDE_DE."""
        # "salvo" indica exceção, mesmo que também mencione termos
        text = "Aplica-se a regra, salvo o disposto nos termos do art. 10."
        start = 50
        end = 57

        rel_type, confidence = classify_rel_type(text, start, end)

        # EXCEPCIONA (salvo) vem antes de DEPENDE_DE (nos termos)
        assert rel_type == "EXCEPCIONA"


# =============================================================================
# TESTE NEGATIVO: REVOGA_TACITAMENTE
# =============================================================================


class TestRevogaTacitamenteNunca:
    """Verifica que REVOGA_TACITAMENTE nunca é retornado."""

    def test_nunca_retorna_revoga_tacitamente(self):
        """Classificador NUNCA deve retornar REVOGA_TACITAMENTE."""
        # Textos variados que poderiam sugerir revogação tácita
        textos = [
            "Este dispositivo conflita com o art. 5º da Lei anterior.",
            "A nova norma é incompatível com o art. 10.",
            "Há conflito entre esta Lei e o art. 3º do Decreto.",
            "O art. 5º perdeu eficácia com a nova Lei.",
            "Revogação tácita do art. 10 da Lei 8.666.",
        ]

        for text in textos:
            rel_type, _ = classify_rel_type(text, 0, len(text))
            assert rel_type != "REVOGA_TACITAMENTE", \
                f"REVOGA_TACITAMENTE retornado para: {text}"
