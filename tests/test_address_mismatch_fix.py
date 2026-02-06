"""
Testes para validar a correção do bug ADDRESS_MISMATCH.

O bug ocorria quando citações internas como "conforme § 1º deste artigo"
eram incorretamente detectadas como novos parágrafos pelo SpanParser.

Este teste verifica que:
1. Citações internas são ignoradas
2. Parágrafos reais são corretamente detectados
3. Os span_ids correspondem ao conteúdo real
"""

import pytest
from src.parsing import SpanParser, AddressValidator


class TestCitationDetection:
    """Testes para detecção de citações vs novos dispositivos."""

    def test_citation_context_detection(self):
        """Verifica que citações internas são detectadas corretamente."""
        parser = SpanParser()

        # Citações que devem ser detectadas
        # Formato: (texto, posição_do_§, posição_fim)
        citations = [
            "conforme § 1º deste artigo",
            "nos termos do § 2º",
            "previsto no § 3º acima",
            "de que trata o § 1º do art. 40",
            "segundo o § único",
            "o § 1º deste artigo estabelece",
        ]

        for text in citations:
            # Encontra posição real do § no texto
            start = text.find("§")
            end = start + 5  # § + número
            result = parser._is_citation_context(text, start, end)
            assert result is True, f"Deveria detectar citação: '{text}'"

    def test_real_paragraph_not_detected_as_citation(self):
        """Verifica que parágrafos reais NÃO são detectados como citação."""
        parser = SpanParser()

        # Parágrafos reais (início de linha, sem contexto de citação)
        real_paragraphs = [
            "§ 1º O estudo técnico preliminar",
            "§ 2º Para os fins desta lei",
            "\n§ 3º As contratações...",  # Com newline antes (comum no markdown)
            "\n  § 4º A fase preparatória...",  # Com espaços após newline
        ]

        for text in real_paragraphs:
            # Encontra posição real do § no texto
            start = text.find("§")
            end = start + 5
            result = parser._is_citation_context(text, start, end)
            assert result is False, f"Não deveria detectar como citação: '{text}'"


class TestSpanParserParagraphExtraction:
    """Testes para extração correta de parágrafos."""

    @pytest.fixture
    def parser(self):
        return SpanParser()

    def test_article_with_internal_citation(self, parser):
        """
        Testa artigo com citação interna (bug original).

        O § 2º cita "§ 1º deste artigo" - isso NÃO deve criar um PAR-040-1 extra.
        """
        markdown = """
Art. 40. O planejamento de compras deverá considerar...

§ 1º O estudo técnico preliminar a que se refere o inciso I do caput deverá evidenciar...

§ 2º Para os fins do disposto no § 1º deste artigo, considera-se que há problema a resolver...

§ 3º As contratações de que trata o § 2º serão precedidas de...

§ 4º A fase preparatória do processo licitatório é caracterizada pelo planejamento...
"""

        doc = parser.parse(markdown)

        # Deve ter exatamente 4 parágrafos
        paragrafos = [s for s in doc.spans if s.span_id.startswith("PAR-040-")]
        assert len(paragrafos) == 4, f"Esperado 4 parágrafos, encontrado {len(paragrafos)}: {[p.span_id for p in paragrafos]}"

        # Verifica IDs corretos
        par_ids = sorted([p.span_id for p in paragrafos])
        expected = ["PAR-040-1", "PAR-040-2", "PAR-040-3", "PAR-040-4"]
        assert par_ids == expected, f"IDs incorretos: {par_ids}"

        # Verifica que cada parágrafo começa com o número correto
        for par in paragrafos:
            numero = par.span_id.split("-")[-1]
            if numero != "UNICO":
                assert f"§ {numero}" in par.text[:20], \
                    f"Mismatch: {par.span_id} não começa com § {numero}: '{par.text[:50]}'"

    def test_article_with_paragraph_unico(self, parser):
        """Testa artigo com parágrafo único."""
        markdown = """
Art. 5º Esta Lei aplica-se a:

I - alienação e concessão de direito real de uso de bens;

II - compras, inclusive por encomenda;

Parágrafo único. O disposto nesta Lei não se aplica às empresas estatais.
"""

        doc = parser.parse(markdown)

        # Deve ter PAR-005-UNICO
        par_unico = [s for s in doc.spans if s.span_id == "PAR-005-UNICO"]
        assert len(par_unico) == 1, "Deveria ter PAR-005-UNICO"
        assert "Parágrafo único" in par_unico[0].text

    def test_complex_article_with_multiple_citations(self, parser):
        """Testa artigo complexo com múltiplas citações cruzadas."""
        markdown = """
Art. 75. É dispensável a licitação:

I - para contratação que envolva valores inferiores a R$ 50.000,00;

§ 1º Para fins do disposto no inciso I do caput, considera-se obra...

§ 2º Os valores referidos nos incisos I e II do caput deste artigo serão...

§ 3º As contratações de que tratam os incisos I e II do caput e o § 1º deste artigo serão...

§ 4º Nas contratações realizadas com base nos incisos I e II do caput e no § 3º...
"""

        doc = parser.parse(markdown)

        # Deve ter exatamente 4 parágrafos (as citações a §1º e §3º não devem criar extras)
        paragrafos = [s for s in doc.spans if s.span_id.startswith("PAR-075-")]
        assert len(paragrafos) == 4, f"Esperado 4 parágrafos, encontrado {len(paragrafos)}"

        # Valida consistência
        validator = AddressValidator()
        for par in paragrafos:
            result = validator.validate_span(par)
            assert not result.is_mismatch, f"ADDRESS_MISMATCH detectado: {result.message}"


class TestAddressValidator:
    """Testes para o AddressValidator."""

    @pytest.fixture
    def validator(self):
        return AddressValidator()

    def test_valid_paragraph(self, validator):
        """Parágrafo com ID correto deve passar."""
        from src.parsing.span_models import Span, SpanType

        span = Span(
            span_id="PAR-040-1",
            span_type=SpanType.PARAGRAFO,
            text="§ 1º O estudo técnico preliminar...",
            start_pos=0,
            end_pos=100,
        )

        result = validator.validate_span(span)
        assert result.is_valid is True
        assert result.is_mismatch is False

    def test_mismatch_paragraph(self, validator):
        """Parágrafo com ID incorreto deve falhar."""
        from src.parsing.span_models import Span, SpanType

        span = Span(
            span_id="PAR-040-1",  # Diz ser § 1º
            span_type=SpanType.PARAGRAFO,
            text="§ 4º A fase preparatória...",  # Mas é § 4º
            start_pos=0,
            end_pos=100,
        )

        result = validator.validate_span(span)
        assert result.is_valid is False
        assert result.is_mismatch is True
        assert "ADDRESS_MISMATCH" in result.message

    def test_valid_article(self, validator):
        """Artigo com ID correto deve passar."""
        from src.parsing.span_models import Span, SpanType

        span = Span(
            span_id="ART-044",
            span_type=SpanType.ARTIGO,
            text="Art. 44. O processo de contratação...",
            start_pos=0,
            end_pos=100,
        )

        result = validator.validate_span(span)
        assert result.is_valid is True

    def test_valid_inciso(self, validator):
        """Inciso com ID correto deve passar."""
        from src.parsing.span_models import Span, SpanType

        span = Span(
            span_id="INC-075-II",
            span_type=SpanType.INCISO,
            text="II - para contratação que envolva...",
            start_pos=0,
            end_pos=100,
        )

        result = validator.validate_span(span)
        assert result.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
