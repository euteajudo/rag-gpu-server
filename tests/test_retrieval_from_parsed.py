"""
Testes das classes de retrieval para ParsedDocument.

PR3 v2.1 - Rebase

Testa RetrievalTextBuilderFromParsedDocument e ParentTextResolverFromParsedDocument.
"""

import pytest
from typing import Optional

from src.parsing.span_models import SpanType, Span, ParsedDocument
from src.retrieval import (
    RetrievalTextBuilderFromParsedDocument,
    ParentTextResolverFromParsedDocument,
    RetrievalContext,
)


class TestParentTextResolverFromParsedDocument:
    """Testes para ParentTextResolverFromParsedDocument."""

    @pytest.fixture
    def sample_parsed_doc(self) -> ParsedDocument:
        """Cria um ParsedDocument de exemplo com hierarquia."""
        spans = [
            Span(
                span_id="ART-001",
                span_type=SpanType.ARTIGO,
                text="Art. 1º Este artigo testa a funcionalidade do sistema.",
                parent_id=None,
            ),
            Span(
                span_id="PAR-001-1",
                span_type=SpanType.PARAGRAFO,
                text="§ 1º Este é o primeiro parágrafo.",
                parent_id="ART-001",
            ),
            Span(
                span_id="INC-001-I",
                span_type=SpanType.INCISO,
                text="I - primeiro inciso;",
                parent_id="ART-001",
            ),
            Span(
                span_id="INC-001-II",
                span_type=SpanType.INCISO,
                text="II - segundo inciso;",
                parent_id="ART-001",
            ),
            Span(
                span_id="ALI-001-I-a",
                span_type=SpanType.ALINEA,
                text="a) primeira alínea;",
                parent_id="INC-001-I",
            ),
            Span(
                span_id="ART-002",
                span_type=SpanType.ARTIGO,
                text="Art. 2º Segundo artigo para teste.",
                parent_id=None,
            ),
        ]
        return ParsedDocument(spans=spans)

    @pytest.fixture
    def resolver(self, sample_parsed_doc) -> ParentTextResolverFromParsedDocument:
        """Cria um resolver com o documento de exemplo."""
        return ParentTextResolverFromParsedDocument(sample_parsed_doc)

    def test_article_returns_none(self, resolver, sample_parsed_doc):
        """Artigo não tem parent, deve retornar None."""
        article = sample_parsed_doc.get_span("ART-001")
        result = resolver.resolve_parent_text(article)
        assert result is None

    def test_paragraph_returns_article_text(self, resolver, sample_parsed_doc):
        """Parágrafo deve retornar texto do artigo pai."""
        paragraph = sample_parsed_doc.get_span("PAR-001-1")
        result = resolver.resolve_parent_text(paragraph)

        assert result is not None
        assert "Art. 1º" in result

    def test_inciso_returns_article_text(self, resolver, sample_parsed_doc):
        """Inciso deve retornar texto do artigo pai."""
        inciso = sample_parsed_doc.get_span("INC-001-I")
        result = resolver.resolve_parent_text(inciso)

        assert result is not None
        assert "Art. 1º" in result

    def test_alinea_returns_inciso_text(self, resolver, sample_parsed_doc):
        """Alínea retorna texto do inciso direto (seu parent), não do artigo raiz."""
        alinea = sample_parsed_doc.get_span("ALI-001-I-a")
        result = resolver.resolve_parent_text(alinea)

        # Retorna o parent direto (inciso), não o artigo raiz
        assert result is not None
        assert "I - primeiro inciso" in result

    def test_orphan_span_returns_none(self):
        """Span sem parent_id retorna None."""
        orphan = Span(
            span_id="TEXTO-001",
            span_type=SpanType.TEXTO,
            text="Texto órfão.",
            parent_id=None,
        )
        doc = ParsedDocument(spans=[orphan])
        resolver = ParentTextResolverFromParsedDocument(doc)

        result = resolver.resolve_parent_text(orphan)
        assert result is None

    def test_invalid_parent_id_returns_none(self):
        """Span com parent_id inválido retorna None."""
        invalid_span = Span(
            span_id="INC-999-I",
            span_type=SpanType.INCISO,
            text="I - inciso com parent inválido;",
            parent_id="ART-999",  # Não existe no documento
        )
        doc = ParsedDocument(spans=[invalid_span])
        resolver = ParentTextResolverFromParsedDocument(doc)

        result = resolver.resolve_parent_text(invalid_span)
        assert result is None

    def test_caches_parent_text(self, resolver, sample_parsed_doc):
        """Resolver deve cachear texto do parent."""
        inciso_i = sample_parsed_doc.get_span("INC-001-I")
        inciso_ii = sample_parsed_doc.get_span("INC-001-II")

        # Primeira chamada
        result_1 = resolver.resolve_parent_text(inciso_i)
        assert result_1 is not None

        # Segunda chamada para mesmo parent (diferente filho)
        result_2 = resolver.resolve_parent_text(inciso_ii)
        assert result_2 is not None

        # Ambos devem ter o mesmo texto do parent (Art. 1)
        assert result_1 == result_2

        # Verifica que está no cache
        assert "ART-001" in resolver._parent_text_cache


class TestRetrievalContext:
    """Testes para RetrievalContext dataclass."""

    def test_retrieval_context_creation(self):
        """Deve criar RetrievalContext com todos os campos."""
        ctx = RetrievalContext(
            retrieval_text="[CONTEXTO: ...]\n\nI - inciso;",
            parent_text="Art. 5º O caput do artigo.",
            context_header="LEI LEI-14133-2021, Art. 5, inciso I",
        )

        assert ctx.retrieval_text == "[CONTEXTO: ...]\n\nI - inciso;"
        assert ctx.parent_text == "Art. 5º O caput do artigo."
        assert ctx.context_header == "LEI LEI-14133-2021, Art. 5, inciso I"

    def test_retrieval_context_optional_parent(self):
        """parent_text é opcional."""
        ctx = RetrievalContext(
            retrieval_text="[CONTEXTO: ...]\n\nArt. 5º Texto.",
            parent_text=None,
            context_header="LEI LEI-14133-2021, Art. 5",
        )

        assert ctx.parent_text is None


class TestRetrievalTextBuilderFromParsedDocument:
    """Testes para RetrievalTextBuilderFromParsedDocument."""

    @pytest.fixture
    def sample_parsed_doc(self) -> ParsedDocument:
        """Cria um ParsedDocument de exemplo."""
        spans = [
            Span(
                span_id="ART-014",
                span_type=SpanType.ARTIGO,
                text="Art. 14. A elaboração do ETP é facultada nas seguintes hipóteses:",
                parent_id=None,
            ),
            Span(
                span_id="INC-014-I",
                span_type=SpanType.INCISO,
                text="I - é facultada nas hipóteses previstas no inciso III do art. 75;",
                parent_id="ART-014",
            ),
            Span(
                span_id="INC-014-II",
                span_type=SpanType.INCISO,
                text="II - nos casos de prorrogações de contratos contínuos;",
                parent_id="ART-014",
            ),
            Span(
                span_id="PAR-014-UNICO",
                span_type=SpanType.PARAGRAFO,
                text="Parágrafo único. O ETP simplificado deverá conter...",
                parent_id="ART-014",
            ),
        ]
        return ParsedDocument(spans=spans)

    @pytest.fixture
    def builder(self, sample_parsed_doc) -> RetrievalTextBuilderFromParsedDocument:
        """Cria um builder com o documento de exemplo."""
        return RetrievalTextBuilderFromParsedDocument(
            parsed_doc=sample_parsed_doc,
            document_id="IN-58-2022",
            document_type="IN",
        )

    def test_build_returns_retrieval_context(self, builder, sample_parsed_doc):
        """build() deve retornar RetrievalContext."""
        article = sample_parsed_doc.get_span("ART-014")
        result = builder.build(article)

        assert isinstance(result, RetrievalContext)
        assert hasattr(result, "retrieval_text")
        assert hasattr(result, "parent_text")
        assert hasattr(result, "context_header")

    def test_build_for_article(self, builder, sample_parsed_doc):
        """Deve construir RetrievalContext para artigo."""
        article = sample_parsed_doc.get_span("ART-014")
        result = builder.build(article)

        # retrieval_text contém o texto
        assert "A elaboração do ETP é facultada" in result.retrieval_text

        # context_header contém identificadores
        assert "IN" in result.context_header
        assert "IN-58-2022" in result.context_header

        # Artigo não tem parent_text
        assert result.parent_text is None

    def test_build_for_inciso(self, builder, sample_parsed_doc):
        """Deve construir RetrievalContext para inciso com parent."""
        inciso = sample_parsed_doc.get_span("INC-014-I")
        result = builder.build(inciso)

        # retrieval_text contém o texto do inciso
        assert "é facultada nas hipóteses previstas" in result.retrieval_text

        # context_header contém identificadores
        assert "inciso" in result.context_header.lower()

        # parent_text contém texto do artigo pai
        assert result.parent_text is not None
        assert "A elaboração do ETP é facultada" in result.parent_text

    def test_build_for_paragraph_unico(self, builder, sample_parsed_doc):
        """Deve construir RetrievalContext para parágrafo único."""
        paragraph = sample_parsed_doc.get_span("PAR-014-UNICO")
        result = builder.build(paragraph)

        # retrieval_text contém o texto do parágrafo
        assert "O ETP simplificado" in result.retrieval_text

        # parent_text contém texto do artigo pai
        assert result.parent_text is not None

    def test_different_incisos_different_output(self, builder, sample_parsed_doc):
        """Incisos diferentes do mesmo artigo devem gerar RetrievalContext diferentes."""
        inciso_i = sample_parsed_doc.get_span("INC-014-I")
        inciso_ii = sample_parsed_doc.get_span("INC-014-II")

        result_i = builder.build(inciso_i)
        result_ii = builder.build(inciso_ii)

        # retrieval_text devem ser diferentes
        assert result_i.retrieval_text != result_ii.retrieval_text

        # Mas ambos têm o mesmo parent_text (artigo pai)
        assert result_i.parent_text == result_ii.parent_text

    def test_deterministic_output(self, builder, sample_parsed_doc):
        """retrieval_text deve ser determinístico (mesmo input = mesmo output)."""
        inciso = sample_parsed_doc.get_span("INC-014-I")

        results = [builder.build(inciso) for _ in range(5)]

        # Todos os retrieval_text devem ser idênticos
        retrieval_texts = [r.retrieval_text for r in results]
        assert len(set(retrieval_texts)) == 1, "retrieval_text deve ser determinístico"

    def test_build_all_returns_dict(self, builder, sample_parsed_doc):
        """build_all() deve retornar dicionário span_id -> RetrievalContext."""
        result = builder.build_all()

        assert isinstance(result, dict)
        assert len(result) == len(sample_parsed_doc.spans)

        # Cada valor deve ser RetrievalContext
        for span_id, ctx in result.items():
            assert isinstance(ctx, RetrievalContext)


class TestRetrievalTextBuilderEdgeCases:
    """Testes de casos extremos para RetrievalTextBuilderFromParsedDocument."""

    def test_span_without_parent(self):
        """Span sem parent_id não deve ter parent_text."""
        doc = ParsedDocument(spans=[
            Span(span_id="ART-001", span_type=SpanType.ARTIGO, text="Art. 1º Texto.", parent_id=None),
        ])
        builder = RetrievalTextBuilderFromParsedDocument(doc, "LEI-14133-2021", "LEI")

        result = builder.build(doc.get_span("ART-001"))

        assert result.parent_text is None

    def test_orphan_inciso_without_article(self):
        """Inciso órfão (sem artigo no doc) deve funcionar sem parent."""
        orphan_inciso = Span(
            span_id="INC-999-I",
            span_type=SpanType.INCISO,
            text="I - inciso órfão;",
            parent_id="ART-999",  # Não existe
        )
        doc = ParsedDocument(spans=[orphan_inciso])
        builder = RetrievalTextBuilderFromParsedDocument(doc, "LEI-14133-2021", "LEI")

        result = builder.build(orphan_inciso)

        # Deve funcionar, apenas sem parent_text
        assert "I - inciso órfão" in result.retrieval_text
        # parent_text deve ser None pois parent não existe
        assert result.parent_text is None

    def test_empty_text_span(self):
        """Span com texto vazio deve funcionar."""
        empty_span = Span(
            span_id="ART-001",
            span_type=SpanType.ARTIGO,
            text="",
            parent_id=None,
        )
        doc = ParsedDocument(spans=[empty_span])
        builder = RetrievalTextBuilderFromParsedDocument(doc, "LEI-14133-2021", "LEI")

        result = builder.build(empty_span)

        # Deve funcionar, context_header preenchido
        assert result.context_header  # Não vazio


class TestArticleNumberExtraction:
    """Testes de extração do article_number do span."""

    def test_article_number_in_context_header(self):
        """article_number deve aparecer no context_header."""
        spans = [
            Span(
                span_id="ART-014",
                span_type=SpanType.ARTIGO,
                text="Art. 14. Texto.",
                parent_id=None,
                article_number="14",
            ),
        ]
        doc = ParsedDocument(spans=spans)
        builder = RetrievalTextBuilderFromParsedDocument(doc, "IN-58-2022", "IN")

        result = builder.build(spans[0])

        # context_header deve conter referência ao artigo
        assert "14" in result.context_header or "Art" in result.context_header


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
