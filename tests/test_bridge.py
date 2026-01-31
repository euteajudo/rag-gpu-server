"""
Testes do módulo bridge: ParsedDocument -> ChunkPart[].

PR3 v2.1 - Rebase

Testa a conversão de estruturas do parsing (ParsedDocument, Span)
para estruturas físicas (ChunkPart) usadas no Milvus.
"""

import pytest
from typing import List

from src.parsing.span_models import SpanType, Span, ParsedDocument
from src.spans.span_types import DeviceType, ChunkPart
from src.bridge import (
    ParsedDocumentChunkPartsBuilder,
    build_chunk_parts,
    map_span_type_to_device_type,
    find_root_article_span_id,
)


class TestSpanTypeToDeviceTypeMapping:
    """Testes do mapeamento SpanType -> DeviceType."""

    def test_artigo_to_article(self):
        """SpanType.ARTIGO -> DeviceType.ARTICLE."""
        assert map_span_type_to_device_type(SpanType.ARTIGO) == DeviceType.ARTICLE

    def test_paragrafo_to_paragraph(self):
        """SpanType.PARAGRAFO -> DeviceType.PARAGRAPH."""
        assert map_span_type_to_device_type(SpanType.PARAGRAFO) == DeviceType.PARAGRAPH

    def test_inciso_to_inciso(self):
        """SpanType.INCISO -> DeviceType.INCISO."""
        assert map_span_type_to_device_type(SpanType.INCISO) == DeviceType.INCISO

    def test_alinea_to_alinea(self):
        """SpanType.ALINEA -> DeviceType.ALINEA."""
        assert map_span_type_to_device_type(SpanType.ALINEA) == DeviceType.ALINEA

    def test_header_to_ementa(self):
        """SpanType.HEADER -> DeviceType.EMENTA."""
        assert map_span_type_to_device_type(SpanType.HEADER) == DeviceType.EMENTA

    def test_structural_types_to_unknown(self):
        """Tipos estruturais (CAPITULO, SECAO) mapeiam para UNKNOWN."""
        structural_types = [
            SpanType.CAPITULO,
            SpanType.SECAO,
            SpanType.SUBSECAO,
            SpanType.TITULO,
            SpanType.TEXTO,
            SpanType.ASSINATURA,
        ]
        for span_type in structural_types:
            result = map_span_type_to_device_type(span_type)
            assert result == DeviceType.UNKNOWN, f"{span_type} deveria mapear para UNKNOWN"

    def test_item_to_alinea(self):
        """SpanType.ITEM -> DeviceType.ALINEA (similar a alínea)."""
        assert map_span_type_to_device_type(SpanType.ITEM) == DeviceType.ALINEA


class TestFindRootArticleSpanId:
    """Testes para find_root_article_span_id."""

    @pytest.fixture
    def sample_parsed_doc(self) -> ParsedDocument:
        """Cria um ParsedDocument de exemplo com hierarquia."""
        spans = [
            Span(span_id="ART-001", span_type=SpanType.ARTIGO, text="Art. 1º Teste.", parent_id=None),
            Span(span_id="PAR-001-1", span_type=SpanType.PARAGRAFO, text="§ 1º Parágrafo.", parent_id="ART-001"),
            Span(span_id="INC-001-I", span_type=SpanType.INCISO, text="I - inciso.", parent_id="ART-001"),
            Span(span_id="ALI-001-I-a", span_type=SpanType.ALINEA, text="a) alínea.", parent_id="INC-001-I"),
            Span(span_id="ART-002", span_type=SpanType.ARTIGO, text="Art. 2º Teste 2.", parent_id=None),
        ]
        return ParsedDocument(spans=spans)

    def test_article_returns_itself(self, sample_parsed_doc):
        """Artigo retorna seu próprio span_id."""
        artigo = sample_parsed_doc.get_span("ART-001")
        result = find_root_article_span_id(artigo, sample_parsed_doc)
        assert result == "ART-001"

    def test_paragraph_returns_parent_article(self, sample_parsed_doc):
        """Parágrafo retorna o artigo pai."""
        paragrafo = sample_parsed_doc.get_span("PAR-001-1")
        result = find_root_article_span_id(paragrafo, sample_parsed_doc)
        assert result == "ART-001"

    def test_inciso_returns_root_article(self, sample_parsed_doc):
        """Inciso retorna o artigo raiz."""
        inciso = sample_parsed_doc.get_span("INC-001-I")
        result = find_root_article_span_id(inciso, sample_parsed_doc)
        assert result == "ART-001"

    def test_alinea_returns_root_article(self, sample_parsed_doc):
        """Alínea navega pela hierarquia até encontrar o artigo raiz."""
        alinea = sample_parsed_doc.get_span("ALI-001-I-a")
        result = find_root_article_span_id(alinea, sample_parsed_doc)
        assert result == "ART-001"

    def test_orphan_span_uses_fallback(self):
        """Span sem parent usa fallback baseado no span_id."""
        # Cria span órfão mas com span_id que segue o padrão
        orphan = Span(span_id="INC-005-I", span_type=SpanType.INCISO, text="I - teste.", parent_id=None)
        artigo = Span(span_id="ART-005", span_type=SpanType.ARTIGO, text="Art. 5º Teste.", parent_id=None)
        doc = ParsedDocument(spans=[artigo, orphan])

        result = find_root_article_span_id(orphan, doc)
        assert result == "ART-005"

    def test_span_without_article_returns_none(self):
        """Span sem artigo na hierarquia e sem fallback retorna None."""
        orphan = Span(span_id="TEXTO-001", span_type=SpanType.TEXTO, text="Texto.", parent_id=None)
        doc = ParsedDocument(spans=[orphan])

        result = find_root_article_span_id(orphan, doc)
        assert result is None


class TestBuildChunkParts:
    """Testes para build_chunk_parts."""

    @pytest.fixture
    def simple_parsed_doc(self) -> ParsedDocument:
        """Cria um ParsedDocument simples para teste."""
        spans = [
            Span(span_id="ART-001", span_type=SpanType.ARTIGO, text="Art. 1º Teste.", parent_id=None),
            Span(span_id="INC-001-I", span_type=SpanType.INCISO, text="I - inciso.", parent_id="ART-001"),
        ]
        return ParsedDocument(spans=spans)

    def test_creates_chunk_parts(self, simple_parsed_doc):
        """Deve criar ChunkParts a partir do ParsedDocument."""
        chunks = build_chunk_parts(
            parsed_doc=simple_parsed_doc,
            document_id="LEI-14133-2021",
            document_type="LEI",
        )

        # Deve criar 2 chunks (1 artigo + 1 inciso)
        assert len(chunks) == 2

    def test_chunk_part_has_correct_ids(self, simple_parsed_doc):
        """ChunkPart deve ter IDs no formato correto."""
        chunks = build_chunk_parts(
            parsed_doc=simple_parsed_doc,
            document_id="LEI-14133-2021",
            document_type="LEI",
        )

        article_chunk = next(c for c in chunks if c.span_id == "ART-001")

        # logical_node_id: prefix:document_id#span_id
        assert article_chunk.logical_node_id == "leis:LEI-14133-2021#ART-001"

        # node_id: logical_node_id@P00
        assert article_chunk.node_id == "leis:LEI-14133-2021#ART-001@P00"

        # chunk_id: document_id#span_id@P00
        assert article_chunk.chunk_id == "LEI-14133-2021#ART-001@P00"

    def test_chunk_part_has_correct_device_type(self, simple_parsed_doc):
        """ChunkPart deve ter device_type mapeado corretamente."""
        chunks = build_chunk_parts(
            parsed_doc=simple_parsed_doc,
            document_id="LEI-14133-2021",
            document_type="LEI",
        )

        article_chunk = next(c for c in chunks if c.span_id == "ART-001")
        inciso_chunk = next(c for c in chunks if c.span_id == "INC-001-I")

        assert article_chunk.device_type == DeviceType.ARTICLE
        assert inciso_chunk.device_type == DeviceType.INCISO

    def test_chunk_part_has_article_number(self, simple_parsed_doc):
        """ChunkPart deve ter article_number extraído do artigo raiz."""
        chunks = build_chunk_parts(
            parsed_doc=simple_parsed_doc,
            document_id="LEI-14133-2021",
            document_type="LEI",
        )

        for chunk in chunks:
            assert chunk.article_number == "001", f"chunk {chunk.span_id} deveria ter article_number='001'"

    def test_chunk_part_has_parent_chunk_id(self, simple_parsed_doc):
        """ChunkPart de inciso deve ter parent_chunk_id apontando para artigo."""
        chunks = build_chunk_parts(
            parsed_doc=simple_parsed_doc,
            document_id="LEI-14133-2021",
            document_type="LEI",
        )

        article_chunk = next(c for c in chunks if c.span_id == "ART-001")
        inciso_chunk = next(c for c in chunks if c.span_id == "INC-001-I")

        # Artigo não tem parent
        assert article_chunk.parent_chunk_id is None

        # Inciso aponta para artigo
        assert inciso_chunk.parent_chunk_id == "LEI-14133-2021#ART-001@P00"

    def test_ignores_unknown_device_types(self):
        """Spans com device_type UNKNOWN não devem gerar ChunkParts."""
        spans = [
            Span(span_id="ART-001", span_type=SpanType.ARTIGO, text="Art. 1º Teste.", parent_id=None),
            Span(span_id="CAP-I", span_type=SpanType.CAPITULO, text="CAPÍTULO I", parent_id=None),
            Span(span_id="SEC-I", span_type=SpanType.SECAO, text="Seção I", parent_id=None),
        ]
        doc = ParsedDocument(spans=spans)

        chunks = build_chunk_parts(
            parsed_doc=doc,
            document_id="LEI-14133-2021",
            document_type="LEI",
        )

        # Só o artigo deve gerar ChunkPart
        assert len(chunks) == 1
        assert chunks[0].span_id == "ART-001"

    def test_custom_prefix(self, simple_parsed_doc):
        """Deve usar prefix customizado quando fornecido."""
        chunks = build_chunk_parts(
            parsed_doc=simple_parsed_doc,
            document_id="ACORDAO-123-2021",
            document_type="ACORDAO",
            prefix="tcu",
        )

        article_chunk = next(c for c in chunks if c.span_id == "ART-001")
        assert article_chunk.logical_node_id == "tcu:ACORDAO-123-2021#ART-001"


class TestParsedDocumentChunkPartsBuilder:
    """Testes para ParsedDocumentChunkPartsBuilder."""

    def test_builder_creates_chunk_parts(self):
        """Builder deve criar ChunkParts a partir do ParsedDocument."""
        spans = [
            Span(span_id="ART-001", span_type=SpanType.ARTIGO, text="Art. 1º Teste.", parent_id=None),
        ]
        doc = ParsedDocument(spans=spans)

        builder = ParsedDocumentChunkPartsBuilder(
            document_id="LEI-14133-2021",
            document_type="LEI",
        )
        chunks = builder.build(doc)

        assert len(chunks) == 1
        assert chunks[0].logical_node_id == "leis:LEI-14133-2021#ART-001"

    def test_builder_uses_default_prefix(self):
        """Builder deve usar prefix padrão baseado no document_type."""
        builder = ParsedDocumentChunkPartsBuilder(
            document_id="IN-58-2022",
            document_type="IN",
        )
        assert builder.prefix == "ins"

    def test_builder_uses_custom_prefix(self):
        """Builder deve usar prefix customizado quando fornecido."""
        builder = ParsedDocumentChunkPartsBuilder(
            document_id="AC-1234-2021",
            document_type="ACORDAO",
            prefix="acordaos",
        )
        assert builder.prefix == "acordaos"

    def test_build_from_spans(self):
        """build_from_spans deve criar ChunkParts a partir de lista de Spans."""
        spans = [
            Span(span_id="ART-001", span_type=SpanType.ARTIGO, text="Art. 1º Teste.", parent_id=None),
            Span(span_id="ART-002", span_type=SpanType.ARTIGO, text="Art. 2º Teste 2.", parent_id=None),
        ]

        builder = ParsedDocumentChunkPartsBuilder(
            document_id="LEI-14133-2021",
            document_type="LEI",
        )
        chunks = builder.build_from_spans(spans)

        assert len(chunks) == 2


class TestSpanIdPreservation:
    """
    Testes que verificam que span_id é preservado corretamente.

    Importante: O span_id vem diretamente do SpanParser e NÃO contém
    sufixos PART-*. IDs como ART-006-PART-1 são usados APENAS internamente
    no ArticleOrchestrator para dividir artigos muito grandes.
    """

    def test_span_id_preserved_without_part_suffix(self):
        """span_id deve ser preservado sem sufixo PART-*."""
        spans = [
            Span(span_id="ART-005", span_type=SpanType.ARTIGO, text="Art. 5º Teste.", parent_id=None),
            Span(span_id="PAR-005-1", span_type=SpanType.PARAGRAFO, text="§ 1º Par.", parent_id="ART-005"),
            Span(span_id="INC-005-I", span_type=SpanType.INCISO, text="I - inciso.", parent_id="ART-005"),
        ]
        doc = ParsedDocument(spans=spans)

        chunks = build_chunk_parts(
            parsed_doc=doc,
            document_id="IN-58-2022",
            document_type="IN",
        )

        # Verifica que nenhum span_id contém PART-*
        for chunk in chunks:
            assert "-PART-" not in chunk.span_id, f"span_id não deve conter PART-*: {chunk.span_id}"
            assert "-P" not in chunk.span_id or chunk.span_id.startswith("PAR-"), \
                f"span_id não deve conter sufixo de parte: {chunk.span_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
