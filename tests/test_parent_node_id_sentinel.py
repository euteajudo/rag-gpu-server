"""
Testes Sentinela para parent_node_id no ChunkMaterializer.

Garante que:
1. Artigos (top-level) têm parent_node_id = "" (string vazia)
2. Parágrafos apontam para o artigo pai: "leis:{document_id}#ART-xxx"
3. Incisos apontam para PAR ou ART conforme hierarquia
4. Campo parent_chunk_id NUNCA aparece no payload Milvus final

PR26/01/2025 - Refatoração definitiva parent_chunk_id → parent_node_id
"""

import pytest
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Adiciona src ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestArticleParentNodeIdEmpty:
    """
    Sentinela #1: Artigos (top-level) devem ter parent_node_id = "".

    Artigos são a raiz da hierarquia documental e não possuem pai.
    """

    def test_article_has_empty_parent_node_id(self):
        """Verifica que artigos materializados têm parent_node_id vazio."""
        from chunking.chunk_materializer import ChunkMaterializer, MaterializedChunk
        from parsing.span_models import Span, SpanType, ParsedDocument
        from parsing.article_orchestrator import ArticleChunk

        # Cria ParsedDocument com span de artigo
        parsed_doc = ParsedDocument()
        article_span = Span(
            span_id="ART-005",
            span_type=SpanType.ARTIGO,
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            parent_id=None,
        )
        parsed_doc.add_span(article_span)

        # Cria ArticleChunk para materialização
        article_chunk = ArticleChunk(
            article_id="ART-005",
            article_number="5",
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            citations=["ART-005"],
            inciso_ids=[],
            paragrafo_ids=[],
        )

        materializer = ChunkMaterializer(
            document_id="IN-65-2021",
            tipo_documento="IN",
            numero="65",
            ano=2021,
        )

        # Materializa o artigo
        chunks = materializer.materialize_article(article_chunk, parsed_doc)

        assert len(chunks) >= 1
        article_chunk_result = chunks[0]  # Primeiro chunk é o artigo pai

        # SENTINELA: parent_node_id deve ser string vazia para artigos
        assert article_chunk_result.parent_node_id == "", (
            f"Artigo deveria ter parent_node_id='', mas tem: '{article_chunk_result.parent_node_id}'"
        )

    def test_article_milvus_dict_has_empty_parent_node_id(self):
        """Verifica que to_milvus_dict() de artigo tem parent_node_id vazio."""
        from chunking.chunk_materializer import MaterializedChunk, DeviceType, ChunkLevel

        chunk = MaterializedChunk(
            chunk_id="IN-65-2021#ART-005",
            node_id="leis:IN-65-2021#ART-005",
            parent_node_id="",  # Artigo não tem pai
            span_id="ART-005",
            device_type=DeviceType.ARTICLE,
            chunk_level=ChunkLevel.ARTICLE,
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            retrieval_text="Art. 5º O estudo técnico preliminar será elaborado.",
        )

        milvus_dict = chunk.to_milvus_dict()

        assert "parent_node_id" in milvus_dict
        assert milvus_dict["parent_node_id"] == "", (
            f"Milvus dict deveria ter parent_node_id='', mas tem: '{milvus_dict['parent_node_id']}'"
        )


class TestParagraphParentNodeIdPointsToArticle:
    """
    Sentinela #2: Parágrafos devem apontar para o artigo pai.

    Formato esperado: "leis:{document_id}#ART-xxx"
    """

    def test_paragraph_points_to_article(self):
        """Verifica que parágrafo tem parent_node_id apontando para artigo."""
        from chunking.chunk_materializer import ChunkMaterializer
        from parsing.span_models import Span, SpanType, ParsedDocument
        from parsing.article_orchestrator import ArticleChunk

        # Cria ParsedDocument com artigo e parágrafo
        parsed_doc = ParsedDocument()

        article_span = Span(
            span_id="ART-005",
            span_type=SpanType.ARTIGO,
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            parent_id=None,
        )
        parsed_doc.add_span(article_span)

        paragraph_span = Span(
            span_id="PAR-005-1",
            span_type=SpanType.PARAGRAFO,
            text="§ 1º O estudo técnico preliminar a que se refere...",
            parent_id="ART-005",
        )
        parsed_doc.add_span(paragraph_span)

        # Cria ArticleChunk com parágrafo
        article_chunk = ArticleChunk(
            article_id="ART-005",
            article_number="5",
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            citations=["ART-005", "PAR-005-1"],
            inciso_ids=[],
            paragrafo_ids=["PAR-005-1"],
        )

        materializer = ChunkMaterializer(
            document_id="IN-65-2021",
            tipo_documento="IN",
            numero="65",
            ano=2021,
        )

        chunks = materializer.materialize_article(article_chunk, parsed_doc)

        # Encontra o chunk de parágrafo
        paragraph_chunks = [c for c in chunks if c.span_id == "PAR-005-1"]
        assert len(paragraph_chunks) >= 1, "Deveria ter pelo menos um chunk de parágrafo"

        paragraph_chunk = paragraph_chunks[0]
        expected_parent = "leis:IN-65-2021#ART-005"

        # SENTINELA: parent_node_id deve apontar para o artigo pai
        assert paragraph_chunk.parent_node_id == expected_parent, (
            f"Parágrafo deveria ter parent_node_id='{expected_parent}', "
            f"mas tem: '{paragraph_chunk.parent_node_id}'"
        )

    def test_paragraph_milvus_dict_has_article_parent(self):
        """Verifica que to_milvus_dict() de parágrafo aponta para artigo."""
        from chunking.chunk_materializer import MaterializedChunk, DeviceType, ChunkLevel

        chunk = MaterializedChunk(
            chunk_id="IN-65-2021#PAR-005-1",
            node_id="leis:IN-65-2021#PAR-005-1",
            parent_node_id="leis:IN-65-2021#ART-005",  # Aponta para artigo
            span_id="PAR-005-1",
            device_type=DeviceType.PARAGRAPH,
            chunk_level=ChunkLevel.DEVICE,
            text="§ 1º O estudo técnico preliminar a que se refere...",
            retrieval_text="§ 1º O estudo técnico preliminar a que se refere...",
        )

        milvus_dict = chunk.to_milvus_dict()

        assert "parent_node_id" in milvus_dict
        assert milvus_dict["parent_node_id"] == "leis:IN-65-2021#ART-005", (
            f"Milvus dict de parágrafo deveria apontar para artigo, "
            f"mas tem: '{milvus_dict['parent_node_id']}'"
        )


class TestIncisoParentNodeIdHierarchy:
    """
    Sentinela #3: Incisos devem apontar para PAR ou ART conforme hierarquia.

    - Se inciso é filho de parágrafo: parent_node_id = "leis:{doc}#PAR-xxx-y"
    - Se inciso é filho direto de artigo: parent_node_id = "leis:{doc}#ART-xxx"
    """

    def test_inciso_under_paragraph_points_to_paragraph(self):
        """Inciso filho de parágrafo deve apontar para o parágrafo."""
        from chunking.chunk_materializer import ChunkMaterializer
        from parsing.span_models import Span, SpanType, ParsedDocument
        from parsing.article_orchestrator import ArticleChunk

        # Cria ParsedDocument com hierarquia completa
        parsed_doc = ParsedDocument()

        article_span = Span(
            span_id="ART-005",
            span_type=SpanType.ARTIGO,
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            parent_id=None,
        )
        parsed_doc.add_span(article_span)

        paragraph_span = Span(
            span_id="PAR-005-1",
            span_type=SpanType.PARAGRAFO,
            text="§ 1º Compete ao setor requisitante:",
            parent_id="ART-005",
        )
        parsed_doc.add_span(paragraph_span)

        # Inciso com parent_id apontando para o parágrafo
        inciso_span = Span(
            span_id="INC-005-I",
            span_type=SpanType.INCISO,
            text="I - elaboração do estudo técnico preliminar;",
            parent_id="PAR-005-1",  # Filho do parágrafo
        )
        parsed_doc.add_span(inciso_span)

        # Cria ArticleChunk
        article_chunk = ArticleChunk(
            article_id="ART-005",
            article_number="5",
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            citations=["ART-005", "PAR-005-1", "INC-005-I"],
            inciso_ids=["INC-005-I"],
            paragrafo_ids=["PAR-005-1"],
        )

        materializer = ChunkMaterializer(
            document_id="IN-65-2021",
            tipo_documento="IN",
            numero="65",
            ano=2021,
        )

        chunks = materializer.materialize_article(article_chunk, parsed_doc)

        # Encontra o chunk de inciso
        inciso_chunks = [c for c in chunks if c.span_id == "INC-005-I"]
        assert len(inciso_chunks) >= 1, "Deveria ter pelo menos um chunk de inciso"

        inciso_chunk = inciso_chunks[0]
        expected_parent = "leis:IN-65-2021#PAR-005-1"

        # SENTINELA: inciso filho de parágrafo deve apontar para o parágrafo
        assert inciso_chunk.parent_node_id == expected_parent, (
            f"Inciso filho de parágrafo deveria ter parent_node_id='{expected_parent}', "
            f"mas tem: '{inciso_chunk.parent_node_id}'"
        )

    def test_inciso_under_article_points_to_article(self):
        """Inciso filho direto de artigo deve apontar para o artigo."""
        from chunking.chunk_materializer import ChunkMaterializer
        from parsing.span_models import Span, SpanType, ParsedDocument
        from parsing.article_orchestrator import ArticleChunk

        # Cria ParsedDocument
        parsed_doc = ParsedDocument()

        article_span = Span(
            span_id="ART-005",
            span_type=SpanType.ARTIGO,
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            parent_id=None,
        )
        parsed_doc.add_span(article_span)

        # Inciso diretamente sob o artigo (sem parágrafo intermediário)
        inciso_span = Span(
            span_id="INC-005-I",
            span_type=SpanType.INCISO,
            text="I - elaboração do estudo técnico preliminar;",
            parent_id="ART-005",  # Diretamente filho do artigo
        )
        parsed_doc.add_span(inciso_span)

        # Cria ArticleChunk sem parágrafos
        article_chunk = ArticleChunk(
            article_id="ART-005",
            article_number="5",
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            citations=["ART-005", "INC-005-I"],
            inciso_ids=["INC-005-I"],
            paragrafo_ids=[],  # Sem parágrafos
        )

        materializer = ChunkMaterializer(
            document_id="IN-65-2021",
            tipo_documento="IN",
            numero="65",
            ano=2021,
        )

        chunks = materializer.materialize_article(article_chunk, parsed_doc)

        # Encontra o chunk de inciso
        inciso_chunks = [c for c in chunks if c.span_id == "INC-005-I"]
        assert len(inciso_chunks) >= 1, "Deveria ter pelo menos um chunk de inciso"

        inciso_chunk = inciso_chunks[0]
        expected_parent = "leis:IN-65-2021#ART-005"

        # SENTINELA: inciso direto de artigo deve apontar para o artigo
        assert inciso_chunk.parent_node_id == expected_parent, (
            f"Inciso direto de artigo deveria ter parent_node_id='{expected_parent}', "
            f"mas tem: '{inciso_chunk.parent_node_id}'"
        )


class TestParentChunkIdNeverInMilvusPayload:
    """
    Sentinela #4: Campo parent_chunk_id NUNCA deve aparecer no payload Milvus.

    Após a refatoração, apenas parent_node_id deve existir no output final.
    Este teste garante que não há resquícios do campo antigo.
    """

    def test_milvus_dict_never_has_parent_chunk_id(self):
        """Verifica que to_milvus_dict() nunca contém 'parent_chunk_id'."""
        from chunking.chunk_materializer import MaterializedChunk, DeviceType, ChunkLevel

        # Testa com artigo (parent_node_id vazio)
        article_chunk = MaterializedChunk(
            chunk_id="IN-65-2021#ART-005",
            node_id="leis:IN-65-2021#ART-005",
            parent_node_id="",
            span_id="ART-005",
            device_type=DeviceType.ARTICLE,
            chunk_level=ChunkLevel.ARTICLE,
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            retrieval_text="Art. 5º O estudo técnico preliminar será elaborado.",
        )

        article_dict = article_chunk.to_milvus_dict()

        # SENTINELA CRÍTICA: parent_chunk_id NUNCA deve existir no payload
        assert "parent_chunk_id" not in article_dict, (
            "FALHA SENTINELA: 'parent_chunk_id' encontrado no payload Milvus de artigo! "
            f"Keys presentes: {list(article_dict.keys())}"
        )

        # Testa com parágrafo (parent_node_id preenchido)
        paragraph_chunk = MaterializedChunk(
            chunk_id="IN-65-2021#PAR-005-1",
            node_id="leis:IN-65-2021#PAR-005-1",
            parent_node_id="leis:IN-65-2021#ART-005",
            span_id="PAR-005-1",
            device_type=DeviceType.PARAGRAPH,
            chunk_level=ChunkLevel.DEVICE,
            text="§ 1º O estudo técnico preliminar a que se refere...",
            retrieval_text="§ 1º O estudo técnico preliminar a que se refere...",
        )

        paragraph_dict = paragraph_chunk.to_milvus_dict()

        # SENTINELA CRÍTICA: parent_chunk_id NUNCA deve existir no payload
        assert "parent_chunk_id" not in paragraph_dict, (
            "FALHA SENTINELA: 'parent_chunk_id' encontrado no payload Milvus de parágrafo! "
            f"Keys presentes: {list(paragraph_dict.keys())}"
        )

    def test_all_materialized_chunks_use_parent_node_id_only(self):
        """Verifica que todos os chunks materializados usam apenas parent_node_id."""
        from chunking.chunk_materializer import ChunkMaterializer
        from parsing.span_models import Span, SpanType, ParsedDocument
        from parsing.article_orchestrator import ArticleChunk

        # Cria hierarquia completa: artigo → parágrafo → inciso
        parsed_doc = ParsedDocument()

        article_span = Span(
            span_id="ART-005",
            span_type=SpanType.ARTIGO,
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            parent_id=None,
        )
        parsed_doc.add_span(article_span)

        paragraph_span = Span(
            span_id="PAR-005-1",
            span_type=SpanType.PARAGRAFO,
            text="§ 1º Compete ao setor requisitante:",
            parent_id="ART-005",
        )
        parsed_doc.add_span(paragraph_span)

        inciso_span = Span(
            span_id="INC-005-I",
            span_type=SpanType.INCISO,
            text="I - elaboração do estudo técnico preliminar;",
            parent_id="PAR-005-1",
        )
        parsed_doc.add_span(inciso_span)

        article_chunk = ArticleChunk(
            article_id="ART-005",
            article_number="5",
            text="Art. 5º O estudo técnico preliminar será elaborado.",
            citations=["ART-005", "PAR-005-1", "INC-005-I"],
            inciso_ids=["INC-005-I"],
            paragrafo_ids=["PAR-005-1"],
        )

        materializer = ChunkMaterializer(
            document_id="IN-65-2021",
            tipo_documento="IN",
            numero="65",
            ano=2021,
        )

        chunks = materializer.materialize_article(article_chunk, parsed_doc)

        # SENTINELA: Verifica TODOS os chunks
        for chunk in chunks:
            milvus_dict = chunk.to_milvus_dict()

            # Deve ter parent_node_id
            assert "parent_node_id" in milvus_dict, (
                f"Chunk {chunk.span_id} não tem 'parent_node_id' no payload Milvus"
            )

            # NUNCA deve ter parent_chunk_id
            assert "parent_chunk_id" not in milvus_dict, (
                f"FALHA SENTINELA: Chunk {chunk.span_id} tem 'parent_chunk_id' no payload! "
                f"Isso é um resquício da implementação antiga."
            )


class TestParentNodeIdFormat:
    """
    Testes adicionais para garantir o formato correto de parent_node_id.

    Formato: "leis:{document_id}#{parent_span_id}"
    - Sempre com prefixo "leis:" (para documentos legais)
    - NUNCA com sufixo @Pxx (isso é node_id físico, não lógico)
    """

    def test_parent_node_id_has_leis_prefix(self):
        """Verifica que parent_node_id de filhos tem prefixo 'leis:'."""
        from chunking.chunk_materializer import MaterializedChunk, DeviceType, ChunkLevel

        chunk = MaterializedChunk(
            chunk_id="IN-65-2021#PAR-005-1",
            node_id="leis:IN-65-2021#PAR-005-1",
            parent_node_id="leis:IN-65-2021#ART-005",
            span_id="PAR-005-1",
            device_type=DeviceType.PARAGRAPH,
            chunk_level=ChunkLevel.DEVICE,
            text="§ 1º Texto do parágrafo.",
            retrieval_text="§ 1º Texto do parágrafo.",
        )

        assert chunk.parent_node_id.startswith("leis:"), (
            f"parent_node_id deveria começar com 'leis:', mas tem: '{chunk.parent_node_id}'"
        )

    def test_parent_node_id_never_has_part_suffix(self):
        """Verifica que parent_node_id NUNCA tem sufixo @Pxx."""
        from chunking.chunk_materializer import MaterializedChunk, DeviceType, ChunkLevel

        chunk = MaterializedChunk(
            chunk_id="IN-65-2021#PAR-005-1",
            node_id="leis:IN-65-2021#PAR-005-1",
            parent_node_id="leis:IN-65-2021#ART-005",  # Correto: sem @Pxx
            span_id="PAR-005-1",
            device_type=DeviceType.PARAGRAPH,
            chunk_level=ChunkLevel.DEVICE,
            text="§ 1º Texto do parágrafo.",
            retrieval_text="§ 1º Texto do parágrafo.",
        )

        assert "@P" not in chunk.parent_node_id, (
            f"parent_node_id NÃO deveria ter sufixo @Pxx (node_id físico), "
            f"mas tem: '{chunk.parent_node_id}'"
        )

    def test_empty_parent_node_id_is_string_not_none(self):
        """Verifica que parent_node_id vazio é string vazia, não None."""
        from chunking.chunk_materializer import MaterializedChunk, DeviceType, ChunkLevel

        chunk = MaterializedChunk(
            chunk_id="IN-65-2021#ART-005",
            node_id="leis:IN-65-2021#ART-005",
            parent_node_id="",  # String vazia, não None
            span_id="ART-005",
            device_type=DeviceType.ARTICLE,
            chunk_level=ChunkLevel.ARTICLE,
            text="Art. 5º Texto.",
            retrieval_text="Art. 5º Texto.",
        )

        # Deve ser string vazia, não None
        assert chunk.parent_node_id is not None, (
            "parent_node_id de artigo deveria ser string vazia, não None"
        )
        assert isinstance(chunk.parent_node_id, str), (
            "parent_node_id deveria ser string"
        )
        assert chunk.parent_node_id == "", (
            f"parent_node_id de artigo deveria ser '', mas é: {chunk.parent_node_id!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
