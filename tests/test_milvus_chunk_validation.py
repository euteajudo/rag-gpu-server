"""
Testes de validação do MilvusChunk.

PR3 v2.1 - Hard Reset RAG Architecture

Verifica que:
1. device_type é normalizado para enum (ART|PAR|INC|ALI)
2. article_number_int é derivado automaticamente
3. parent_chunk_id é validado (artigo@P00 para filhos, vazio para artigos)
"""

import pytest
from dataclasses import FrozenInstanceError


class TestDeviceTypeNormalization:
    """Testes para normalização de device_type."""

    def test_device_type_lowercase_normalized(self):
        """Verifica que device_type lowercase é normalizado para CAPS."""
        from sinks.milvus_writer import MilvusChunk

        chunk = MilvusChunk(
            node_id="leis:LEI-14133-2021#ART-005@P00",
            logical_node_id="leis:LEI-14133-2021#ART-005",
            chunk_id="LEI-14133-2021#ART-005@P00",
            parent_chunk_id=None,
            part_index=0,
            part_total=1,
            text="Art. 5º Texto do artigo.",
            retrieval_text="Texto para busca.",
            parent_text=None,
            device_type="article",
        )

        assert chunk.device_type == "ART"

    def test_device_type_variations(self):
        """Verifica que várias variações de device_type são normalizadas."""
        from sinks.milvus_writer import DeviceType

        # Variações de artigo
        assert DeviceType.from_string("art").value == "ART"
        assert DeviceType.from_string("article").value == "ART"
        assert DeviceType.from_string("artigo").value == "ART"
        assert DeviceType.from_string("ART").value == "ART"

        # Variações de parágrafo
        assert DeviceType.from_string("par").value == "PAR"
        assert DeviceType.from_string("paragraph").value == "PAR"
        assert DeviceType.from_string("paragrafo").value == "PAR"
        assert DeviceType.from_string("parágrafo").value == "PAR"

        # Variações de inciso
        assert DeviceType.from_string("inc").value == "INC"
        assert DeviceType.from_string("inciso").value == "INC"

        # Variações de alínea
        assert DeviceType.from_string("ali").value == "ALI"
        assert DeviceType.from_string("alinea").value == "ALI"
        assert DeviceType.from_string("alínea").value == "ALI"

        # Desconhecido
        assert DeviceType.from_string("outro").value == "UNKNOWN"
        assert DeviceType.from_string("").value == "UNKNOWN"


class TestArticleNumberIntDerivation:
    """Testes para derivação automática de article_number_int."""

    def test_article_number_int_derived_from_simple_number(self):
        """Verifica que article_number_int é derivado de números simples."""
        from sinks.milvus_writer import MilvusChunk

        chunk = MilvusChunk(
            node_id="leis:LEI-14133-2021#ART-005@P00",
            logical_node_id="leis:LEI-14133-2021#ART-005",
            chunk_id="LEI-14133-2021#ART-005@P00",
            parent_chunk_id=None,
            part_index=0,
            part_total=1,
            text="Art. 5º Texto.",
            retrieval_text="Texto.",
            parent_text=None,
            device_type="article",
            article_number="5",
        )

        assert chunk.article_number_int == 5

    def test_article_number_int_derived_from_complex_number(self):
        """Verifica que article_number_int extrai número de formatos complexos."""
        from sinks.milvus_writer import MilvusChunk

        # Formato "10-A"
        chunk = MilvusChunk(
            node_id="test@P00",
            logical_node_id="test",
            chunk_id="test@P00",
            parent_chunk_id=None,
            part_index=0,
            part_total=1,
            text="Art. 10-A.",
            retrieval_text="Texto.",
            parent_text=None,
            device_type="article",
            article_number="10-A",
        )

        assert chunk.article_number_int == 10

    def test_article_number_int_preserves_explicit_value(self):
        """Verifica que article_number_int explícito não é sobrescrito."""
        from sinks.milvus_writer import MilvusChunk

        chunk = MilvusChunk(
            node_id="test@P00",
            logical_node_id="test",
            chunk_id="test@P00",
            parent_chunk_id=None,
            part_index=0,
            part_total=1,
            text="Art. 5.",
            retrieval_text="Texto.",
            parent_text=None,
            device_type="article",
            article_number="5",
            article_number_int=999,  # Explícito
        )

        assert chunk.article_number_int == 999  # Mantém valor explícito


class TestParentChunkIdValidation:
    """Testes para validação de parent_chunk_id."""

    def test_article_clears_parent_chunk_id(self):
        """Verifica que artigos têm parent_chunk_id limpo."""
        from sinks.milvus_writer import MilvusChunk

        chunk = MilvusChunk(
            node_id="leis:LEI-14133-2021#ART-005@P00",
            logical_node_id="leis:LEI-14133-2021#ART-005",
            chunk_id="LEI-14133-2021#ART-005@P00",
            parent_chunk_id="LEI-14133-2021#ART-001@P00",  # Será limpo
            part_index=0,
            part_total=1,
            text="Art. 5º Texto.",
            retrieval_text="Texto.",
            parent_text=None,
            device_type="article",
        )

        assert chunk.parent_chunk_id is None  # Limpo automaticamente

    def test_child_keeps_valid_parent_chunk_id(self):
        """Verifica que filhos mantêm parent_chunk_id válido (@P00)."""
        from sinks.milvus_writer import MilvusChunk

        chunk = MilvusChunk(
            node_id="leis:LEI-14133-2021#PAR-005-1@P00",
            logical_node_id="leis:LEI-14133-2021#PAR-005-1",
            chunk_id="LEI-14133-2021#PAR-005-1@P00",
            parent_chunk_id="LEI-14133-2021#ART-005@P00",  # Válido
            part_index=0,
            part_total=1,
            text="§ 1º Texto do parágrafo.",
            retrieval_text="Texto.",
            parent_text="Art. 5º Texto do artigo.",
            device_type="paragraph",
        )

        assert chunk.parent_chunk_id == "LEI-14133-2021#ART-005@P00"


class TestSchemaVersion:
    """Testes para schema_version atualizado."""

    def test_schema_version_is_2_1_0(self):
        """Verifica que schema_version é 2.1.0."""
        from sinks.milvus_writer import MilvusChunk

        chunk = MilvusChunk(
            node_id="test@P00",
            logical_node_id="test",
            chunk_id="test@P00",
            parent_chunk_id=None,
            part_index=0,
            part_total=1,
            text="Texto.",
            retrieval_text="Texto.",
            parent_text=None,
        )

        assert chunk.schema_version == "2.1.0"


class TestDocumentVersion:
    """Testes para campo document_version."""

    def test_document_version_optional(self):
        """Verifica que document_version é opcional."""
        from sinks.milvus_writer import MilvusChunk

        chunk = MilvusChunk(
            node_id="test@P00",
            logical_node_id="test",
            chunk_id="test@P00",
            parent_chunk_id=None,
            part_index=0,
            part_total=1,
            text="Texto.",
            retrieval_text="Texto.",
            parent_text=None,
        )

        assert chunk.document_version is None

    def test_document_version_set(self):
        """Verifica que document_version pode ser definido."""
        from sinks.milvus_writer import MilvusChunk

        chunk = MilvusChunk(
            node_id="test@P00",
            logical_node_id="test",
            chunk_id="test@P00",
            parent_chunk_id=None,
            part_index=0,
            part_total=1,
            text="Texto.",
            retrieval_text="Texto.",
            parent_text=None,
            document_version="v1.0.0-sha256abc123",
        )

        assert chunk.document_version == "v1.0.0-sha256abc123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
