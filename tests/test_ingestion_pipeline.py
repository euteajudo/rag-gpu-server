"""
Testes do pipeline de ingestão PR3 v2.

PR3 v2 - Hard Reset RAG Architecture
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.canonical import (
    build_logical_node_id,
    build_node_id,
    build_chunk_id,
    build_parent_chunk_id,
    get_prefix_for_document_type,
    SCHEMA_VERSION,
)
from src.spans import (
    Span,
    ChunkPart,
    DeviceType,
    split_text_with_offsets,
    MAX_TEXT_CHARS,
    OVERLAP_CHARS,
)


class TestIDConventions:
    """Testes das convenções de ID."""

    def test_build_logical_node_id(self):
        """Test: logical_node_id format."""
        result = build_logical_node_id("leis", "LEI-14133-2021", "ART-005")
        assert result == "leis:LEI-14133-2021#ART-005"

    def test_build_node_id_part_zero(self):
        """Test: node_id com part_index=0."""
        logical_id = "leis:LEI-14133-2021#ART-005"
        result = build_node_id(logical_id, 0)
        assert result == "leis:LEI-14133-2021#ART-005@P00"

    def test_build_node_id_part_nonzero(self):
        """Test: node_id com part_index > 0."""
        logical_id = "leis:LEI-14133-2021#ART-005"
        result = build_node_id(logical_id, 5)
        assert result == "leis:LEI-14133-2021#ART-005@P05"

    def test_build_chunk_id(self):
        """Test: chunk_id format."""
        result = build_chunk_id("LEI-14133-2021", "ART-005", 0)
        assert result == "LEI-14133-2021#ART-005@P00"

    def test_build_parent_chunk_id(self):
        """Test: parent_chunk_id format."""
        result = build_parent_chunk_id("LEI-14133-2021", "ART-005")
        assert result == "LEI-14133-2021#ART-005@P00"

    def test_build_parent_chunk_id_none(self):
        """Test: parent_chunk_id is None for root."""
        result = build_parent_chunk_id("LEI-14133-2021", None)
        assert result is None

    def test_get_prefix_for_document_type(self):
        """Test: prefixos por tipo de documento."""
        assert get_prefix_for_document_type("LEI") == "leis"
        assert get_prefix_for_document_type("DECRETO") == "decretos"
        assert get_prefix_for_document_type("IN") == "ins"
        assert get_prefix_for_document_type("UNKNOWN") == "docs"


class TestSplitter:
    """Testes do splitter de texto."""

    def test_split_small_text(self):
        """Test: texto pequeno não é dividido."""
        text = "Este é um texto pequeno."
        result = split_text_with_offsets(text)
        assert len(result) == 1
        assert result[0][0] == text
        assert result[0][1] == 0
        assert result[0][2] == len(text)

    def test_split_large_text(self):
        """Test: texto grande é dividido com overlap."""
        # Cria texto maior que MAX_TEXT_CHARS
        text = "A" * (MAX_TEXT_CHARS + 1000)
        result = split_text_with_offsets(text)

        # Deve ter múltiplas partes
        assert len(result) > 1

        # Verifica overlap entre partes
        first_end = result[0][2]
        second_start = result[1][1]
        overlap = first_end - second_start
        assert overlap > 0  # Deve haver overlap

    def test_split_empty_text(self):
        """Test: texto vazio retorna lista vazia."""
        result = split_text_with_offsets("")
        assert result == []

    def test_split_offsets_continuous(self):
        """Test: offsets são contínuos considerando overlap."""
        text = "A" * (MAX_TEXT_CHARS * 2)
        result = split_text_with_offsets(text)

        # Verifica que cada parte tem offsets válidos
        for part_text, char_start, char_end in result:
            assert char_start >= 0
            assert char_end > char_start
            assert part_text == text[char_start:char_end]


class TestSpanTypes:
    """Testes dos tipos de span."""

    def test_span_creation(self):
        """Test: criação de Span."""
        span = Span(
            logical_node_id="leis:LEI-14133-2021#ART-005",
            document_id="LEI-14133-2021",
            span_id="ART-005",
            parent_span_id=None,
            device_type=DeviceType.ARTICLE,
            text="Art. 5º O estudo técnico preliminar...",
            article_number="5",
            document_type="LEI",
        )

        assert span.logical_node_id == "leis:LEI-14133-2021#ART-005"
        assert span.device_type == DeviceType.ARTICLE
        assert span.parent_span_id is None

    def test_span_validation_required_fields(self):
        """Test: validação de campos obrigatórios."""
        with pytest.raises(ValueError, match="logical_node_id"):
            Span(
                logical_node_id="",
                document_id="LEI-14133-2021",
                span_id="ART-005",
                parent_span_id=None,
                device_type=DeviceType.ARTICLE,
                text="Texto",
            )

    def test_chunk_part_creation(self):
        """Test: criação de ChunkPart."""
        part = ChunkPart(
            node_id="leis:LEI-14133-2021#ART-005@P00",
            logical_node_id="leis:LEI-14133-2021#ART-005",
            chunk_id="LEI-14133-2021#ART-005@P00",
            parent_chunk_id=None,
            part_index=0,
            part_total=1,
            text="Art. 5º O estudo técnico preliminar...",
            char_start=0,
            char_end=100,
            document_id="LEI-14133-2021",
            span_id="ART-005",
            device_type=DeviceType.ARTICLE,
        )

        assert part.is_split is False
        assert part.is_first_part is True
        assert part.is_last_part is True

    def test_chunk_part_is_split(self):
        """Test: propriedade is_split para múltiplas partes."""
        part = ChunkPart(
            node_id="leis:LEI-14133-2021#ART-005@P01",
            logical_node_id="leis:LEI-14133-2021#ART-005",
            chunk_id="LEI-14133-2021#ART-005@P01",
            parent_chunk_id=None,
            part_index=1,
            part_total=3,
            text="Continuação do artigo...",
            char_start=8000,
            char_end=16000,
            document_id="LEI-14133-2021",
            span_id="ART-005",
            device_type=DeviceType.ARTICLE,
        )

        assert part.is_split is True
        assert part.is_first_part is False
        assert part.is_last_part is False


class TestDeviceType:
    """Testes do enum DeviceType."""

    def test_device_types(self):
        """Test: todos os tipos de dispositivo."""
        assert DeviceType.ARTICLE.value == "article"
        assert DeviceType.PARAGRAPH.value == "paragraph"
        assert DeviceType.INCISO.value == "inciso"
        assert DeviceType.ALINEA.value == "alinea"
        assert DeviceType.CAPUT.value == "caput"
        assert DeviceType.EMENTA.value == "ementa"
        assert DeviceType.PREAMBULO.value == "preambulo"


class TestSchemaVersion:
    """Testes de versionamento."""

    def test_schema_version(self):
        """Test: versão do schema está definida."""
        assert SCHEMA_VERSION == "2.0.0"


# Happy path integration test (requires mocks)
class TestIngestionPipelineHappyPath:
    """Teste do happy path do pipeline."""

    @pytest.fixture
    def mock_services(self):
        """Fixture com mocks dos serviços."""
        with patch('src.orchestrator.ingestion_runner.Canonicalizer') as mock_canon, \
             patch('src.orchestrator.ingestion_runner.EmbeddingClient') as mock_embed, \
             patch('src.orchestrator.ingestion_runner.MilvusWriter') as mock_milvus, \
             patch('src.orchestrator.ingestion_runner.Neo4jEdgeWriter') as mock_neo4j, \
             patch('src.orchestrator.ingestion_runner.ObjectStorageClient') as mock_storage, \
             patch('src.orchestrator.ingestion_runner.DocumentRegistryService') as mock_registry:

            # Mock canonicalizer
            mock_canon_instance = MagicMock()
            mock_canon_instance.canonicalize.return_value = MagicMock(
                markdown="Art. 1º Teste.\n\nArt. 2º Teste 2.",
                sha256="abc123",
                page_count=1,
                char_count=100,
            )
            mock_canon.return_value = mock_canon_instance

            # Mock embedder
            mock_embed_instance = MagicMock()
            mock_embed_instance.embed_batch.return_value = [
                MagicMock(dense_vector=[0.1] * 1024, sparse_vector={1: 0.5}),
                MagicMock(dense_vector=[0.2] * 1024, sparse_vector={2: 0.5}),
            ]
            mock_embed.return_value = mock_embed_instance

            # Mock Milvus
            mock_milvus_instance = MagicMock()
            mock_milvus_instance.upsert_batch.return_value = 2
            mock_milvus.return_value = mock_milvus_instance

            # Mock Neo4j
            mock_neo4j_instance = MagicMock()
            mock_neo4j_instance.upsert_node.return_value = True
            mock_neo4j_instance.create_edges_batch.return_value = 1
            mock_neo4j.return_value = mock_neo4j_instance

            # Mock Storage
            mock_storage_instance = MagicMock()
            mock_storage_instance.put_source_pdf.return_value = "docs/test/source.pdf"
            mock_storage_instance.put_canonical_md.return_value = "docs/test/canonical.md"
            mock_storage_instance.put_manifest.return_value = "docs/test/manifest.json"
            mock_storage.return_value = mock_storage_instance

            # Mock Registry
            mock_registry_instance = MagicMock()
            mock_registry_instance.create_or_get_document.return_value = MagicMock(id="test-uuid")
            mock_registry.return_value = mock_registry_instance

            yield {
                'canonicalizer': mock_canon_instance,
                'embedder': mock_embed_instance,
                'milvus': mock_milvus_instance,
                'neo4j': mock_neo4j_instance,
                'storage': mock_storage_instance,
                'registry': mock_registry_instance,
            }

    def test_happy_path_returns_success(self, mock_services):
        """Test: pipeline completo retorna sucesso."""
        from src.orchestrator import IngestionRunner, IngestionConfig

        config = IngestionConfig(
            minio_access_key="test",
            minio_secret_key="test",
            neo4j_password="test",
        )
        runner = IngestionRunner(config)

        result = runner.run(
            pdf_bytes=b"fake pdf content",
            document_id="TEST-DOC-001",
            document_type="LEI",
        )

        assert result.success is True
        assert result.document_id == "TEST-DOC-001"
        assert result.ingest_run_id is not None
        assert result.manifest is not None
        assert result.manifest.status == "success"

    def test_happy_path_calls_all_services(self, mock_services):
        """Test: pipeline chama todos os serviços."""
        from src.orchestrator import IngestionRunner, IngestionConfig

        config = IngestionConfig(
            minio_access_key="test",
            minio_secret_key="test",
            neo4j_password="test",
        )
        runner = IngestionRunner(config)

        runner.run(
            pdf_bytes=b"fake pdf content",
            document_id="TEST-DOC-001",
            document_type="LEI",
        )

        # Verifica que todos os serviços foram chamados
        mock_services['canonicalizer'].canonicalize.assert_called_once()
        mock_services['embedder'].embed_batch.assert_called_once()
        mock_services['milvus'].upsert_batch.assert_called_once()
        mock_services['milvus'].flush.assert_called_once()
        mock_services['storage'].put_source_pdf.assert_called_once()
        mock_services['storage'].put_canonical_md.assert_called_once()
        mock_services['storage'].put_manifest.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
