"""Testes unitarios para o ArticleValidator."""

import pytest
from src.ingestion.article_validator import ArticleValidator


class MockChunk:
    """Mock de ProcessedChunk para testes."""
    def __init__(self, span_id: str):
        self.span_id = span_id


class TestArticleValidator:
    """Testes do ArticleValidator."""

    def test_validate_disabled(self):
        """Deve processar chunks mesmo se validacao desabilitada (para manifesto)."""
        validator = ArticleValidator(validate_enabled=False)
        chunks = [MockChunk("ART-001"), MockChunk("ART-002")]
        result = validator.validate(chunks)

        assert result.validate_enabled == False
        # Mesmo desabilitado, processa chunks e calcula status
        assert result.status == "passed"
        assert result.chunks_manifest == ["ART-001", "ART-002"]

    def test_validate_all_articles_found(self):
        """Deve passar se todos os artigos esperados foram encontrados."""
        validator = ArticleValidator(
            validate_enabled=True,
            expected_first=1,
            expected_last=3
        )
        chunks = [
            MockChunk("ART-001"),
            MockChunk("PAR-001-1"),  # Paragrafo (ignorado)
            MockChunk("ART-002"),
            MockChunk("INC-002-I"),  # Inciso (ignorado)
            MockChunk("ART-003"),
        ]
        result = validator.validate(chunks)

        assert result.status == "passed"
        assert result.found_articles == ["1", "2", "3"]
        assert result.missing_articles == []
        assert result.coverage_percent == 100.0

    def test_validate_missing_articles(self):
        """Deve reportar artigos faltantes."""
        validator = ArticleValidator(
            validate_enabled=True,
            expected_first=1,
            expected_last=5
        )
        chunks = [
            MockChunk("ART-001"),
            MockChunk("ART-002"),
            # ART-003 faltando
            MockChunk("ART-004"),
            # ART-005 faltando
        ]
        result = validator.validate(chunks)

        assert result.status == "failed"  # coverage < 95%
        assert result.found_articles == ["1", "2", "4"]
        assert result.missing_articles == ["3", "5"]
        assert result.has_gaps == True

    def test_validate_split_articles(self):
        """Deve detectar artigos splitados corretamente."""
        validator = ArticleValidator(
            validate_enabled=True,
            expected_first=1,
            expected_last=3
        )
        chunks = [
            MockChunk("ART-001"),
            MockChunk("ART-002-P1"),  # Art 2 splitado em 3 partes
            MockChunk("ART-002-P2"),
            MockChunk("ART-002-P3"),
            MockChunk("ART-003"),
        ]
        result = validator.validate(chunks)

        assert result.status == "passed"
        assert "2" in result.found_articles
        assert len(result.split_articles) == 1
        assert result.split_articles[0].article_number == "2"
        assert result.split_articles[0].parts_count == 3

    def test_validate_duplicate_articles(self):
        """Deve detectar artigos duplicados."""
        validator = ArticleValidator(
            validate_enabled=True,
            expected_first=1,
            expected_last=2
        )
        chunks = [
            MockChunk("ART-001"),
            MockChunk("ART-001"),  # Duplicado
            MockChunk("ART-002"),
        ]
        result = validator.validate(chunks)

        assert result.has_duplicates == True
        assert "1" in result.duplicate_articles

    def test_validate_chunks_manifest(self):
        """Deve gerar manifesto de chunks corretamente."""
        validator = ArticleValidator(
            validate_enabled=True,
            expected_first=1,
            expected_last=2
        )
        chunks = [
            MockChunk("ART-001"),
            MockChunk("PAR-001-1"),
            MockChunk("INC-001-I"),
            MockChunk("ART-002"),
        ]
        result = validator.validate(chunks)

        assert result.total_chunks_generated == 4
        assert result.chunks_manifest == ["ART-001", "PAR-001-1", "INC-001-I", "ART-002"]

    def test_to_dict(self):
        """Deve serializar resultado para dict corretamente."""
        validator = ArticleValidator(
            validate_enabled=True,
            expected_first=1,
            expected_last=2
        )
        chunks = [MockChunk("ART-001"), MockChunk("ART-002")]
        result = validator.validate(chunks)

        d = result.to_dict()

        assert isinstance(d, dict)
        assert "status" in d
        assert "found_articles" in d
        assert "missing_articles" in d
        assert "split_articles" in d
        assert "chunks_manifest" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
