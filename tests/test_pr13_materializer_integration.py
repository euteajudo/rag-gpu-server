# -*- coding: utf-8 -*-
"""
Integration tests for PR13 STRICT offset resolution in ChunkMaterializer.

Tests the complete flow from ArticleChunk to MaterializedChunk with valid offsets.
"""

import pytest
import sys
sys.path.insert(0, '/workspace/rag-gpu-server')

from dataclasses import dataclass, field
from typing import List, Optional


# Mock span models to avoid import issues
@dataclass
class MockSpan:
    span_id: str
    text: str
    parent_id: Optional[str] = None
    start_pos: int = -1
    end_pos: int = -1


@dataclass
class MockParsedDocument:
    spans: List[MockSpan] = field(default_factory=list)
    source_text: str = ""
    _span_map: dict = field(default_factory=dict)
    _children_map: dict = field(default_factory=dict)

    def add_span(self, span: MockSpan):
        self.spans.append(span)
        self._span_map[span.span_id] = span
        if span.parent_id:
            if span.parent_id not in self._children_map:
                self._children_map[span.parent_id] = []
            self._children_map[span.parent_id].append(span)

    def get_span(self, span_id: str) -> Optional[MockSpan]:
        return self._span_map.get(span_id)

    def get_children(self, span_id: str) -> List[MockSpan]:
        return self._children_map.get(span_id, [])


@dataclass
class MockArticleChunk:
    article_id: str
    article_number: str
    text: str
    paragrafo_ids: List[str] = field(default_factory=list)
    inciso_ids: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)


class TestChunkMaterializerOffsetResolution:
    """Tests for ChunkMaterializer with PR13 offset resolution."""

    def test_article_gets_offsets_from_offsets_map(self):
        """Article chunk receives offsets from offsets_map (SpanParser)."""
        from src.chunking.chunk_materializer import ChunkMaterializer, DeviceType

        canonical_text = "Art. 1º O texto completo do artigo.\n"

        materializer = ChunkMaterializer(
            document_id="TEST-001",
            offsets_map={"ART-001": (0, 35)},
            canonical_hash="abc123",
            canonical_text=canonical_text,
        )

        article_chunk = MockArticleChunk(
            article_id="ART-001",
            article_number="1",
            text="Art. 1º O texto completo do artigo.",
        )

        parsed_doc = MockParsedDocument(source_text=canonical_text)
        parsed_doc.add_span(MockSpan(
            span_id="ART-001",
            text="Art. 1º O texto completo do artigo.",
            start_pos=0,
            end_pos=35,
        ))

        chunks = materializer.materialize_article(article_chunk, parsed_doc, include_children=False)

        assert len(chunks) == 1
        article = chunks[0]
        assert article.canonical_start == 0
        assert article.canonical_end == 35
        assert article.canonical_hash == "abc123"
        assert article.device_type == DeviceType.ARTICLE

    def test_paragraph_offsets_resolved_from_canonical_text(self):
        """Paragraph chunk offsets resolved by searching within article range."""
        from src.chunking.chunk_materializer import ChunkMaterializer, DeviceType

        canonical_text = "Art. 1º O caput do artigo.\n§ 1º O parágrafo primeiro.\n"

        materializer = ChunkMaterializer(
            document_id="TEST-001",
            offsets_map={"ART-001": (0, 54)},  # Only article has offset
            canonical_hash="hash123",
            canonical_text=canonical_text,
        )

        article_chunk = MockArticleChunk(
            article_id="ART-001",
            article_number="1",
            text="Art. 1º O caput do artigo.\n§ 1º O parágrafo primeiro.",
            paragrafo_ids=["PAR-001-1"],
        )

        parsed_doc = MockParsedDocument(source_text=canonical_text)
        parsed_doc.add_span(MockSpan(
            span_id="ART-001",
            text="Art. 1º O caput do artigo.\n§ 1º O parágrafo primeiro.",
            start_pos=0,
            end_pos=54,
        ))
        parsed_doc.add_span(MockSpan(
            span_id="PAR-001-1",
            text="§ 1º O parágrafo primeiro.",
            parent_id="ART-001",
            # No start_pos/end_pos - must be resolved
        ))

        chunks = materializer.materialize_article(article_chunk, parsed_doc)

        assert len(chunks) == 2

        paragraph = [c for c in chunks if c.device_type == DeviceType.PARAGRAPH][0]

        # Paragraph should have resolved offsets
        assert paragraph.canonical_start >= 0
        assert paragraph.canonical_end > paragraph.canonical_start
        assert paragraph.canonical_hash == "hash123"

        # Verify offsets are correct by slicing
        assert canonical_text[paragraph.canonical_start:paragraph.canonical_end] == "§ 1º O parágrafo primeiro."

    def test_inciso_offsets_resolved_within_article_range(self):
        """Inciso chunk offsets resolved by searching within article range."""
        from src.chunking.chunk_materializer import ChunkMaterializer, DeviceType

        canonical_text = "Art. 1º O caput:\nI - primeiro inciso;\nII - segundo inciso.\n"

        materializer = ChunkMaterializer(
            document_id="TEST-001",
            offsets_map={"ART-001": (0, 59)},
            canonical_hash="hash123",
            canonical_text=canonical_text,
        )

        article_chunk = MockArticleChunk(
            article_id="ART-001",
            article_number="1",
            text="Art. 1º O caput:\nI - primeiro inciso;\nII - segundo inciso.",
            inciso_ids=["INC-001-I", "INC-001-II"],
        )

        parsed_doc = MockParsedDocument(source_text=canonical_text)
        parsed_doc.add_span(MockSpan(
            span_id="ART-001",
            text="Art. 1º O caput:\nI - primeiro inciso;\nII - segundo inciso.",
            start_pos=0,
            end_pos=59,
        ))
        parsed_doc.add_span(MockSpan(
            span_id="INC-001-I",
            text="I - primeiro inciso;",
            parent_id="ART-001",
        ))
        parsed_doc.add_span(MockSpan(
            span_id="INC-001-II",
            text="II - segundo inciso.",
            parent_id="ART-001",
        ))

        chunks = materializer.materialize_article(article_chunk, parsed_doc)

        incisos = [c for c in chunks if c.device_type == DeviceType.INCISO]
        assert len(incisos) == 2

        for inciso in incisos:
            assert inciso.canonical_start >= 0
            assert inciso.canonical_end > inciso.canonical_start
            assert inciso.canonical_hash == "hash123"
            # Verify the slicing works
            sliced = canonical_text[inciso.canonical_start:inciso.canonical_end]
            assert sliced in canonical_text

    def test_all_evidence_chunks_have_valid_offsets_not_sentinel(self):
        """All evidence chunks (article, paragraph, inciso) have valid offsets, not sentinel."""
        from src.chunking.chunk_materializer import ChunkMaterializer, DeviceType

        canonical_text = "Art. 1º O caput.\n§ 1º Parágrafo único.\nI - inciso um;\nII - inciso dois.\n"

        materializer = ChunkMaterializer(
            document_id="TEST-001",
            offsets_map={"ART-001": (0, 73)},
            canonical_hash="hash_test",
            canonical_text=canonical_text,
        )

        article_chunk = MockArticleChunk(
            article_id="ART-001",
            article_number="1",
            text="Art. 1º O caput.\n§ 1º Parágrafo único.\nI - inciso um;\nII - inciso dois.",
            paragrafo_ids=["PAR-001-1"],
            inciso_ids=["INC-001-I", "INC-001-II"],
        )

        parsed_doc = MockParsedDocument(source_text=canonical_text)
        parsed_doc.add_span(MockSpan("ART-001", "Art. 1º O caput.\n§ 1º Parágrafo único.\nI - inciso um;\nII - inciso dois.", start_pos=0, end_pos=73))
        parsed_doc.add_span(MockSpan("PAR-001-1", "§ 1º Parágrafo único.", parent_id="ART-001"))
        parsed_doc.add_span(MockSpan("INC-001-I", "I - inciso um;", parent_id="ART-001"))
        parsed_doc.add_span(MockSpan("INC-001-II", "II - inciso dois.", parent_id="ART-001"))

        chunks = materializer.materialize_article(article_chunk, parsed_doc)

        # All chunks should have valid offsets (PR13 STRICT: no sentinels)
        EVIDENCE_TYPES = {DeviceType.ARTICLE, DeviceType.PARAGRAPH, DeviceType.INCISO}

        for chunk in chunks:
            if chunk.device_type in EVIDENCE_TYPES:
                # Valid offsets: start >= 0, end > start, hash non-empty
                assert chunk.canonical_start >= 0, f"{chunk.span_id} has sentinel start"
                assert chunk.canonical_end > chunk.canonical_start, f"{chunk.span_id} has invalid end"
                assert chunk.canonical_hash != "", f"{chunk.span_id} has empty hash"

                # Verify slicing actually works
                sliced = canonical_text[chunk.canonical_start:chunk.canonical_end]
                assert len(sliced) > 0, f"{chunk.span_id} slices to empty string"

    def test_missing_canonical_text_raises_error(self):
        """Missing canonical_text raises OffsetResolutionError for child chunks."""
        from src.chunking.chunk_materializer import ChunkMaterializer
        from src.chunking.canonical_offsets import OffsetResolutionError

        materializer = ChunkMaterializer(
            document_id="TEST-001",
            offsets_map={"ART-001": (0, 50)},
            canonical_hash="hash123",
            canonical_text="",  # Empty!
        )

        article_chunk = MockArticleChunk(
            article_id="ART-001",
            article_number="1",
            text="Art. 1º O caput.\n§ 1º Parágrafo.",
            paragrafo_ids=["PAR-001-1"],
        )

        parsed_doc = MockParsedDocument()
        parsed_doc.add_span(MockSpan("ART-001", "Art. 1º O caput.\n§ 1º Parágrafo.", start_pos=0, end_pos=50))
        parsed_doc.add_span(MockSpan("PAR-001-1", "§ 1º Parágrafo.", parent_id="ART-001"))

        with pytest.raises(OffsetResolutionError) as exc_info:
            materializer.materialize_article(article_chunk, parsed_doc)

        assert "NO_CANONICAL_TEXT" in exc_info.value.reason

    def test_ambiguous_text_raises_error(self):
        """Ambiguous text (multiple occurrences) raises OffsetResolutionError."""
        from src.chunking.chunk_materializer import ChunkMaterializer
        from src.chunking.canonical_offsets import OffsetResolutionError

        canonical_text = "Art. 1º Repete.\n§ 1º Repete.\n§ 2º Repete.\n"

        materializer = ChunkMaterializer(
            document_id="TEST-001",
            offsets_map={"ART-001": (0, 42)},
            canonical_hash="hash123",
            canonical_text=canonical_text,
        )

        article_chunk = MockArticleChunk(
            article_id="ART-001",
            article_number="1",
            text="Art. 1º Repete.\n§ 1º Repete.\n§ 2º Repete.",
            paragrafo_ids=["PAR-001-1"],
        )

        parsed_doc = MockParsedDocument(source_text=canonical_text)
        parsed_doc.add_span(MockSpan("ART-001", "Art. 1º Repete.\n§ 1º Repete.\n§ 2º Repete.", start_pos=0, end_pos=42))
        # Paragraph text is ambiguous - "Repete." appears 3 times
        parsed_doc.add_span(MockSpan("PAR-001-1", "Repete.", parent_id="ART-001"))

        with pytest.raises(OffsetResolutionError) as exc_info:
            materializer.materialize_article(article_chunk, parsed_doc)

        assert "AMBIGUOUS" in exc_info.value.reason


class TestInvariantValidation:
    """Tests for PR13 trio coherence invariant."""

    def test_valid_trio_all_set(self):
        """Valid trio: start >= 0, end > start, hash non-empty."""
        from src.chunking.chunk_materializer import MaterializedChunk, DeviceType, ChunkLevel

        chunk = MaterializedChunk(
            node_id="leis:TEST-001#ART-001",
            chunk_id="TEST-001#ART-001",
            parent_node_id="",
            span_id="ART-001",
            device_type=DeviceType.ARTICLE,
            chunk_level=ChunkLevel.ARTICLE,
            text="Test",
            canonical_start=0,
            canonical_end=100,
            canonical_hash="abc123",
        )

        # Trio coherence: all valid
        assert chunk.canonical_start >= 0
        assert chunk.canonical_end > chunk.canonical_start
        assert chunk.canonical_hash != ""

    def test_sentinel_trio_all_defaults(self):
        """Sentinel trio: start == -1, end == -1, hash == ''."""
        from src.chunking.chunk_materializer import MaterializedChunk, DeviceType, ChunkLevel

        chunk = MaterializedChunk(
            node_id="leis:TEST-001#TOC",
            chunk_id="TEST-001#TOC",
            parent_node_id="",
            span_id="TOC",
            device_type=DeviceType.PART,  # Non-evidence type
            chunk_level=ChunkLevel.DEVICE,
            text="Test",
            # Defaults: canonical_start=-1, canonical_end=-1, canonical_hash=""
        )

        # Trio coherence: all sentinel
        assert chunk.canonical_start == -1
        assert chunk.canonical_end == -1
        assert chunk.canonical_hash == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
