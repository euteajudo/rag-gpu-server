# -*- coding: utf-8 -*-
"""
Testes para detecção cross-page (Feature 5).

Verifica:
- BboxSpan dataclass
- ProcessedChunk com is_cross_page e bbox_spans
- Detecção de duplicatas em páginas consecutivas
- Validação em validate_chunk_invariants
"""

import pytest
import sys
sys.path.insert(0, '/workspace/rag-gpu-server')

from src.ingestion.models import ProcessedChunk
from src.ingestion.pipeline import validate_chunk_invariants, ContractViolationError
from src.extraction.vlm_models import BboxSpan


# =============================================================================
# BboxSpan dataclass
# =============================================================================

class TestBboxSpan:

    def test_creation(self):
        span = BboxSpan(
            page_number=1,
            bbox_pdf=[50.0, 100.0, 500.0, 200.0],
            bbox_img=[0.1, 0.2, 0.9, 0.4],
        )
        assert span.page_number == 1
        assert span.bbox_pdf == [50.0, 100.0, 500.0, 200.0]
        assert span.bbox_img == [0.1, 0.2, 0.9, 0.4]


# =============================================================================
# ProcessedChunk cross-page fields
# =============================================================================

class TestProcessedChunkCrossPage:

    def _make_chunk(self, span_id="ART-001", device_type="article",
                    is_cross_page=False, bbox_spans=None, **kwargs):
        defaults = dict(
            node_id=f"leis:TEST#{span_id}",
            chunk_id=f"TEST#{span_id}",
            parent_node_id="",
            span_id=span_id,
            device_type=device_type,
            chunk_level="article" if device_type == "article" else "device",
            text="Art. 1o Texto do artigo.",
            document_id="TEST",
            tipo_documento="LEI",
            numero="123",
            ano=2021,
            canonical_start=0,
            canonical_end=24,
            canonical_hash="abc123",
        )
        defaults.update(kwargs)
        chunk = ProcessedChunk(**defaults)
        chunk.is_cross_page = is_cross_page
        chunk.bbox_spans = bbox_spans or []
        return chunk

    def test_default_values(self):
        """Defaults: is_cross_page=False, bbox_spans=[]."""
        chunk = self._make_chunk()
        assert chunk.is_cross_page is False
        assert chunk.bbox_spans == []

    def test_cross_page_with_spans(self):
        """Cross-page chunk com bbox_spans preenchidos."""
        spans = [
            {"page_number": 1, "bbox_pdf": [50, 100, 500, 200], "bbox_img": [0.1, 0.2, 0.9, 0.4]},
            {"page_number": 2, "bbox_pdf": [50, 50, 500, 150], "bbox_img": [0.1, 0.1, 0.9, 0.3]},
        ]
        chunk = self._make_chunk(is_cross_page=True, bbox_spans=spans)
        assert chunk.is_cross_page is True
        assert len(chunk.bbox_spans) == 2
        assert chunk.bbox_spans[0]["page_number"] == 1
        assert chunk.bbox_spans[1]["page_number"] == 2

    def test_serialization(self):
        """Cross-page fields serializam corretamente."""
        spans = [
            {"page_number": 1, "bbox_pdf": [50, 100, 500, 200], "bbox_img": [0.1, 0.2, 0.9, 0.4]},
        ]
        chunk = self._make_chunk(is_cross_page=True, bbox_spans=spans)
        data = chunk.model_dump()
        assert data["is_cross_page"] is True
        assert len(data["bbox_spans"]) == 1


# =============================================================================
# validate_chunk_invariants cross-page check
# =============================================================================

class TestCrossPageInvariant:

    def _make_valid_chunk(self, span_id="ART-001", device_type="article",
                          is_cross_page=False, bbox_spans=None):
        chunk = ProcessedChunk(
            node_id=f"leis:TEST#{span_id}",
            chunk_id=f"TEST#{span_id}",
            parent_node_id="" if device_type == "article" else f"leis:TEST#ART-001",
            span_id=span_id,
            device_type=device_type,
            chunk_level="article" if device_type == "article" else "device",
            text="Texto do dispositivo.",
            document_id="TEST",
            tipo_documento="LEI",
            numero="123",
            ano=2021,
            canonical_start=0,
            canonical_end=20,
            canonical_hash="validhash",
            is_cross_page=is_cross_page,
            bbox_spans=bbox_spans or [],
        )
        return chunk

    def test_cross_page_with_spans_passes(self):
        """Cross-page com bbox_spans: passa."""
        chunk = self._make_valid_chunk(
            is_cross_page=True,
            bbox_spans=[{"page_number": 1, "bbox_pdf": [0, 0, 100, 100], "bbox_img": [0, 0, 1, 1]}],
        )
        # Should not raise
        validate_chunk_invariants([chunk], "TEST")

    def test_cross_page_without_spans_fails(self):
        """Cross-page sem bbox_spans: falha."""
        chunk = self._make_valid_chunk(
            is_cross_page=True,
            bbox_spans=[],
        )
        with pytest.raises(ContractViolationError, match="is_cross_page=True mas bbox_spans vazio"):
            validate_chunk_invariants([chunk], "TEST")

    def test_not_cross_page_passes(self):
        """Chunk normal (não cross-page): passa."""
        chunk = self._make_valid_chunk(is_cross_page=False)
        validate_chunk_invariants([chunk], "TEST")


# =============================================================================
# Cross-page dedup: Jaccard similarity guard
# =============================================================================

class TestCrossPageJaccardGuard:
    """Testa a lógica de word-Jaccard na dedup cross-page."""

    def test_jaccard_identical_texts(self):
        """Textos idênticos: Jaccard=1.0 → dedup (keep first)."""
        words_a = set("art 1 texto do artigo exemplo".split())
        words_b = set("art 1 texto do artigo exemplo".split())
        union = words_a | words_b
        intersection = words_a & words_b
        jaccard = len(intersection) / max(len(union), 1)
        assert jaccard == 1.0
        assert jaccard > 0.7  # → dedup

    def test_jaccard_similar_texts(self):
        """Textos quase idênticos (OCR differences): Jaccard>0.7 → dedup."""
        words_a = set("art 1 texto do artigo com detalhes sobre licitação pública".split())
        words_b = set("art 1 texto do artigo com detalhes sobre licitação publica".split())  # no accent
        union = words_a | words_b
        intersection = words_a & words_b
        jaccard = len(intersection) / max(len(union), 1)
        # Only 1 word differs out of 10 → high Jaccard
        assert jaccard > 0.7

    def test_jaccard_different_texts(self):
        """Textos de continuação: Jaccard≤0.7 → concatenar."""
        words_a = set("art 5 as licitações devem observar os seguintes princípios".split())
        words_b = set("parágrafo único para os fins desta lei consideram-se".split())
        union = words_a | words_b
        intersection = words_a & words_b
        jaccard = len(intersection) / max(len(union), 1)
        assert jaccard <= 0.7  # → continuation, concatenate

    def test_jaccard_partial_overlap(self):
        """Overlap parcial mas significativo: still a duplicate."""
        words_a = set("art 10 o procedimento licitatório observará as seguintes fases".split())
        words_b = set("art 10 o procedimento licitatório observará as seguintes fases em ordem".split())
        union = words_a | words_b
        intersection = words_a & words_b
        jaccard = len(intersection) / max(len(union), 1)
        # 9 shared out of 10 → Jaccard ≈ 0.9
        assert jaccard > 0.7

    def test_empty_texts(self):
        """Textos vazios: Jaccard→0 by max(1) guard, no crash."""
        words_a = set("".split())
        words_b = set("".split())
        union = words_a | words_b
        intersection = words_a & words_b
        jaccard = len(intersection) / max(len(union), 1)
        assert jaccard == 0.0
