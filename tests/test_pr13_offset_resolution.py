# -*- coding: utf-8 -*-
"""
Tests for PR13 STRICT offset resolution.

Validates deterministic offset resolution for child chunks.
"""

import pytest
import sys
sys.path.insert(0, '/workspace/rag-gpu-server')

from src.chunking.canonical_offsets import (
    resolve_child_offsets,
    OffsetResolutionError,
    normalize_canonical_text,
    compute_canonical_hash,
)


class TestResolveChildOffsets:
    """Tests for resolve_child_offsets function."""

    def test_exact_match_single_occurrence(self):
        """Child text found exactly once within parent range."""
        canonical_text = "Art. 1º O texto do artigo completo.\n§ 1º O parágrafo primeiro.\n§ 2º O parágrafo segundo."
        parent_start = 0
        parent_end = len(canonical_text)
        chunk_text = "§ 1º O parágrafo primeiro."

        start, end = resolve_child_offsets(
            canonical_text=canonical_text,
            parent_start=parent_start,
            parent_end=parent_end,
            chunk_text=chunk_text,
        )

        # Verify offsets point to exact text
        assert canonical_text[start:end] == chunk_text.strip()
        assert start > 0  # Not at beginning
        assert end > start

    def test_not_found_raises_error(self):
        """Child text not found raises OffsetResolutionError."""
        canonical_text = "Art. 1º O texto do artigo completo.\n§ 1º O parágrafo primeiro."
        parent_start = 0
        parent_end = len(canonical_text)
        chunk_text = "Texto que não existe no documento."

        with pytest.raises(OffsetResolutionError) as exc_info:
            resolve_child_offsets(
                canonical_text=canonical_text,
                parent_start=parent_start,
                parent_end=parent_end,
                chunk_text=chunk_text,
                span_id="PAR-001-X",
            )

        assert "NOT_FOUND" in exc_info.value.reason
        assert "PAR-001-X" in str(exc_info.value)

    def test_ambiguous_multiple_matches_raises_error(self):
        """Multiple occurrences of chunk text raises OffsetResolutionError."""
        canonical_text = "Art. 1º A contratação.\nI - A contratação.\nII - A contratação."
        parent_start = 0
        parent_end = len(canonical_text)
        chunk_text = "A contratação."

        with pytest.raises(OffsetResolutionError) as exc_info:
            resolve_child_offsets(
                canonical_text=canonical_text,
                parent_start=parent_start,
                parent_end=parent_end,
                chunk_text=chunk_text,
                span_id="INC-001-I",
            )

        assert "AMBIGUOUS" in exc_info.value.reason

    def test_empty_chunk_text_raises_error(self):
        """Empty chunk text raises OffsetResolutionError."""
        canonical_text = "Art. 1º Texto qualquer."
        parent_start = 0
        parent_end = len(canonical_text)

        with pytest.raises(OffsetResolutionError) as exc_info:
            resolve_child_offsets(
                canonical_text=canonical_text,
                parent_start=parent_start,
                parent_end=parent_end,
                chunk_text="   ",  # whitespace only
                span_id="PAR-001-1",
            )

        assert "EMPTY_TEXT" in exc_info.value.reason

    def test_search_within_parent_range_only(self):
        """Search only within parent's range, not entire document."""
        canonical_text = "Art. 1º Texto único.\nArt. 2º Texto único.\nArt. 3º Mais texto."
        # Simulate article 2 range
        art2_start = canonical_text.index("Art. 2º")
        art2_end = canonical_text.index("Art. 3º")
        chunk_text = "Texto único."

        # Should find exactly one occurrence within Art. 2 range
        start, end = resolve_child_offsets(
            canonical_text=canonical_text,
            parent_start=art2_start,
            parent_end=art2_end,
            chunk_text=chunk_text,
        )

        # Result should be within Art. 2 range
        assert start >= art2_start
        assert end <= art2_end
        assert canonical_text[start:end] == chunk_text.strip()

    def test_outside_parent_range_not_found(self):
        """Text that exists outside parent range is not found."""
        canonical_text = "Art. 1º Texto do artigo um.\nArt. 2º Texto do artigo dois."
        # Simulate article 2 range only
        art2_start = canonical_text.index("Art. 2º")
        art2_end = len(canonical_text)
        chunk_text = "Texto do artigo um."  # Exists in Art. 1, not Art. 2

        with pytest.raises(OffsetResolutionError) as exc_info:
            resolve_child_offsets(
                canonical_text=canonical_text,
                parent_start=art2_start,
                parent_end=art2_end,
                chunk_text=chunk_text,
            )

        assert "NOT_FOUND" in exc_info.value.reason

    def test_absolute_offset_calculation(self):
        """Returned offsets are absolute (relative to canonical_text start)."""
        canonical_text = "AAAA" + "Art. 1º Texto.\n§ 1º Parágrafo." + "BBBB"
        art_start = 4  # After "AAAA"
        art_end = 4 + len("Art. 1º Texto.\n§ 1º Parágrafo.")
        chunk_text = "§ 1º Parágrafo."

        start, end = resolve_child_offsets(
            canonical_text=canonical_text,
            parent_start=art_start,
            parent_end=art_end,
            chunk_text=chunk_text,
        )

        # Absolute offset should work with canonical_text directly
        assert canonical_text[start:end] == chunk_text.strip()

    def test_whitespace_stripped_from_search(self):
        """Leading/trailing whitespace is stripped from chunk_text before search."""
        canonical_text = "Art. 1º Texto.\n§ 1º Parágrafo com conteúdo."
        parent_start = 0
        parent_end = len(canonical_text)
        chunk_text = "  \n  § 1º Parágrafo com conteúdo.  \t  "  # Extra whitespace

        start, end = resolve_child_offsets(
            canonical_text=canonical_text,
            parent_start=parent_start,
            parent_end=parent_end,
            chunk_text=chunk_text,
        )

        assert canonical_text[start:end] == chunk_text.strip()


class TestNormalizeCanonicalText:
    """Tests for normalize_canonical_text function."""

    def test_empty_returns_empty(self):
        """Empty string returns empty."""
        assert normalize_canonical_text("") == ""

    def test_normalizes_crlf_to_lf(self):
        """Windows line endings are normalized to Unix."""
        text = "Line 1\r\nLine 2\r\n"
        result = normalize_canonical_text(text)
        assert "\r" not in result
        assert result == "Line 1\nLine 2\n"

    def test_removes_trailing_whitespace(self):
        """Trailing whitespace removed from each line."""
        text = "Line 1   \nLine 2\t\t\n"
        result = normalize_canonical_text(text)
        assert result == "Line 1\nLine 2\n"

    def test_ensures_single_newline_at_end(self):
        """Exactly one newline at end of text."""
        text = "Content\n\n\n"
        result = normalize_canonical_text(text)
        assert result == "Content\n"
        assert result.endswith("\n")
        assert not result.endswith("\n\n")

    def test_unicode_nfc_normalization(self):
        """Unicode is NFC normalized."""
        # NFD form: 'é' as 'e' + combining accent
        nfd_text = "café"  # Could be NFD
        result = normalize_canonical_text(nfd_text)
        # After NFC, should be canonical form
        assert len(result.rstrip()) == 4  # 'c', 'a', 'f', 'é'


class TestComputeCanonicalHash:
    """Tests for compute_canonical_hash function."""

    def test_deterministic_hash(self):
        """Same text always produces same hash."""
        text = "Art. 1º O texto do artigo."
        hash1 = compute_canonical_hash(text)
        hash2 = compute_canonical_hash(text)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

    def test_different_text_different_hash(self):
        """Different text produces different hash."""
        hash1 = compute_canonical_hash("Texto A")
        hash2 = compute_canonical_hash("Texto B")
        assert hash1 != hash2


class TestOffsetResolutionErrorAttributes:
    """Tests for OffsetResolutionError exception attributes."""

    def test_attributes_preserved(self):
        """Error attributes are preserved."""
        error = OffsetResolutionError(
            message="Test error",
            document_id="DOC-001",
            span_id="PAR-001-1",
            device_type="paragraph",
            reason="TEST_REASON",
        )

        assert error.document_id == "DOC-001"
        assert error.span_id == "PAR-001-1"
        assert error.device_type == "paragraph"
        assert error.reason == "TEST_REASON"
        assert "DOC-001" in str(error)
        assert "PAR-001-1" in str(error)
        assert "TEST_REASON" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
