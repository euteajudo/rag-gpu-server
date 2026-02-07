# -*- coding: utf-8 -*-
"""
PR13 Acceptance Tests (A–E) — Contract Alignment.

Validates that the VLM pipeline satisfies PR13 contracts:
- A: validate_chunk_invariants() gate aborts on violations
- B: extract_snippet_by_offsets() pure slicing when hash matches
- C: resolve_child_offsets() hierarchy containment
- D: _resolve_vlm_offsets() resolves all evidence sentinel offsets
- E: RunPod→VPS contract: required fields + snippet round-trip
"""

import pytest
import sys
sys.path.insert(0, '/workspace/rag-gpu-server')

from src.ingestion.pipeline import (
    validate_chunk_invariants,
    ContractViolationError,
    IngestionPipeline,
    PipelineResult,
)
from src.ingestion.models import IngestRequest, IngestStatus, ProcessedChunk
from src.chunking.canonical_offsets import (
    extract_snippet_by_offsets,
    resolve_child_offsets,
    OffsetResolutionError,
)
from src.utils.canonical_utils import (
    normalize_canonical_text,
    compute_canonical_hash,
)
from src.extraction.vlm_models import (
    BlockData,
    PageData,
    DeviceExtraction,
    PageExtraction,
    DocumentExtraction,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_chunk(
    span_id: str,
    device_type: str,
    canonical_start: int = -1,
    canonical_end: int = -1,
    canonical_hash: str = "",
    **overrides,
) -> ProcessedChunk:
    """Create a minimal ProcessedChunk for testing."""
    defaults = dict(
        node_id=f"leis:DOC#{span_id}",
        chunk_id=f"DOC#{span_id}",
        span_id=span_id,
        device_type=device_type,
        chunk_level="article" if device_type == "article" else "device",
        text="Texto de teste do dispositivo.",
        parent_text="",
        retrieval_text="Texto de teste do dispositivo.",
        document_id="DOC",
        tipo_documento="LEI",
        numero="1",
        ano=2021,
        parent_node_id="" if device_type == "article" else f"leis:DOC#ART-001",
        canonical_start=canonical_start,
        canonical_end=canonical_end,
        canonical_hash=canonical_hash,
    )
    defaults.update(overrides)
    return ProcessedChunk(**defaults)


VALID_HASH = compute_canonical_hash(normalize_canonical_text("dummy text\n"))


# =============================================================================
# Test A — validate_chunk_invariants() gate
# =============================================================================

class TestA_ValidateChunkInvariants:
    """validate_chunk_invariants() aborts entire document on violations."""

    def test_A1_sentinel_evidence_aborts(self):
        """A single article with sentinel offsets must abort."""
        chunk = _make_chunk("ART-001", "article", -1, -1, "")
        with pytest.raises(ContractViolationError, match="EVIDENCE SEM OFFSET"):
            validate_chunk_invariants([chunk], "DOC")

    def test_A2_valid_offsets_passes(self):
        """A chunk with valid offsets should pass without error."""
        chunk = _make_chunk(
            "ART-001", "article",
            canonical_start=100,
            canonical_end=200,
            canonical_hash=VALID_HASH,
        )
        # Must not raise
        validate_chunk_invariants([chunk], "DOC")

    def test_A3_one_sentinel_among_valid_aborts_all(self):
        """If any evidence chunk has sentinel offsets, ALL are aborted."""
        c1 = _make_chunk(
            "ART-001", "article",
            canonical_start=0,
            canonical_end=100,
            canonical_hash=VALID_HASH,
        )
        c2 = _make_chunk(
            "INC-001-1", "inciso",
            canonical_start=-1,
            canonical_end=-1,
            canonical_hash="",
        )
        with pytest.raises(ContractViolationError):
            validate_chunk_invariants([c1, c2], "DOC")

    def test_A4_incoherent_trio_hash_empty_offsets_valid(self):
        """Trio incoherent: valid start/end but empty hash."""
        chunk = _make_chunk(
            "ART-001", "article",
            canonical_start=100,
            canonical_end=200,
            canonical_hash="",
        )
        with pytest.raises(ContractViolationError, match="trio incoerente"):
            validate_chunk_invariants([chunk], "DOC")

    def test_A5_incoherent_trio_sentinel_offsets_with_hash(self):
        """Trio incoherent: sentinel offsets (-1,-1) but non-empty hash."""
        chunk = _make_chunk(
            "ART-001", "article",
            canonical_start=-1,
            canonical_end=-1,
            canonical_hash=VALID_HASH,
        )
        with pytest.raises(ContractViolationError, match="trio incoerente"):
            validate_chunk_invariants([chunk], "DOC")

    def test_A6_node_id_without_leis_prefix(self):
        """node_id without 'leis:' prefix must be rejected."""
        chunk = _make_chunk(
            "ART-001", "article",
            canonical_start=0,
            canonical_end=100,
            canonical_hash=VALID_HASH,
            node_id="DOC#ART-001",  # Missing leis: prefix
        )
        with pytest.raises(ContractViolationError, match="node_id sem prefixo"):
            validate_chunk_invariants([chunk], "DOC")

    def test_A7_child_without_parent_node_id(self):
        """A paragraph without parent_node_id must be rejected."""
        chunk = _make_chunk(
            "PAR-001-1", "paragraph",
            canonical_start=0,
            canonical_end=50,
            canonical_hash=VALID_HASH,
            parent_node_id="",  # Missing parent
        )
        with pytest.raises(ContractViolationError, match="sem parent_node_id"):
            validate_chunk_invariants([chunk], "DOC")

    def test_A8_node_id_with_at_P_rejected(self):
        """node_id containing @P is forbidden."""
        chunk = _make_chunk(
            "ART-001", "article",
            canonical_start=0,
            canonical_end=100,
            canonical_hash=VALID_HASH,
            node_id="leis:DOC#ART-001@P00",
        )
        with pytest.raises(ContractViolationError, match="@Pxx"):
            validate_chunk_invariants([chunk], "DOC")


# =============================================================================
# Test B — extract_snippet_by_offsets() pure slicing
# =============================================================================

class TestB_ExtractSnippetByOffsets:
    """extract_snippet_by_offsets returns (snippet, True) via pure slicing."""

    def test_B1_pure_slicing_when_hash_matches(self):
        """When hash matches and offsets are valid, returns pure slice."""
        canonical = normalize_canonical_text(
            "Art. 1\u00ba Exemplo de texto legal.\nArt. 2\u00ba Segundo artigo.\n"
        )
        hash_ = compute_canonical_hash(canonical)
        # Extract a known substring
        start = canonical.index("Exemplo")
        end = canonical.index("legal.") + len("legal.")
        snippet, used_offsets = extract_snippet_by_offsets(canonical, start, end, hash_)
        assert used_offsets is True
        assert snippet == canonical[start:end]

    def test_B2_hash_mismatch_returns_empty(self):
        """When hash doesn't match, returns empty fallback."""
        canonical = normalize_canonical_text("Art. 1\u00ba Exemplo.\n")
        snippet, used_offsets = extract_snippet_by_offsets(canonical, 0, 10, "wrong_hash")
        assert used_offsets is False
        assert snippet == ""

    def test_B3_sentinel_offsets_return_empty(self):
        """Sentinel offsets (-1, -1) return empty fallback."""
        canonical = normalize_canonical_text("Art. 1\u00ba Texto.\n")
        hash_ = compute_canonical_hash(canonical)
        snippet, used_offsets = extract_snippet_by_offsets(canonical, -1, -1, hash_)
        assert used_offsets is False
        assert snippet == ""

    def test_B4_empty_hash_returns_empty(self):
        """Empty stored_hash returns empty fallback."""
        canonical = normalize_canonical_text("Art. 1\u00ba Texto.\n")
        snippet, used_offsets = extract_snippet_by_offsets(canonical, 0, 10, "")
        assert used_offsets is False
        assert snippet == ""

    def test_B5_start_gte_end_returns_empty(self):
        """When start >= end, returns empty fallback."""
        canonical = normalize_canonical_text("Art. 1\u00ba Texto.\n")
        hash_ = compute_canonical_hash(canonical)
        snippet, used_offsets = extract_snippet_by_offsets(canonical, 10, 5, hash_)
        assert used_offsets is False
        assert snippet == ""


# =============================================================================
# Test C — resolve_child_offsets() hierarchy containment
# =============================================================================

class TestC_HierarchyContainment:
    """Children offsets within parent, no overlap between siblings."""

    def test_C1_children_within_parent(self):
        """Three distinct paragraphs resolved within article range."""
        canonical = "Art. 1\u00ba Caput do artigo. \u00a7 1\u00ba Primeiro par. \u00a7 2\u00ba Segundo par.\n"
        art_start, art_end = 0, len(canonical)

        p1_start, p1_end = resolve_child_offsets(
            canonical, art_start, art_end, "\u00a7 1\u00ba Primeiro par."
        )
        p2_start, p2_end = resolve_child_offsets(
            canonical, art_start, art_end, "\u00a7 2\u00ba Segundo par."
        )

        # Children within parent
        assert p1_start >= art_start and p1_end <= art_end
        assert p2_start >= art_start and p2_end <= art_end
        # Siblings ordered and non-overlapping
        assert p1_end <= p2_start

    def test_C2_child_not_found_raises_error(self):
        """Child text not present in parent raises NOT_FOUND."""
        canonical = "Art. 1\u00ba Texto simples do artigo.\n"
        with pytest.raises(OffsetResolutionError) as exc_info:
            resolve_child_offsets(
                canonical, 0, len(canonical),
                "Texto inexistente no artigo.",
                span_id="PAR-001-X",
            )
        assert exc_info.value.reason in ("NOT_FOUND", "NOT_FOUND_WHITESPACE_MISMATCH")

    def test_C3_ambiguous_child_raises_error(self):
        """Child appearing twice in parent raises AMBIGUOUS."""
        canonical = "Art. 1\u00ba A mesma frase. I - A mesma frase. II - A mesma frase.\n"
        with pytest.raises(OffsetResolutionError) as exc_info:
            resolve_child_offsets(
                canonical, 0, len(canonical),
                "A mesma frase.",
                span_id="INC-001-1",
            )
        assert "AMBIGUOUS" in exc_info.value.reason

    def test_C4_nested_hierarchy_art_inc_ali(self):
        """Nested hierarchy: alinea inside inciso inside article."""
        canonical = "Art. 1\u00ba Caput. I - primeiro inciso; a) al\u00ednea a; b) al\u00ednea b;\n"
        art_start, art_end = 0, len(canonical)

        inc_start, inc_end = resolve_child_offsets(
            canonical, art_start, art_end,
            "I - primeiro inciso; a) al\u00ednea a; b) al\u00ednea b;"
        )
        ali_start, ali_end = resolve_child_offsets(
            canonical, inc_start, inc_end,
            "a) al\u00ednea a;"
        )

        # Alinea inside inciso, inciso inside article
        assert inc_start >= art_start and inc_end <= art_end
        assert ali_start >= inc_start and ali_end <= inc_end

    def test_C5_siblings_ordered_no_overlap(self):
        """Siblings: incisos ordered and non-overlapping."""
        canonical = (
            "Art. 1\u00ba Caput do artigo.\n"
            "I - primeiro inciso;\n"
            "II - segundo inciso;\n"
            "III - terceiro inciso.\n"
        )
        art_start, art_end = 0, len(canonical)

        inc1_start, inc1_end = resolve_child_offsets(
            canonical, art_start, art_end, "I - primeiro inciso;"
        )
        inc2_start, inc2_end = resolve_child_offsets(
            canonical, art_start, art_end, "II - segundo inciso;"
        )
        inc3_start, inc3_end = resolve_child_offsets(
            canonical, art_start, art_end, "III - terceiro inciso."
        )

        # Ordered and non-overlapping
        assert inc1_end <= inc2_start
        assert inc2_end <= inc3_start


# =============================================================================
# Test D — _resolve_vlm_offsets() resolves sentinel offsets
# =============================================================================

class TestD_ResolveVlmOffsets:
    """After _resolve_vlm_offsets(), zero evidence chunks with sentinels."""

    @staticmethod
    def _build_extraction(canonical_text: str, devices_per_page: dict):
        """
        Build a DocumentExtraction + pages_data with proper block offsets.

        Args:
            canonical_text: the full canonical text (normalized)
            devices_per_page: {page_num: [(device_type, identifier, parent_id, text, bbox_norm), ...]}

        Returns:
            (DocumentExtraction, canonical_hash)
        """
        canonical_hash = compute_canonical_hash(canonical_text)
        pages = []
        pages_data_list = []

        for page_num, devices in sorted(devices_per_page.items()):
            # Build devices
            dev_list = []
            for dtype, ident, parent_ident, text, bbox in devices:
                dev_list.append(DeviceExtraction(
                    device_type=dtype,
                    identifier=ident,
                    text=text,
                    parent_identifier=parent_ident,
                    bbox=bbox,
                    confidence=0.95,
                ))

            pages.append(PageExtraction(
                page_number=page_num,
                devices=dev_list,
            ))

            # Build PageData with blocks covering the whole canonical text
            # For simplicity, one big block per page covering the full text
            page_text_start = canonical_text.find(
                next((d[3] for d in devices if d[3].strip()), ""), 0
            )
            if page_text_start < 0:
                page_text_start = 0

            # Find the end of the last device text on this page
            page_text_end = page_text_start
            for d in devices:
                pos = canonical_text.find(d[3].strip(), page_text_start)
                if pos >= 0:
                    end = pos + len(d[3].strip())
                    if end > page_text_end:
                        page_text_end = end

            if page_text_end <= page_text_start:
                page_text_end = len(canonical_text)

            # Create block that covers the device's text region
            block = BlockData(
                block_index=0,
                char_start=page_text_start,
                char_end=page_text_end,
                bbox_pdf=[50.0, 50.0, 500.0, 700.0],
                text=canonical_text[page_text_start:page_text_end],
                page_number=page_num,
            )

            pd = PageData(
                page_number=page_num,
                image_png=b"",
                image_base64="",
                text=canonical_text[page_text_start:page_text_end],
                width=612.0,
                height=792.0,
                img_width=1224,
                img_height=1584,
                blocks=[block],
                char_start=page_text_start,
                char_end=page_text_end,
            )
            pages_data_list.append(pd)

        extraction = DocumentExtraction(
            document_id="DOC-TEST",
            pages=pages,
            canonical_text=canonical_text,
            canonical_hash=canonical_hash,
            total_devices=sum(len(d) for d in devices_per_page.values()),
            pages_data=pages_data_list,
        )

        return extraction, canonical_hash

    def test_D1_bbox_matching_resolves_all(self):
        """Devices with matching bboxes + blocks get resolved via Phase A."""
        canonical = normalize_canonical_text(
            "Art. 1\u00ba O servidor p\u00fablico fica obrigado.\n"
            "\u00a7 1\u00ba O prazo \u00e9 de trinta dias.\n"
            "\u00a7 2\u00ba A notifica\u00e7\u00e3o ser\u00e1 pessoal.\n"
        )

        # All devices on page 1 with bbox overlapping the single block
        devices = {
            1: [
                ("artigo", "Art. 1\u00ba", "", "Art. 1\u00ba O servidor p\u00fablico fica obrigado.",
                 [0.05, 0.05, 0.90, 0.35]),
                ("paragrafo", "\u00a7 1\u00ba", "Art. 1\u00ba", "\u00a7 1\u00ba O prazo \u00e9 de trinta dias.",
                 [0.05, 0.35, 0.90, 0.60]),
                ("paragrafo", "\u00a7 2\u00ba", "Art. 1\u00ba", "\u00a7 2\u00ba A notifica\u00e7\u00e3o ser\u00e1 pessoal.",
                 [0.05, 0.60, 0.90, 0.85]),
            ]
        }

        extraction, canonical_hash = self._build_extraction(canonical, devices)
        request = IngestRequest(
            document_id="DOC-TEST",
            tipo_documento="LEI",
            numero="1",
            ano=2021,
        )
        result = PipelineResult(status=IngestStatus.PROCESSING, document_id="DOC-TEST")

        pipeline = IngestionPipeline()
        chunks = pipeline._vlm_to_processed_chunks(extraction, request, result)

        EVIDENCE = {"article", "paragraph", "inciso", "alinea"}
        for chunk in chunks:
            if chunk.device_type in EVIDENCE:
                assert chunk.canonical_start >= 0, \
                    f"{chunk.span_id} has sentinel start={chunk.canonical_start}"
                assert chunk.canonical_end > chunk.canonical_start, \
                    f"{chunk.span_id} has sentinel end={chunk.canonical_end}"
                assert chunk.canonical_hash != "", \
                    f"{chunk.span_id} has empty canonical_hash"

    def test_D2_find_fallback_resolves_without_bbox(self):
        """Devices without bbox get resolved via Phase B (find)."""
        canonical = normalize_canonical_text(
            "Art. 2\u00ba As despesas ser\u00e3o cobertas.\n"
        )

        # Device with empty bbox
        devices = {
            1: [
                ("artigo", "Art. 2\u00ba", "",
                 "Art. 2\u00ba As despesas ser\u00e3o cobertas.",
                 []),  # No bbox
            ]
        }

        extraction, canonical_hash = self._build_extraction(canonical, devices)
        request = IngestRequest(
            document_id="DOC-TEST",
            tipo_documento="LEI",
            numero="2",
            ano=2021,
        )
        result = PipelineResult(status=IngestStatus.PROCESSING, document_id="DOC-TEST")

        pipeline = IngestionPipeline()
        chunks = pipeline._vlm_to_processed_chunks(extraction, request, result)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.canonical_start >= 0, "Should be resolved via find()"
        assert chunk.canonical_end > chunk.canonical_start
        assert chunk.canonical_hash == canonical_hash

    def test_D3_child_resolved_via_parent_range(self):
        """Child without bbox/find-match resolved via Phase C (parent range)."""
        # Construct canonical where the inciso text is only findable within the article
        canonical = normalize_canonical_text(
            "Art. 1\u00ba O artigo que trata das obriga\u00e7\u00f5es.\n"
            "I - primeira obriga\u00e7\u00e3o espec\u00edfica;\n"
            "II - segunda obriga\u00e7\u00e3o espec\u00edfica;\n"
        )

        devices = {
            1: [
                ("artigo", "Art. 1\u00ba", "",
                 "Art. 1\u00ba O artigo que trata das obriga\u00e7\u00f5es.\n"
                 "I - primeira obriga\u00e7\u00e3o espec\u00edfica;\n"
                 "II - segunda obriga\u00e7\u00e3o espec\u00edfica;",
                 [0.05, 0.05, 0.95, 0.95]),
                ("inciso", "I", "Art. 1\u00ba",
                 "I - primeira obriga\u00e7\u00e3o espec\u00edfica;",
                 []),  # No bbox — will need find or parent resolve
                ("inciso", "II", "Art. 1\u00ba",
                 "II - segunda obriga\u00e7\u00e3o espec\u00edfica;",
                 []),  # No bbox
            ]
        }

        extraction, canonical_hash = self._build_extraction(canonical, devices)
        request = IngestRequest(
            document_id="DOC-TEST",
            tipo_documento="LEI",
            numero="1",
            ano=2021,
        )
        result = PipelineResult(status=IngestStatus.PROCESSING, document_id="DOC-TEST")

        pipeline = IngestionPipeline()
        chunks = pipeline._vlm_to_processed_chunks(extraction, request, result)

        EVIDENCE = {"article", "paragraph", "inciso", "alinea"}
        for chunk in chunks:
            if chunk.device_type in EVIDENCE:
                assert chunk.canonical_start >= 0, \
                    f"{chunk.span_id} still sentinel after resolution"
                assert chunk.canonical_end > chunk.canonical_start
                assert chunk.canonical_hash == canonical_hash


# =============================================================================
# Test E — RunPod→VPS contract: required fields + snippet round-trip
# =============================================================================

class TestE_ContractRunPodToVPS:
    """Chunks must pass gate and contain all required fields."""

    @staticmethod
    def _build_complete_chunks():
        """Build chunks that mimic real pipeline output."""
        canonical = normalize_canonical_text(
            "Art. 1\u00ba Esta Lei estabelece normas gerais.\n"
            "\u00a7 1\u00ba O \u00f3rg\u00e3o competente definir\u00e1 os procedimentos.\n"
            "I - primeiro inciso do par\u00e1grafo;\n"
        )
        canonical_hash = compute_canonical_hash(canonical)

        art_text = "Art. 1\u00ba Esta Lei estabelece normas gerais."
        par_text = "\u00a7 1\u00ba O \u00f3rg\u00e3o competente definir\u00e1 os procedimentos."
        inc_text = "I - primeiro inciso do par\u00e1grafo;"

        art_start = canonical.index(art_text)
        art_end = art_start + len(art_text)
        par_start = canonical.index(par_text)
        par_end = par_start + len(par_text)
        inc_start = canonical.index(inc_text)
        inc_end = inc_start + len(inc_text)

        chunks = [
            _make_chunk(
                "ART-001", "article",
                canonical_start=art_start,
                canonical_end=art_end,
                canonical_hash=canonical_hash,
                text=art_text,
                retrieval_text=art_text,
                article_number="1",
                page_number=1,
                bbox=[50.0, 50.0, 500.0, 200.0],
                parent_node_id="",
            ),
            _make_chunk(
                "PAR-001-1", "paragraph",
                canonical_start=par_start,
                canonical_end=par_end,
                canonical_hash=canonical_hash,
                text=par_text,
                retrieval_text=par_text,
                article_number="1",
                page_number=1,
                bbox=[50.0, 200.0, 500.0, 350.0],
                parent_node_id="leis:DOC#ART-001",
            ),
            _make_chunk(
                "INC-001-1", "inciso",
                canonical_start=inc_start,
                canonical_end=inc_end,
                canonical_hash=canonical_hash,
                text=inc_text,
                retrieval_text=inc_text,
                article_number="1",
                page_number=1,
                bbox=[50.0, 350.0, 500.0, 450.0],
                parent_node_id="leis:DOC#ART-001",
            ),
        ]
        return chunks, canonical, canonical_hash

    def test_E1_gate_passes(self):
        """All chunks pass validate_chunk_invariants without error."""
        chunks, _, _ = self._build_complete_chunks()
        # Must not raise
        validate_chunk_invariants(chunks, "DOC")

    def test_E2_node_id_format(self):
        """node_id starts with 'leis:' and does NOT contain '@P'."""
        chunks, _, _ = self._build_complete_chunks()
        for chunk in chunks:
            assert chunk.node_id.startswith("leis:"), \
                f"node_id missing prefix: {chunk.node_id}"
            assert "@P" not in chunk.node_id, \
                f"node_id contains @P: {chunk.node_id}"

    def test_E3_required_fields_present(self):
        """All direct fields needed for Milvus leis_v4 are present in model_dump."""
        REQUIRED_FIELDS = {
            "node_id", "span_id", "parent_node_id", "device_type", "chunk_level",
            "chunk_id", "text", "retrieval_text", "document_id", "tipo_documento",
            "numero", "ano", "article_number", "aliases", "canonical_start",
            "canonical_end", "canonical_hash",
            "origin_type", "origin_reference", "origin_reference_name",
            "is_external_material", "origin_confidence", "origin_reason",
            "page_number", "bbox", "citations",
        }

        chunks, _, _ = self._build_complete_chunks()
        for chunk in chunks:
            d = chunk.model_dump()
            for field_name in REQUIRED_FIELDS:
                assert field_name in d, \
                    f"Field '{field_name}' missing from model_dump() of {chunk.span_id}"

    def test_E4_derivable_fields(self):
        """citations is a list so VPS can derive has_citations/citations_count."""
        chunks, _, _ = self._build_complete_chunks()
        for chunk in chunks:
            d = chunk.model_dump()
            assert isinstance(d["citations"], list), \
                f"citations should be list, got {type(d['citations'])}"
            # VPS can derive:
            has_citations = len(d["citations"]) > 0
            citations_count = len(d["citations"])
            assert isinstance(has_citations, bool)
            assert isinstance(citations_count, int)

    def test_E5_snippet_round_trip(self):
        """For each chunk with valid offsets, snippet extraction returns exact text."""
        chunks, canonical, canonical_hash = self._build_complete_chunks()
        for chunk in chunks:
            if chunk.canonical_start >= 0:
                snippet, used = extract_snippet_by_offsets(
                    canonical,
                    chunk.canonical_start,
                    chunk.canonical_end,
                    chunk.canonical_hash,
                )
                assert used is True, \
                    f"Snippet fallback for {chunk.span_id}"
                assert snippet == chunk.text, \
                    f"Snippet mismatch for {chunk.span_id}: " \
                    f"got '{snippet[:40]}...' expected '{chunk.text[:40]}...'"

    def test_E6_evidence_offsets_coherent(self):
        """All evidence chunks have coherent PR13 trio (not sentinel)."""
        chunks, _, _ = self._build_complete_chunks()
        EVIDENCE = {"article", "paragraph", "inciso", "alinea"}
        for chunk in chunks:
            if chunk.device_type in EVIDENCE:
                assert chunk.canonical_start >= 0, \
                    f"{chunk.span_id}: start is sentinel"
                assert chunk.canonical_end > chunk.canonical_start, \
                    f"{chunk.span_id}: end <= start"
                assert chunk.canonical_hash != "", \
                    f"{chunk.span_id}: empty hash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
