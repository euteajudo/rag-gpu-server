# -*- coding: utf-8 -*-
"""
Testes: Bugs 1 e 2 do CitationExtractor — multi-ref com leis externas.

Bug 1: Plural "arts." seguido de lei externa perdia artigos.
Bug 2: Multi-ref "arts. 28, 29 e 33 da Lei X" não capturava todos os artigos.

@author: Equipe VectorGov
"""

import sys
from pathlib import Path

import importlib.util


def load_module(name: str, file_path: Path):
    """Carrega módulo diretamente do arquivo."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

citation_extractor_module = load_module(
    "citation_extractor",
    src_path / "chunking" / "citation_extractor.py",
)

CitationExtractor = citation_extractor_module.CitationExtractor
extract_citations_from_chunk = citation_extractor_module.extract_citations_from_chunk
normalize_citations = citation_extractor_module.normalize_citations


# =============================================================================
# Bug 1: "arts." plural com lei externa
# =============================================================================


class TestBug1ArtsPlural:
    """arts. (plural) antes de lei externa deve capturar todos os artigos."""

    def test_arts_plural_two_articles(self):
        """'arts. 5 e 10 da Lei 1234' deve gerar refs para ART-005 e ART-010."""
        ce = CitationExtractor(current_document_id="LEI-1234-2020")
        refs = ce.extract("conforme arts. 5 e 10 da Lei 1234")
        spans = {r.span_ref for r in refs if r.span_ref}
        assert "ART-005" in spans
        assert "ART-010" in spans

    def test_art_singular_still_works(self):
        """Regressão: 'art. 5 da Lei 1234' continua funcionando."""
        ce = CitationExtractor(current_document_id="LEI-1234-2020")
        refs = ce.extract("conforme art. 5 da Lei 1234")
        spans = {r.span_ref for r in refs if r.span_ref}
        assert "ART-005" in spans

    def test_artigos_plural_written_out(self):
        """'artigos 1 e 2 da Lei 14.133' captura ambos."""
        ce = CitationExtractor(current_document_id="LEI-14133-2021")
        refs = ce.extract("artigos 1 e 2 da Lei 14.133/2021")
        spans = {r.span_ref for r in refs if r.span_ref}
        assert "ART-001" in spans
        assert "ART-002" in spans

    def test_arts_plural_targets_external_doc(self):
        """Artigos devem apontar para o doc externo, não o current_document."""
        ce = CitationExtractor(current_document_id="IN-58-2022")
        refs = ce.extract("arts. 3 e 7 da Lei 14.133/2021")
        targets = {r.target_node_id for r in refs if r.target_node_id}
        # Lei 14.133 normaliza doc_id com ponto
        assert any("ART-003" in t for t in targets), f"ART-003 missing from {targets}"
        assert any("ART-007" in t for t in targets), f"ART-007 missing from {targets}"
        # Confirma que apontam para Lei 14.133, não para IN-58
        assert all("IN-58" not in t for t in targets if "ART-" in t)


# =============================================================================
# Bug 2: Multi-ref com vírgula/e/a + lei externa
# =============================================================================


class TestBug2MultiRefExternal:
    """Multi-ref (vírgula, 'e', range 'a') antes de lei externa."""

    def test_comma_and_e_separator(self):
        """'arts. 28, 29 e 33 da Lei Complementar nº 101' → 3 artigos."""
        ce = CitationExtractor(current_document_id="LC-101-2000")
        refs = ce.extract(
            "nos termos dos arts. 28, 29 e 33 da Lei Complementar no 101"
        )
        spans = {r.span_ref for r in refs if r.span_ref}
        assert "ART-028" in spans, f"ART-028 missing from {spans}"
        assert "ART-029" in spans, f"ART-029 missing from {spans}"
        assert "ART-033" in spans, f"ART-033 missing from {spans}"

    def test_range_with_external_law(self):
        """'arts. 62 a 70 da Lei 14.133/2021' → 9 artigos (62-70)."""
        ce = CitationExtractor(current_document_id="LEI-14133-2021")
        refs = ce.extract("arts. 62 a 70 da Lei 14.133/2021")
        spans = [r.span_ref for r in refs if r.span_ref and r.span_ref.startswith("ART-")]
        assert len(spans) == 9, f"Expected 9 articles, got {len(spans)}: {spans}"
        assert "ART-062" in spans
        assert "ART-070" in spans

    def test_range_internal_still_works(self):
        """Regressão: range interno 'arts. 62 a 70' sem lei externa."""
        ce = CitationExtractor(current_document_id="LEI-14133-2021")
        refs = ce.extract("arts. 62 a 70")
        spans = [r.span_ref for r in refs if r.span_ref and r.span_ref.startswith("ART-")]
        assert len(spans) == 9

    def test_multi_ref_doc_id_correct(self):
        """Todos os artigos do multi-ref apontam para o doc externo correto."""
        ce = CitationExtractor(current_document_id="IN-58-2022")
        refs = ce.extract("arts. 28, 29 e 33 da Lei Complementar no 101")
        for r in refs:
            if r.span_ref and r.span_ref.startswith("ART-"):
                assert r.doc_id == "LC-101-2000", f"{r.span_ref} has wrong doc_id: {r.doc_id}"
                assert r.target_node_id.startswith("leis:LC-101-2000#")

    def test_multi_ref_no_duplicates(self):
        """Multi-ref não gera referências duplicadas."""
        ce = CitationExtractor(current_document_id="LC-101-2000")
        refs = ce.extract("arts. 28, 29 e 33 da Lei Complementar no 101")
        targets = [r.target_node_id for r in refs if r.target_node_id]
        assert len(targets) == len(set(targets)), f"Duplicates found: {targets}"

    def test_comma_only_separator(self):
        """'arts. 1, 2, 3 do Decreto 10.024' → 3 artigos."""
        ce = CitationExtractor()
        refs = ce.extract("arts. 1, 2, 3 do Decreto 10.024/2019")
        spans = {r.span_ref for r in refs if r.span_ref and r.span_ref.startswith("ART-")}
        assert "ART-001" in spans
        assert "ART-002" in spans
        assert "ART-003" in spans


# =============================================================================
# Integração: multi-ref + normalize (self-loop filtering)
# =============================================================================


class TestMultiRefWithNormalize:
    """Multi-ref combinado com normalize_citations (remoção de self-loop)."""

    def test_multi_ref_self_loop_filtered(self):
        """Se chunk é ART-028 e multi-ref inclui art 28, ele é filtrado."""
        result = extract_citations_from_chunk(
            text="conforme arts. 28, 29 e 33 da Lei Complementar no 101",
            document_id="LC-101-2000",
            chunk_node_id="leis:LC-101-2000#ART-028",
            document_type="LC",
        )
        targets = [c["target_node_id"] for c in result]
        assert "leis:LC-101-2000#ART-028" not in targets, "Self-loop not filtered"
        assert "leis:LC-101-2000#ART-029" in targets
        assert "leis:LC-101-2000#ART-033" in targets

    def test_multi_ref_preserves_rel_type(self):
        """Multi-ref deve preservar rel_type e rel_type_confidence."""
        result = extract_citations_from_chunk(
            text="regulamenta os arts. 10 e 15 da Lei 14.133/2021",
            document_id="LEI-14133-2021",
            chunk_node_id="leis:LEI-14133-2021#ART-001",
            document_type="LEI",
        )
        for c in result:
            assert "rel_type" in c
            assert "rel_type_confidence" in c
            assert isinstance(c["rel_type"], str)
            assert isinstance(c["rel_type_confidence"], (int, float))
