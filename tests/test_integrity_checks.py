# -*- coding: utf-8 -*-
"""
Testes unitários para os 5 novos checks do IntegrityValidator (Feature IV).

Checks testados:
1. bbox_text_coherence — Jaccard de palavras entre text_vlm e text_pymupdf
2. bbox_containment — Bbox do filho contida na do pai
3. offset_coverage — Proporção de matches com offset válido
4. hierarchy_consistency — Profundidade do tipo filho > pai
5. cross_page_continuity — Artigos em páginas consecutivas
"""

import pytest
import sys
sys.path.insert(0, '/workspace/rag-gpu-server')

from src.inspection.pipeline import InspectionPipeline
from src.inspection.models import (
    BBox,
    ReconciliationMatch,
    VLMElement,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_match(
    text_vlm: str = "",
    text_pymupdf: str = "",
    text_reconciled: str = "",
    quality: str = "exact",
    page: int = 0,
) -> ReconciliationMatch:
    return ReconciliationMatch(
        pymupdf_block_index=0,
        vlm_element_id="elem_0_0",
        match_quality=quality,
        text_pymupdf=text_pymupdf,
        text_vlm=text_vlm,
        text_reconciled=text_reconciled or text_pymupdf,
        bbox_overlap=0.5,
        page=page,
    )


def _make_element(
    element_id: str,
    element_type: str,
    page: int = 0,
    bbox: BBox = None,
    parent_id: str = None,
    text: str = "",
) -> VLMElement:
    return VLMElement(
        element_id=element_id,
        element_type=element_type,
        text=text,
        bbox=bbox,
        confidence=0.9,
        page=page,
        parent_id=parent_id,
        children_ids=[],
    )


# =============================================================================
# Check 6: bbox_text_coherence
# =============================================================================

class TestBboxTextCoherence:

    def test_coherent_texts_pass(self):
        """Textos com alta sobreposição de palavras passam."""
        matches = [
            _make_match(
                text_vlm="Art. 1o Esta Lei estabelece normas",
                text_pymupdf="Art. 1º Esta Lei estabelece normas",
            ),
        ]
        check = InspectionPipeline._check_bbox_text_coherence(matches)
        assert check.passed is True
        assert check.check_name == "bbox_text_coherence"

    def test_incoherent_texts_fail(self):
        """Textos completamente diferentes falham se > 10%."""
        matches = [
            _make_match(
                text_vlm="Texto completamente diferente aqui",
                text_pymupdf="Art. 1o Esta Lei estabelece normas gerais",
            ),
        ]
        check = InspectionPipeline._check_bbox_text_coherence(matches)
        assert check.passed is False

    def test_empty_matches(self):
        """Sem matches exact/partial: passa."""
        matches = [
            _make_match(quality="unmatched_vlm"),
        ]
        check = InspectionPipeline._check_bbox_text_coherence(matches)
        assert check.passed is True

    def test_low_failure_rate_passes(self):
        """Se < 10% das matches falham, passa."""
        # 19 boas + 1 ruim = 5% failure rate
        good = [
            _make_match(
                text_vlm=f"Art. {i}o texto do artigo legal número {i}",
                text_pymupdf=f"Art. {i}º texto do artigo legal número {i}",
            )
            for i in range(19)
        ]
        bad = [
            _make_match(
                text_vlm="Completamente diferente xyz abc def",
                text_pymupdf="Art. 99o outro texto totalmente distinto ghi jkl",
            ),
        ]
        check = InspectionPipeline._check_bbox_text_coherence(good + bad)
        assert check.passed is True


# =============================================================================
# Check 7: bbox_containment
# =============================================================================

class TestBboxContainment:

    def test_child_contained_passes(self):
        """Filho geometricamente contido no pai: passa."""
        elements = {
            "parent": _make_element(
                "parent", "artigo",
                bbox=BBox(x0=50, y0=50, x1=500, y1=500),
            ),
            "child": _make_element(
                "child", "paragrafo",
                bbox=BBox(x0=60, y0=60, x1=400, y1=200),
                parent_id="parent",
            ),
        }
        check = InspectionPipeline._check_bbox_containment(elements)
        assert check.passed is True

    def test_child_outside_fails(self):
        """Filho fora do pai: falha."""
        elements = {
            "parent": _make_element(
                "parent", "artigo",
                bbox=BBox(x0=50, y0=50, x1=200, y1=200),
            ),
            "child": _make_element(
                "child", "paragrafo",
                bbox=BBox(x0=300, y0=300, x1=500, y1=500),
                parent_id="parent",
            ),
        }
        check = InspectionPipeline._check_bbox_containment(elements)
        assert check.passed is False
        assert check.details["violations"] == 1

    def test_no_parent_passes(self):
        """Elementos sem pai: passa (nada para validar)."""
        elements = {
            "art1": _make_element("art1", "artigo", bbox=BBox(x0=50, y0=50, x1=500, y1=500)),
        }
        check = InspectionPipeline._check_bbox_containment(elements)
        assert check.passed is True

    def test_partial_overlap_fails(self):
        """Filho com < 50% de overlap: falha."""
        elements = {
            "parent": _make_element(
                "parent", "artigo",
                bbox=BBox(x0=0, y0=0, x1=100, y1=100),
            ),
            "child": _make_element(
                "child", "paragrafo",
                bbox=BBox(x0=80, y0=80, x1=200, y1=200),  # small overlap
                parent_id="parent",
            ),
        }
        check = InspectionPipeline._check_bbox_containment(elements)
        assert check.passed is False


# =============================================================================
# Check 8: offset_coverage
# =============================================================================

class TestOffsetCoverage:

    def test_all_found_passes(self):
        """Todos os matches encontrados no canonical: passa."""
        canonical = "Art. 1o Esta Lei estabelece normas gerais de licitação."
        matches = [
            _make_match(text_reconciled="Art. 1o Esta Lei estabelece normas gerais de licitação."),
        ]
        check = InspectionPipeline._check_offset_coverage(matches, canonical)
        assert check.passed is True

    def test_none_found_fails(self):
        """Nenhum match encontrado no canonical: falha."""
        canonical = "Texto completamente diferente"
        matches = [
            _make_match(text_reconciled="Art. 1o Esta Lei estabelece"),
        ]
        check = InspectionPipeline._check_offset_coverage(matches, canonical)
        assert check.passed is False

    def test_above_80_percent_passes(self):
        """Acima de 80% de cobertura: passa."""
        canonical = "Art. 1 texto. Art. 2 texto. Art. 3 texto. Art. 4 texto. Art. 5 outro."
        matches = [
            _make_match(text_reconciled="Art. 1 texto."),
            _make_match(text_reconciled="Art. 2 texto."),
            _make_match(text_reconciled="Art. 3 texto."),
            _make_match(text_reconciled="Art. 4 texto."),
            _make_match(text_reconciled="NOT FOUND"),
        ]
        check = InspectionPipeline._check_offset_coverage(matches, canonical)
        assert check.passed is True
        assert check.details["rate"] == 0.8

    def test_unmatched_excluded(self):
        """Matches unmatched não contam."""
        canonical = "some text"
        matches = [
            _make_match(text_reconciled="some text"),
            _make_match(quality="unmatched_vlm"),
            _make_match(quality="unmatched_pymupdf"),
        ]
        check = InspectionPipeline._check_offset_coverage(matches, canonical)
        assert check.passed is True
        assert check.details["total"] == 1


# =============================================================================
# Check 9: hierarchy_consistency
# =============================================================================

class TestHierarchyConsistency:

    def test_correct_hierarchy_passes(self):
        """artigo > paragrafo > inciso > alinea: passa."""
        elements = {
            "art": _make_element("art", "artigo"),
            "par": _make_element("par", "paragrafo", parent_id="art"),
            "inc": _make_element("inc", "inciso", parent_id="par"),
            "ali": _make_element("ali", "alinea", parent_id="inc"),
        }
        check = InspectionPipeline._check_hierarchy_consistency(elements)
        assert check.passed is True

    def test_inverted_hierarchy_fails(self):
        """artigo como filho de paragrafo: falha."""
        elements = {
            "par": _make_element("par", "paragrafo"),
            "art": _make_element("art", "artigo", parent_id="par"),
        }
        check = InspectionPipeline._check_hierarchy_consistency(elements)
        assert check.passed is False
        assert check.details["violations"] == 1

    def test_same_depth_fails(self):
        """Dois artigos em relação pai-filho: falha."""
        elements = {
            "art1": _make_element("art1", "artigo"),
            "art2": _make_element("art2", "artigo", parent_id="art1"),
        }
        check = InspectionPipeline._check_hierarchy_consistency(elements)
        assert check.passed is False

    def test_no_hierarchy_passes(self):
        """Sem relações pai-filho: passa."""
        elements = {
            "art1": _make_element("art1", "artigo"),
            "art2": _make_element("art2", "artigo"),
        }
        check = InspectionPipeline._check_hierarchy_consistency(elements)
        assert check.passed is True


# =============================================================================
# Check 10: cross_page_continuity
# =============================================================================

class TestCrossPageContinuity:

    def test_consecutive_pages_passes(self):
        """Artigos em páginas 0, 1, 2: passa."""
        elements = {
            "art1": _make_element("art1", "artigo", page=0),
            "art2": _make_element("art2", "artigo", page=1),
            "art3": _make_element("art3", "artigo", page=2),
        }
        check = InspectionPipeline._check_cross_page_continuity(elements)
        assert check.passed is True

    def test_gap_fails(self):
        """Artigos em páginas 0, 5: falha (gap > 1)."""
        elements = {
            "art1": _make_element("art1", "artigo", page=0),
            "art2": _make_element("art2", "artigo", page=5),
        }
        check = InspectionPipeline._check_cross_page_continuity(elements)
        assert check.passed is False
        assert check.details["violations"] == 1

    def test_single_article_passes(self):
        """Um único artigo: passa (nada para comparar)."""
        elements = {
            "art1": _make_element("art1", "artigo", page=0),
        }
        check = InspectionPipeline._check_cross_page_continuity(elements)
        assert check.passed is True

    def test_same_page_passes(self):
        """Múltiplos artigos na mesma página: passa."""
        elements = {
            "art1": _make_element("art1", "artigo", page=0),
            "art2": _make_element("art2", "artigo", page=0),
            "art3": _make_element("art3", "artigo", page=0),
        }
        check = InspectionPipeline._check_cross_page_continuity(elements)
        assert check.passed is True

    def test_non_article_elements_ignored(self):
        """Elementos não-artigo não afetam o check."""
        elements = {
            "art1": _make_element("art1", "artigo", page=0),
            "par1": _make_element("par1", "paragrafo", page=5),
            "art2": _make_element("art2", "artigo", page=1),
        }
        check = InspectionPipeline._check_cross_page_continuity(elements)
        assert check.passed is True
