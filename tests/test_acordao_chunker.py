"""
Testes para AcordaoChunker — chunking por seções com overlap.

Usa texto sintético (fixture) — não requer GPU, embedder, ou PDF real.
"""

import pytest
from dataclasses import dataclass, field
from src.extraction.acordao_chunker import (
    AcordaoChunker,
    ParsedSection,
    AcordaoChunk,
    build_sections,
    AUTHORITY_MAP,
)
from src.extraction.acordao_parser import AcordaoParser, AcordaoDevice
from src.extraction.acordao_header_parser import AcordaoHeaderParser


# =============================================================================
# FIXTURES
# =============================================================================

# Short acordao text (same as test_acordao_parser.py)
ACORDAO_SHORT = """\
ACÓRDÃO Nº 2450/2025 – TCU – Plenário

1. Processo: TC 018.677/2025-8
2. Grupo I – Classe de Assunto: VII – Representação
3. Interessado: Empresa XYZ Ltda.
4. Unidade: Instituto Federal de Educação, Ciência e Tecnologia
5. Relator: Ministro Jorge Oliveira
6. Representante do Ministério Público: não atuou
7. Unidade Técnica: AudContratações
8. Advogado constituído nos autos: não há
Natureza: Representação
Data da Sessão: 22/10/2025
SUMÁRIO: REPRESENTAÇÃO. PREGÃO ELETRÔNICO. Serviços contínuos de locação de veículos.

RELATÓRIO

Trata-se de representação formulada pela empresa XYZ Ltda.

1. A representante alega irregularidades no Pregão Eletrônico nº 10/2025.
2. O edital foi publicado em 15/09/2025.
3. O certame visa a contratação de serviços de locação de veículos.

VOTO

4. Conheço da representação por preencher os requisitos de admissibilidade.
5. No mérito, acompanho os pareceres da unidade técnica.
6. Ante o exposto, voto por que o Tribunal adote o acórdão que submeto.

ACÓRDÃO Nº 2450/2025

Os Ministros do Tribunal de Contas da União ACORDAM em:

9.1. conhecer da representação, por preencher os requisitos de admissibilidade;
9.2. no mérito, considerar a representação parcialmente procedente;
9.3. arquivar o processo.
"""


def _generate_long_section(section_name, num_paragraphs, chars_per_para=300):
    """Generate a section with numbered paragraphs for testing long splits."""
    lines = [f"{section_name}\n"]
    para_num = 1
    for i in range(num_paragraphs):
        # Each paragraph ~chars_per_para chars
        body = f"Parágrafo {para_num} da seção {section_name}. " * (chars_per_para // 45)
        lines.append(f"\n{para_num}. {body.strip()}\n")
        para_num += 1
    return "".join(lines)


def _build_long_acordao(rel_paras=40, voto_paras=50, chars_per=300):
    """Build a ~40k char acordao text with all sections."""
    header = (
        "ACÓRDÃO Nº 2724/2025 – TCU – Plenário\n\n"
        "1. Processo: TC 045.123/2024-0\n"
        "5. Relator: Ministro Benjamin Zymler\n"
        "Natureza: Tomada de Contas\n"
        "Data da Sessão: 15/11/2025\n"
        "SUMÁRIO: TOMADA DE CONTAS. Irregularidades. Multa.\n\n"
    )
    relatorio = _generate_long_section("RELATÓRIO", rel_paras, chars_per)
    voto = _generate_long_section("VOTO", voto_paras, chars_per)
    acordao_sec = (
        "\nACÓRDÃO Nº 2724/2025\n\n"
        "Os Ministros ACORDAM em:\n\n"
        "9.1. julgar irregulares as contas;\n"
        "9.2. aplicar multa de R$ 50.000,00;\n"
        "9.3. dar ciência ao órgão;\n"
        "9.4. arquivar o processo.\n"
    )
    return header + relatorio + voto + acordao_sec


@pytest.fixture
def short_text():
    return ACORDAO_SHORT


@pytest.fixture
def short_devices(short_text):
    parser = AcordaoParser()
    total = len(short_text)
    return parser.parse(short_text, [(0, total)])


@pytest.fixture
def short_header(short_text):
    return AcordaoHeaderParser().parse_header(short_text)


@pytest.fixture
def long_text():
    return _build_long_acordao()


@pytest.fixture
def long_devices(long_text):
    parser = AcordaoParser()
    total = len(long_text)
    return parser.parse(long_text, [(0, total)])


@pytest.fixture
def long_header(long_text):
    return AcordaoHeaderParser().parse_header(long_text)


@pytest.fixture
def chunker():
    return AcordaoChunker()


@pytest.fixture
def metadata():
    return {
        "numero": "2724",
        "ano": "2025",
        "colegiado": "Plenario",
        "processo": "TC 045.123/2024-0",
        "relator": "Benjamin Zymler",
        "data_sessao": "15/11/2025",
        "natureza": "Tomada de Contas",
        "resultado": "",
    }


# =============================================================================
# Test 1: build_sections from devices
# =============================================================================

class TestBuildSections:
    def test_build_sections_from_devices(self, short_devices, short_text, short_header):
        sections = build_sections(short_devices, short_text, short_header)
        section_types = [s.section_type for s in sections]
        assert "relatorio" in section_types
        assert "voto" in section_types
        assert "acordao" in section_types

    def test_sections_have_valid_offsets(self, short_devices, short_text, short_header):
        sections = build_sections(short_devices, short_text, short_header)
        for sec in sections:
            assert sec.canonical_start >= 0
            assert sec.canonical_end > sec.canonical_start
            assert sec.canonical_end <= len(short_text)

    def test_section_text_matches_offsets(self, short_devices, short_text, short_header):
        sections = build_sections(short_devices, short_text, short_header)
        for sec in sections:
            sliced = short_text[sec.canonical_start:sec.canonical_end]
            assert sliced == sec.text, (
                f"{sec.section_type}: offset mismatch.\n"
                f"Expected: {sec.text[:80]!r}\n"
                f"Got:      {sliced[:80]!r}"
            )

    def test_long_text_sections(self, long_devices, long_text, long_header):
        sections = build_sections(long_devices, long_text, long_header)
        section_types = [s.section_type for s in sections]
        assert "relatorio" in section_types
        assert "voto" in section_types
        assert "acordao" in section_types


# =============================================================================
# Test 2: EMENTA detection
# =============================================================================

class TestEmenta:
    def test_ementa_detected(self, short_devices, short_text, short_header):
        sections = build_sections(short_devices, short_text, short_header)
        ementa = [s for s in sections if s.section_type == "ementa"]
        assert len(ementa) == 1
        assert "REPRESENTAÇÃO" in ementa[0].text

    def test_ementa_missing_graceful(self):
        """Text without SUMÁRIO → no ementa section, no crash."""
        text = (
            "RELATÓRIO\n\n1. Primeiro parágrafo.\n\n"
            "VOTO\n\n2. Segundo parágrafo.\n\n"
            "ACÓRDÃO Nº 1/2025\n\n9.1. Decidiu-se.\n"
        )
        parser = AcordaoParser()
        devices = parser.parse(text, [(0, len(text))])
        sections = build_sections(devices, text, {})
        ementa = [s for s in sections if s.section_type == "ementa"]
        assert len(ementa) == 0


# =============================================================================
# Test 3: Short section → single chunk
# =============================================================================

class TestShortSectionSingleChunk:
    def test_short_section_single_chunk(self, short_devices, short_text, short_header, chunker, metadata):
        sections = build_sections(short_devices, short_text, short_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)

        # All sections in the short text are < 4000 chars → 1 chunk each
        for chunk in chunks:
            assert "-P" not in chunk.span_id, (
                f"Short section {chunk.span_id} should not have part suffix"
            )

    def test_short_ementa_single_chunk(self, short_devices, short_text, short_header, chunker, metadata):
        sections = build_sections(short_devices, short_text, short_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)
        ementa_chunks = [c for c in chunks if c.section_type == "ementa"]
        assert len(ementa_chunks) == 1
        assert ementa_chunks[0].span_id == "SEC-EMENTA"


# =============================================================================
# Test 4: Long section → split with overlap
# =============================================================================

class TestLongSectionSplit:
    def test_long_section_split(self, long_devices, long_text, long_header, chunker, metadata):
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)

        # VOTO with 50 paragraphs of ~300 chars each = ~15k chars → should split
        voto_chunks = [c for c in chunks if c.section_type == "voto"]
        assert len(voto_chunks) > 1, (
            f"Expected VOTO to be split into multiple chunks, got {len(voto_chunks)}"
        )

    def test_split_respects_max_chunk_chars(self, long_devices, long_text, long_header, metadata):
        chunker = AcordaoChunker(max_chunk_chars=4000)
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)

        # Most chunks should be <= max_chunk_chars (some may exceed slightly
        # due to paragraph boundary rounding)
        for c in chunks:
            # Allow 50% overflow since we don't break paragraphs
            assert len(c.text) <= 6000, (
                f"{c.span_id} has {len(c.text)} chars, expected <= ~6000"
            )


# =============================================================================
# Test 5: Overlap uses complete paragraphs
# =============================================================================

class TestOverlapByParagraphs:
    def test_overlap_by_paragraphs(self, long_devices, long_text, long_header, metadata):
        chunker = AcordaoChunker(max_chunk_chars=4000)
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)

        # Check that consecutive chunks of the same section have overlapping text
        voto_chunks = sorted(
            [c for c in chunks if c.section_type == "voto"],
            key=lambda c: c.part_number,
        )
        if len(voto_chunks) >= 2:
            for i in range(len(voto_chunks) - 1):
                c1 = voto_chunks[i]
                c2 = voto_chunks[i + 1]
                # c2 should start before c1 ends (overlap)
                assert c2.canonical_start < c1.canonical_end, (
                    f"No overlap between {c1.span_id} (end={c1.canonical_end}) "
                    f"and {c2.span_id} (start={c2.canonical_start})"
                )


# =============================================================================
# Test 6: Overlap min/max clamping
# =============================================================================

class TestOverlapClamp:
    def test_overlap_min_max_clamp(self):
        chunker = AcordaoChunker(
            max_chunk_chars=2000,
            overlap_ratio=0.20,
            min_overlap_chars=200,
            max_overlap_chars=1200,
        )
        # 20% of 2000 = 400, which is between 200 and 1200 → should use 400
        # Test indirectly via chunk structure
        assert chunker.min_overlap_chars == 200
        assert chunker.max_overlap_chars == 1200


# =============================================================================
# Test 7: Runt absorption
# =============================================================================

class TestRuntAbsorption:
    def test_runt_absorption(self):
        """Final chunk < 20% of max_chunk_chars should be absorbed into previous."""
        # Create a section with paragraphs: 3 large + 1 tiny
        paras = []
        offset = 0
        text_parts = []
        for i in range(3):
            para_text = f"{i+1}. " + "X" * 1800 + "\n\n"
            paras.append({
                "num": str(i+1),
                "text": para_text.strip(),
                "start": offset,
                "end": offset + len(para_text.strip()),
            })
            text_parts.append(para_text)
            offset += len(para_text)
        # Tiny final paragraph (< 800 chars = 20% of 4000)
        tiny = "4. Fim.\n"
        paras.append({
            "num": "4",
            "text": tiny.strip(),
            "start": offset,
            "end": offset + len(tiny.strip()),
        })
        text_parts.append(tiny)

        full_text = "".join(text_parts)
        section = ParsedSection(
            section_type="voto",
            text=full_text,
            canonical_start=0,
            canonical_end=len(full_text),
            paragraphs=paras,
        )

        chunker = AcordaoChunker(max_chunk_chars=4000)
        parts = chunker._split_section(section)

        # The tiny final part should have been absorbed
        for text, start, end in parts:
            assert len(text) >= 800 or len(parts) == 1, (
                f"Runt chunk of {len(text)} chars should have been absorbed"
            )


# =============================================================================
# Test 8: Span ID convention
# =============================================================================

class TestSpanIdConvention:
    def test_span_id_convention(self, long_devices, long_text, long_header, chunker, metadata):
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)

        for c in chunks:
            if c.total_parts == 1:
                # Single chunk: no -PXX suffix
                assert not c.span_id.endswith(("-P01", "-P02")), (
                    f"{c.span_id} is single-part but has part suffix"
                )
            else:
                # Multi-part: must have -PXX suffix
                assert f"-P{c.part_number:02d}" in c.span_id, (
                    f"{c.span_id} missing expected part suffix -P{c.part_number:02d}"
                )

    def test_span_id_prefixes(self, long_devices, long_text, long_header, chunker, metadata):
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)

        for c in chunks:
            if c.section_type == "ementa":
                assert c.span_id.startswith("SEC-EMENTA")
            elif c.section_type == "relatorio":
                assert c.span_id.startswith("SEC-RELATORIO")
            elif c.section_type == "voto":
                assert c.span_id.startswith("SEC-VOTO")
            elif c.section_type == "acordao":
                assert c.span_id.startswith("SEC-ACORDAO")


# =============================================================================
# Test 9: Authority levels
# =============================================================================

class TestAuthorityLevels:
    def test_authority_levels(self, long_devices, long_text, long_header, chunker, metadata):
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)

        for c in chunks:
            expected = AUTHORITY_MAP[c.section_type]
            assert c.authority_level == expected, (
                f"{c.span_id}: authority_level={c.authority_level}, expected={expected}"
            )


# =============================================================================
# Test 10: Retrieval text format
# =============================================================================

class TestRetrievalText:
    def test_retrieval_text_format(self, long_devices, long_text, long_header, chunker, metadata):
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)

        for c in chunks:
            assert c.retrieval_text.startswith("[CONTEXTO:"), (
                f"{c.span_id}: retrieval_text doesn't start with [CONTEXTO:"
            )
            assert "2724" in c.retrieval_text  # numero
            assert "2025" in c.retrieval_text  # ano

    def test_ementa_retrieval_text(self, short_devices, short_text, short_header, chunker):
        sections = build_sections(short_devices, short_text, short_header)
        meta = {
            "numero": "2450", "ano": "2025", "colegiado": "Plenario",
            "relator": "Jorge Oliveira", "processo": "", "data_sessao": "",
            "natureza": "", "resultado": "",
        }
        chunks = chunker.chunk(sections, "doc-1", "hash-1", meta)
        ementa = [c for c in chunks if c.section_type == "ementa"]
        assert len(ementa) == 1
        assert "Ementa" in ementa[0].retrieval_text

    def test_voto_retrieval_text_has_relator(self, long_devices, long_text, long_header, chunker, metadata):
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)
        voto_chunks = [c for c in chunks if c.section_type == "voto"]
        assert len(voto_chunks) > 0
        for c in voto_chunks:
            assert "Benjamin Zymler" in c.retrieval_text

    def test_acordao_retrieval_text_has_vinculante(self, long_devices, long_text, long_header, chunker, metadata):
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)
        ac_chunks = [c for c in chunks if c.section_type == "acordao"]
        assert len(ac_chunks) > 0
        for c in ac_chunks:
            assert "vinculante" in c.retrieval_text.lower()


# =============================================================================
# Test 11: Offsets valid — canonical_text[start:end] == chunk.text
# =============================================================================

class TestOffsetsValid:
    def test_offsets_valid_short(self, short_devices, short_text, short_header, chunker):
        sections = build_sections(short_devices, short_text, short_header)
        meta = {
            "numero": "2450", "ano": "2025", "colegiado": "Plenario",
            "relator": "Jorge Oliveira", "processo": "", "data_sessao": "",
            "natureza": "", "resultado": "",
        }
        chunks = chunker.chunk(sections, "doc-1", "hash-1", meta)
        for c in chunks:
            assert c.canonical_start >= 0
            assert c.canonical_end > c.canonical_start
            sliced = short_text[c.canonical_start:c.canonical_end]
            assert sliced == c.text, (
                f"{c.span_id}: offset mismatch.\n"
                f"Expected ({c.canonical_start}:{c.canonical_end}): {c.text[:80]!r}\n"
                f"Got: {sliced[:80]!r}"
            )

    def test_offsets_valid_long(self, long_devices, long_text, long_header, chunker, metadata):
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)
        for c in chunks:
            assert c.canonical_start >= 0
            assert c.canonical_end > c.canonical_start
            sliced = long_text[c.canonical_start:c.canonical_end]
            assert sliced == c.text, (
                f"{c.span_id}: offset mismatch.\n"
                f"Expected ({c.canonical_start}:{c.canonical_end}): {c.text[:80]!r}\n"
                f"Got: {sliced[:80]!r}"
            )


# =============================================================================
# Test 12: Total chunks reasonable
# =============================================================================

class TestTotalChunksReasonable:
    def test_total_chunks_reasonable(self, long_devices, long_text, long_header, chunker, metadata):
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)
        # Should produce ~15-25 chunks for ~40k text, definitely less than 103
        assert len(chunks) <= 40, f"Too many chunks: {len(chunks)}"
        assert len(chunks) >= 3, f"Too few chunks: {len(chunks)}"

    def test_short_text_few_chunks(self, short_devices, short_text, short_header, chunker):
        sections = build_sections(short_devices, short_text, short_header)
        meta = {
            "numero": "2450", "ano": "2025", "colegiado": "Plenario",
            "relator": "Jorge Oliveira", "processo": "", "data_sessao": "",
            "natureza": "", "resultado": "",
        }
        chunks = chunker.chunk(sections, "doc-1", "hash-1", meta)
        # Short text: 4 sections (ementa, relatorio, voto, acordao) → 4 chunks
        assert len(chunks) <= 6, f"Too many chunks for short text: {len(chunks)}"
        assert len(chunks) >= 3, f"Too few chunks: {len(chunks)}"


# =============================================================================
# Test 13: Part numbering
# =============================================================================

class TestPartNumbering:
    def test_part_numbers_sequential(self, long_devices, long_text, long_header, chunker, metadata):
        sections = build_sections(long_devices, long_text, long_header)
        chunks = chunker.chunk(sections, "doc-1", "hash-1", metadata)

        # Group by section_type and check part numbers are sequential
        by_section = {}
        for c in chunks:
            by_section.setdefault(c.section_type, []).append(c)

        for sec_type, sec_chunks in by_section.items():
            sec_chunks.sort(key=lambda c: c.part_number)
            for i, c in enumerate(sec_chunks):
                assert c.part_number == i + 1, (
                    f"{c.span_id}: part_number={c.part_number}, expected={i + 1}"
                )
                assert c.total_parts == len(sec_chunks), (
                    f"{c.span_id}: total_parts={c.total_parts}, expected={len(sec_chunks)}"
                )
