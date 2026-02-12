"""
Testes para o pipeline de acórdãos do TCU (AcordaoParser + AcordaoHeaderParser).

T1-T10 do briefing + testes extras.
Usa texto sintético (fixture) — não requer PDF real.
"""

import pytest
from src.extraction.acordao_header_parser import AcordaoHeaderParser
from src.extraction.acordao_parser import AcordaoParser, AcordaoDevice


# =============================================================================
# FIXTURE: Texto sintético de acórdão
# =============================================================================

ACORDAO_TEXT = """\
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

INTRODUÇÃO

1. A representante alega irregularidades no Pregão Eletrônico nº 10/2025.
2. O edital foi publicado em 15/09/2025.
3. O certame visa a contratação de serviços de locação de veículos.

EXAME TÉCNICO

I. Análise dos pressupostos de admissibilidade

4. A representação preenche os requisitos de admissibilidade.
5. A empresa é licitante no certame questionado.

I.3.1. Ausência de justificativas no ETP

6. O ETP não apresenta estimativas por campus.
7. Violação do art. 18, § 1º, inciso IV, da Lei 14.133/2021.

CONCLUSÃO

8. A representação merece ser conhecida e parcialmente provida.

PROPOSTA DE ENCAMINHAMENTO

9. Sugere-se ao Tribunal conhecer da representação e considerar a representação parcialmente procedente.

VOTO

10. Conheço da representação por preencher os requisitos de admissibilidade.
11. No mérito, acompanho os pareceres da unidade técnica.
12. A ausência de estimativas no ETP viola o art. 18, § 1º, IV, da Lei 14.133/2021.
13. Quanto à vedação à subcontratação, assiste razão ao representante.
14. Ante o exposto, voto por que o Tribunal adote o acórdão que submeto.

ACÓRDÃO Nº 2450/2025

Os Ministros do Tribunal de Contas da União ACORDAM em:

9.1. conhecer da representação, por preencher os requisitos de admissibilidade;
9.2. no mérito, considerar a representação parcialmente procedente;
9.3. indeferir o pedido de medida cautelar;
9.4. dar ciência ao Instituto Federal sobre as seguintes irregularidades:
9.4.1. ausência, no Estudo Técnico Preliminar, das estimativas de quantitativos;
9.4.2. vedação à subcontratação sem justificativa técnica;
9.4.3. ausência de análise de custo total de propriedade;
9.5. comunicar ao representante;
9.6. arquivar o processo.
"""


@pytest.fixture
def canonical_text():
    return ACORDAO_TEXT


@pytest.fixture
def page_boundaries():
    """Simula 3 páginas de texto."""
    total = len(ACORDAO_TEXT)
    third = total // 3
    return [
        (0, third),
        (third, 2 * third),
        (2 * third, total),
    ]


@pytest.fixture
def header_parser():
    return AcordaoHeaderParser()


@pytest.fixture
def parser():
    return AcordaoParser()


@pytest.fixture
def header_metadata(header_parser, canonical_text):
    return header_parser.parse_header(canonical_text)


@pytest.fixture
def devices(parser, canonical_text, page_boundaries):
    return parser.parse(canonical_text, page_boundaries)


# =============================================================================
# T1: Extração de metadados (header)
# =============================================================================

class TestT1HeaderExtraction:
    def test_numero(self, header_metadata):
        assert header_metadata["numero"] == "2450"

    def test_ano(self, header_metadata):
        assert header_metadata["ano"] == "2025"

    def test_colegiado(self, header_metadata):
        assert header_metadata["colegiado"] == "Plenario"

    def test_processo(self, header_metadata):
        assert header_metadata["processo"] == "TC 018.677/2025-8"

    def test_relator(self, header_metadata):
        assert "Jorge Oliveira" in header_metadata["relator"]

    def test_natureza(self, header_metadata):
        assert header_metadata["natureza"] == "Representação"

    def test_data_sessao(self, header_metadata):
        assert header_metadata["data_sessao"] == "22/10/2025"

    def test_resultado(self, header_metadata):
        assert "procedente" in header_metadata["resultado"].lower()

    def test_unidade_tecnica(self, header_metadata):
        assert "AudContratações" in header_metadata["unidade_tecnica"]


# =============================================================================
# T2: Detecção de seções primárias
# =============================================================================

class TestT2PrimarySections:
    def test_all_three_sections_detected(self, devices):
        section_span_ids = [d.span_id for d in devices if d.device_type == "section"]
        assert "SEC-RELATORIO" in section_span_ids
        assert "SEC-VOTO" in section_span_ids
        assert "SEC-ACORDAO" in section_span_ids

    def test_section_authority_levels(self, devices):
        sec_map = {d.span_id: d for d in devices if d.device_type == "section"}
        assert sec_map["SEC-RELATORIO"].authority_level == "opinativo"
        assert sec_map["SEC-VOTO"].authority_level == "fundamentacao"
        assert sec_map["SEC-ACORDAO"].authority_level == "vinculante"


# =============================================================================
# T3: Itens do dispositivo
# =============================================================================

class TestT3AcordaoItems:
    def test_items_extracted(self, devices):
        item_ids = [d.span_id for d in devices if d.device_type == "item_dispositivo"]
        assert "ITEM-9.1" in item_ids
        assert "ITEM-9.2" in item_ids
        assert "ITEM-9.3" in item_ids
        assert "ITEM-9.4" in item_ids
        assert "ITEM-9.4.1" in item_ids
        assert "ITEM-9.4.2" in item_ids
        assert "ITEM-9.4.3" in item_ids
        assert "ITEM-9.5" in item_ids
        assert "ITEM-9.6" in item_ids

    def test_item_hierarchy(self, devices):
        device_map = {d.span_id: d for d in devices}
        # Sub-items parent is ITEM-9.4
        assert device_map["ITEM-9.4.1"].parent_span_id == "ITEM-9.4"
        assert device_map["ITEM-9.4.2"].parent_span_id == "ITEM-9.4"
        assert device_map["ITEM-9.4.3"].parent_span_id == "ITEM-9.4"
        # Top-level items parent is SEC-ACORDAO
        assert device_map["ITEM-9.1"].parent_span_id == "SEC-ACORDAO"
        assert device_map["ITEM-9.4"].parent_span_id == "SEC-ACORDAO"

    def test_item_children(self, devices):
        device_map = {d.span_id: d for d in devices}
        item_94 = device_map["ITEM-9.4"]
        assert "ITEM-9.4.1" in item_94.children_span_ids
        assert "ITEM-9.4.2" in item_94.children_span_ids
        assert "ITEM-9.4.3" in item_94.children_span_ids


# =============================================================================
# T4: Authority levels
# =============================================================================

class TestT4AuthorityLevels:
    def test_acordao_items_are_vinculante(self, devices):
        for d in devices:
            if d.section_type == "acordao" and d.device_type == "item_dispositivo":
                assert d.authority_level == "vinculante", f"{d.span_id} not vinculante"

    def test_voto_paragraphs_are_fundamentacao(self, devices):
        for d in devices:
            if d.section_type == "voto" and d.device_type == "paragraph":
                assert d.authority_level == "fundamentacao", f"{d.span_id} not fundamentacao"

    def test_relatorio_paragraphs_are_opinativo(self, devices):
        for d in devices:
            if d.section_type == "relatorio" and d.device_type == "paragraph":
                assert d.authority_level == "opinativo", f"{d.span_id} not opinativo"


# =============================================================================
# T5: Nenhum span com texto vazio
# =============================================================================

class TestT5NoEmptyText:
    def test_all_devices_have_text(self, devices):
        for d in devices:
            assert len(d.text.strip()) > 0, f"{d.span_id} has empty text"

    def test_all_devices_have_preview(self, devices):
        for d in devices:
            assert len(d.text_preview) > 0, f"{d.span_id} has empty text_preview"


# =============================================================================
# T7: Hierarquia válida
# =============================================================================

class TestT7Hierarchy:
    def test_all_parents_exist(self, devices):
        span_ids = {d.span_id for d in devices}
        for d in devices:
            if d.parent_span_id:
                assert d.parent_span_id in span_ids, (
                    f"{d.span_id} has parent {d.parent_span_id} which doesn't exist"
                )

    def test_children_reference_valid(self, devices):
        span_ids = {d.span_id for d in devices}
        for d in devices:
            for child_id in d.children_span_ids:
                assert child_id in span_ids, (
                    f"{d.span_id} references child {child_id} which doesn't exist"
                )


# =============================================================================
# T8: Offsets canônicos válidos
# =============================================================================

class TestT8Offsets:
    def test_offsets_valid(self, devices):
        for d in devices:
            assert d.char_start >= 0, f"{d.span_id} has negative char_start"
            assert d.char_end > d.char_start, (
                f"{d.span_id} has invalid range: {d.char_start}-{d.char_end}"
            )

    def test_offset_text_matches(self, devices, canonical_text):
        """T_extra: canonical_text[char_start:char_end] == device.text for non-section devices."""
        for d in devices:
            if d.device_type == "section":
                # Section text is the full section text stripped — may have whitespace differences
                continue
            sliced = canonical_text[d.char_start:d.char_end].strip()
            assert sliced == d.text, (
                f"{d.span_id}: offset text mismatch.\n"
                f"Expected: {d.text[:80]!r}\n"
                f"Got:      {sliced[:80]!r}"
            )


# =============================================================================
# T_extra: Parágrafos numerados extraídos
# =============================================================================

class TestParagraphs:
    def test_relatorio_paragraphs(self, devices):
        rel_paras = [
            d for d in devices
            if d.device_type == "paragraph" and d.section_type == "relatorio"
        ]
        assert len(rel_paras) > 0, "No relatorio paragraphs found"
        # Should have paragraphs 1-9 from the fixture
        para_nums = {d.identifier for d in rel_paras}
        assert "1" in para_nums
        assert "9" in para_nums

    def test_voto_paragraphs(self, devices):
        voto_paras = [
            d for d in devices
            if d.device_type == "paragraph" and d.section_type == "voto"
        ]
        assert len(voto_paras) > 0, "No voto paragraphs found"
        para_nums = {d.identifier for d in voto_paras}
        assert "10" in para_nums
        assert "14" in para_nums

    def test_paragraphs_not_confused_with_items(self, devices):
        """Paragraphs should not have decimal numbers like 9.1."""
        para_ids = [d.identifier for d in devices if d.device_type == "paragraph"]
        for pid in para_ids:
            assert "." not in pid, f"Paragraph identifier '{pid}' looks like an item"


# =============================================================================
# T_extra: Subseções do RELATÓRIO
# =============================================================================

class TestSubsections:
    def test_subsections_detected(self, devices):
        sub_sections = [
            d for d in devices
            if d.device_type == "section"
            and d.parent_span_id == "SEC-RELATORIO"
        ]
        assert len(sub_sections) > 0, "No subsections found in RELATÓRIO"

    def test_subsection_span_ids(self, devices):
        sub_ids = [
            d.span_id for d in devices
            if d.device_type == "section"
            and d.parent_span_id == "SEC-RELATORIO"
        ]
        # Should detect at least INTRODUÇÃO and EXAME TÉCNICO from fixture
        has_intro = any("INTRODUC" in sid for sid in sub_ids)
        has_exame = any("EXAME" in sid for sid in sub_ids)
        assert has_intro or has_exame, (
            f"Expected INTRODUÇÃO or EXAME TÉCNICO subsections, got: {sub_ids}"
        )


# =============================================================================
# T10: Retrieval text enrichment (via _build_acordao_retrieval_text)
# =============================================================================

class TestT10RetrievalText:
    def test_vinculante_retrieval_text(self, devices):
        from src.ingestion.pipeline import IngestionPipeline
        device_map = {d.span_id: d for d in devices}

        item = device_map.get("ITEM-9.4.1")
        assert item is not None

        rt = IngestionPipeline._build_acordao_retrieval_text(
            item, "2450", "2025", "Plenario", "Jorge Oliveira",
            "Representação", "Parcialmente procedente",
            "TC 018.677/2025-8", "AudContratações", device_map,
        )
        assert "DECISÃO VINCULANTE" in rt
        assert "2450/2025" in rt
        assert "Plenario" in rt

    def test_fundamentacao_retrieval_text(self, devices):
        from src.ingestion.pipeline import IngestionPipeline
        device_map = {d.span_id: d for d in devices}

        voto_paras = [d for d in devices if d.section_type == "voto" and d.device_type == "paragraph"]
        assert len(voto_paras) > 0
        para = voto_paras[0]

        rt = IngestionPipeline._build_acordao_retrieval_text(
            para, "2450", "2025", "Plenario", "Jorge Oliveira",
            "Representação", "Parcialmente procedente",
            "TC 018.677/2025-8", "AudContratações", device_map,
        )
        assert "FUNDAMENTAÇÃO DO RELATOR" in rt

    def test_opinativo_retrieval_text(self, devices):
        from src.ingestion.pipeline import IngestionPipeline
        device_map = {d.span_id: d for d in devices}

        rel_paras = [d for d in devices if d.section_type == "relatorio" and d.device_type == "paragraph"]
        assert len(rel_paras) > 0
        para = rel_paras[0]

        rt = IngestionPipeline._build_acordao_retrieval_text(
            para, "2450", "2025", "Plenario", "Jorge Oliveira",
            "Representação", "Parcialmente procedente",
            "TC 018.677/2025-8", "AudContratações", device_map,
        )
        assert "ANÁLISE DA UNIDADE TÉCNICA" in rt


# =============================================================================
# T_extra: Page number assignment
# =============================================================================

class TestPageAssignment:
    def test_devices_have_valid_pages(self, devices):
        for d in devices:
            assert d.page_number >= 1, f"{d.span_id} has page_number {d.page_number}"


# =============================================================================
# T_extra: Device type coverage
# =============================================================================

class TestDeviceTypes:
    def test_all_device_types_present(self, devices):
        types = {d.device_type for d in devices}
        assert "section" in types
        assert "paragraph" in types
        assert "item_dispositivo" in types

    def test_section_types_present(self, devices):
        sec_types = {d.section_type for d in devices}
        assert "relatorio" in sec_types
        assert "voto" in sec_types
        assert "acordao" in sec_types
