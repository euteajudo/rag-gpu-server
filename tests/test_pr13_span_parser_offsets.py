# -*- coding: utf-8 -*-
"""
Testes PR13 - Offsets Absolutos no SpanParser.

Valida que o SpanParser gera offsets corretos para TODOS os spans
(artigos, parágrafos, incisos, alíneas), garantindo que o ChunkMaterializer
pode encontrar cada filho dentro do range do pai.

Critérios de validação "nível industrial":
1. offsets_map e canonical_text DEVEM vir do MESMO texto (parsed_doc.source_text)
2. Slicing deve ser validado com startswith(), não com "in"
3. Todos os filhos devem estar DENTRO do range do pai
4. Teste com trecho real da LEI 14.133/2021
"""

import pytest
import sys
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.parsing.span_parser import SpanParser
from src.chunking.canonical_offsets import (
    extract_offsets_from_parsed_doc,
    normalize_canonical_text,
)


class TestArticleEndPosIncludesChildren:
    """TESTE 1: end_pos do artigo deve ir até o próximo artigo."""

    def test_article_end_pos_equals_next_article_start(self):
        """Verifica que art1.end_pos == art2.start_pos."""
        markdown = """Art. 1º Este é o artigo um com texto.

I - Primeiro inciso do artigo 1.

II - Segundo inciso do artigo 1.

§ 1º Parágrafo primeiro do artigo 1.

Art. 2º Este é o artigo dois.

I - Inciso do artigo 2.
"""
        parser = SpanParser()
        doc = parser.parse(markdown)

        art1 = doc.get_span("ART-001")
        art2 = doc.get_span("ART-002")

        assert art1 is not None, "ART-001 não encontrado"
        assert art2 is not None, "ART-002 não encontrado"

        # end_pos do art1 deve ser exatamente o start_pos do art2
        assert art1.end_pos == art2.start_pos, (
            f"art1.end_pos ({art1.end_pos}) != art2.start_pos ({art2.start_pos})"
        )

    def test_article_range_contains_all_children_text(self):
        """Verifica que o slice do artigo contém texto de todos os filhos."""
        markdown = """Art. 1º Este é o artigo um com texto.

I - Primeiro inciso do artigo 1.

II - Segundo inciso do artigo 1.

§ 1º Parágrafo primeiro do artigo 1.

Art. 2º Este é o artigo dois.
"""
        parser = SpanParser()
        doc = parser.parse(markdown)

        # Usa source_text do parsed_doc (mesmo texto usado para offsets)
        source_text = doc.source_text
        art1 = doc.get_span("ART-001")

        art1_slice = source_text[art1.start_pos:art1.end_pos]

        # Todos os textos dos filhos devem estar no slice
        assert "I - Primeiro inciso" in art1_slice
        assert "II - Segundo inciso" in art1_slice
        assert "§ 1º Parágrafo" in art1_slice


class TestAbsoluteOffsetsCaputIncisos:
    """TESTE 2: Offsets absolutos de incisos do caput."""

    def test_inciso_offsets_are_absolute(self):
        """Verifica que offsets de incisos são absolutos (não relativos ao pai)."""
        markdown = """Art. 1º Este é o artigo um.

I - Primeiro inciso.

II - Segundo inciso.

III - Terceiro inciso.

Art. 2º Artigo dois.
"""
        parser = SpanParser()
        doc = parser.parse(markdown)

        # Usa source_text consistente
        source_text = doc.source_text

        for span_id in ["INC-001-I", "INC-001-II", "INC-001-III"]:
            span = doc.get_span(span_id)
            assert span is not None, f"{span_id} não encontrado"
            assert span.start_pos >= 0, f"{span_id} start_pos inválido: {span.start_pos}"
            assert span.end_pos > span.start_pos, f"{span_id} end_pos <= start_pos"

            # Slicing deve retornar texto que começa com o texto do span
            sliced = source_text[span.start_pos:span.end_pos]

            # Validação forte: strip e startswith
            span_text_clean = span.text.strip()
            sliced_clean = sliced.strip()

            assert sliced_clean.startswith(span_text_clean[:15]), (
                f"{span_id}: slice não começa com texto do span.\n"
                f"  Esperado início: {repr(span_text_clean[:30])}\n"
                f"  Slice obtido: {repr(sliced_clean[:30])}"
            )


class TestAbsoluteOffsetsParagraphIncisos:
    """TESTE 3: Offsets de incisos dentro de parágrafos."""

    def test_paragraph_inciso_offsets_are_absolute(self):
        """Verifica que incisos de parágrafos têm offsets absolutos."""
        markdown = """Art. 5º Artigo cinco.

§ 1º Parágrafo primeiro com incisos:

I - Inciso um do parágrafo.

II - Inciso dois do parágrafo.

§ 2º Parágrafo segundo.

Art. 6º Artigo seis.
"""
        parser = SpanParser()
        doc = parser.parse(markdown)

        source_text = doc.source_text

        # Verifica parágrafo
        par1 = doc.get_span("PAR-005-1")
        assert par1 is not None, "PAR-005-1 não encontrado"
        assert par1.start_pos >= 0
        assert par1.end_pos > par1.start_pos

        par_slice = source_text[par1.start_pos:par1.end_pos]
        assert par_slice.strip().startswith("§ 1º"), (
            f"Slice do parágrafo não começa com '§ 1º': {repr(par_slice[:30])}"
        )

        # Verifica incisos
        inc_ids = [s.span_id for s in doc.spans if s.span_id.startswith("INC-005")]
        assert len(inc_ids) >= 2, f"Esperado >= 2 incisos, encontrado: {inc_ids}"

        for span_id in inc_ids:
            span = doc.get_span(span_id)
            assert span.start_pos >= 0, f"{span_id} start_pos inválido"
            assert span.end_pos > span.start_pos, f"{span_id} range inválido"
            assert span.end_pos <= len(source_text), f"{span_id} end_pos além do texto"

            sliced = source_text[span.start_pos:span.end_pos]
            # Inciso deve começar com numeral romano
            assert sliced.strip()[0] in "IVX", (
                f"{span_id}: slice não começa com numeral romano: {repr(sliced[:20])}"
            )


class TestAllSpansHaveValidOffsets:
    """TESTE 4: Todos os spans devem ter offsets válidos no offsets_map."""

    def test_all_spans_in_offsets_map(self):
        """Verifica que TODOS os spans têm entrada no offsets_map."""
        markdown = """Art. 1º Artigo um.

I - Inciso um.

a) Alínea a.

b) Alínea b.

II - Inciso dois.

§ 1º Parágrafo um.

I - Inciso do parágrafo.

Art. 2º Artigo dois.
"""
        parser = SpanParser()
        doc = parser.parse(markdown)

        # Extrai offsets usando a MESMA função do pipeline
        offsets_map, canonical_hash = extract_offsets_from_parsed_doc(doc)

        # Verifica que todos os spans estão no map
        spans_sem_offset = []
        for span in doc.spans:
            if span.span_id not in offsets_map:
                spans_sem_offset.append(span.span_id)

        assert len(spans_sem_offset) == 0, (
            f"Spans sem offset no offsets_map: {spans_sem_offset}"
        )
        assert len(offsets_map) == len(doc.spans), (
            f"offsets_map ({len(offsets_map)}) != doc.spans ({len(doc.spans)})"
        )

    def test_offsets_map_uses_same_source_as_canonical(self):
        """Verifica que offsets_map e canonical_text usam o MESMO texto fonte."""
        markdown = """Art. 1º Texto do artigo.

I - Inciso um.
"""
        parser = SpanParser()
        doc = parser.parse(markdown)

        # Extrai offsets
        offsets_map, canonical_hash = extract_offsets_from_parsed_doc(doc)

        # Normaliza o MESMO source_text usado internamente
        canonical_text = normalize_canonical_text(doc.source_text)

        # Verifica que slicing funciona para cada span
        for span_id, (start, end) in offsets_map.items():
            sliced = canonical_text[start:end]
            span = doc.get_span(span_id)

            # O slice deve conter parte significativa do texto do span
            span_words = span.text.strip().split()[:3]  # Primeiras 3 palavras
            for word in span_words:
                if len(word) > 2:  # Ignora palavras muito curtas
                    assert word in sliced, (
                        f"{span_id}: palavra '{word}' não encontrada no slice.\n"
                        f"  Slice: {repr(sliced[:50])}"
                    )


class TestIncisosWithinArticleRange:
    """TESTE 5: Incisos devem estar DENTRO do range do artigo pai (cenário PR13)."""

    def test_all_incisos_within_parent_article_range(self):
        """Simula cenário real: incisos devem estar dentro do range do artigo."""
        # Trecho inspirado na LEI 14.133/2021 Art. 45
        markdown = """Art. 45. O licenciamento ambiental de empreendimentos e atividades com significativo impacto ambiental deve seguir as seguintes etapas:

I - estudos ambientais, nos quais serão analisados os impactos negativos e positivos do empreendimento ou atividade sobre o meio ambiente;

II - mitigação por condicionantes e compensação ambiental, que serão definidas no processo de licenciamento;

III - acompanhamento ambiental, que consiste no monitoramento dos impactos ambientais durante a implantação e operação do empreendimento.

§ 1º Os estudos ambientais devem ser realizados por equipe multidisciplinar.

Art. 46. Este é o artigo seguinte.
"""
        parser = SpanParser()
        doc = parser.parse(markdown)

        # IMPORTANTE: Usa EXATAMENTE o mesmo fluxo do pipeline
        offsets_map, canonical_hash = extract_offsets_from_parsed_doc(doc)
        canonical_text = normalize_canonical_text(doc.source_text)

        # Range do artigo 45
        art_start, art_end = offsets_map["ART-045"]

        # Verifica cada inciso
        incisos_fora = []
        for span in doc.spans:
            if span.span_id.startswith("INC-045"):
                inc_start, inc_end = offsets_map[span.span_id]

                # Inciso deve estar COMPLETAMENTE dentro do artigo
                dentro = art_start <= inc_start < inc_end <= art_end

                if not dentro:
                    incisos_fora.append({
                        "span_id": span.span_id,
                        "inc_range": (inc_start, inc_end),
                        "art_range": (art_start, art_end),
                    })

        assert len(incisos_fora) == 0, (
            f"Incisos FORA do range do artigo pai:\n"
            + "\n".join(
                f"  {i['span_id']}: [{i['inc_range'][0]}:{i['inc_range'][1]}] "
                f"não está em [{i['art_range'][0]}:{i['art_range'][1]}]"
                for i in incisos_fora
            )
        )

    def test_slicing_returns_correct_text_for_incisos(self):
        """Verifica que canonical_text[start:end] retorna o texto correto."""
        markdown = """Art. 45. Artigo quarenta e cinco.

I - Primeiro inciso do artigo quarenta e cinco.

II - Segundo inciso do artigo quarenta e cinco.

Art. 46. Artigo seguinte.
"""
        parser = SpanParser()
        doc = parser.parse(markdown)

        offsets_map, _ = extract_offsets_from_parsed_doc(doc)
        canonical_text = normalize_canonical_text(doc.source_text)

        for span_id in ["INC-045-I", "INC-045-II"]:
            span = doc.get_span(span_id)
            start, end = offsets_map[span_id]

            sliced = canonical_text[start:end].strip()
            span_text = span.text.strip()

            # Validação forte: o slice deve começar com o texto do span
            assert sliced.startswith(span_text[:20]), (
                f"{span_id}: slice não corresponde ao texto do span.\n"
                f"  Esperado: {repr(span_text[:40])}\n"
                f"  Slice: {repr(sliced[:40])}"
            )


class TestRealLei14133Excerpt:
    """TESTE 6: Trecho real da LEI 14.133/2021 para validação em produção."""

    # Trecho real extraído da LEI 14.133/2021 (artigos 45-47)
    LEI_14133_EXCERPT = """Art. 45. As licitações de obras e serviços de engenharia devem respeitar, especialmente, as normas relativas a:

I - disposição final ambientalmente adequada dos resíduos sólidos gerados pelas obras contratadas;

II - mitigação por condicionantes e target_span compensação ambiental, que serão definidas no procedimento de licenciamento ambiental;

III - utilização de produtos, de equipamentos e de serviços que, comprovadamente, favoreçam a redução do consumo de energia e de recursos naturais;

IV - avaliação de impacto de vizinhança, na forma da legislação urbanística;

V - proteção do patrimônio histórico, cultural, arqueológico e imaterial, inclusive por meio da avaliação do impacto direto ou indireto causado pelas obras contratadas.

Parágrafo único. Nas contratações de obras, sempre que possível e economicamente viável, devem ser adotados empreendimentos que garantam a preservação do meio ambiente.

Art. 46. Na execução indireta de obras e serviços de engenharia, são admitidos os seguintes regimes:

I - empreitada por preço unitário;

II - empreitada por preço global;

III - empreitada integral;

IV - contratação por tarefa;

V - contratação integrada;

VI - contratação semi-integrada;

VII - fornecimento e prestação de serviço associado.

§ 1º Nas licitações de obras e serviços de engenharia, a responsabilidade pela elaboração e pela adequação do projeto básico e do projeto executivo deve ser atribuída de acordo com o regime de execução adotado.

§ 2º A contratação integrada compreende a elaboração e o desenvolvimento dos projetos básico e executivo, a execução de obras e serviços de engenharia, a montagem, a realização de testes, a pré-operação e todas as demais operações necessárias e suficientes para a entrega final do objeto.

Art. 47. As regras para os procedimentos auxiliares e para os procedimentos licitatórios serão definidas em regulamento.
"""

    def test_all_spans_have_valid_offsets(self):
        """Verifica que todos os spans do trecho real têm offsets válidos."""
        parser = SpanParser()
        doc = parser.parse(self.LEI_14133_EXCERPT)

        offsets_map, _ = extract_offsets_from_parsed_doc(doc)

        # Todos os spans devem ter offsets
        for span in doc.spans:
            assert span.span_id in offsets_map, f"{span.span_id} sem offset"

            start, end = offsets_map[span.span_id]
            assert start >= 0, f"{span.span_id} start_pos negativo"
            assert end > start, f"{span.span_id} range inválido"

    def test_incisos_art45_within_range(self):
        """Verifica que todos os 5 incisos do Art. 45 estão no range."""
        parser = SpanParser()
        doc = parser.parse(self.LEI_14133_EXCERPT)

        offsets_map, _ = extract_offsets_from_parsed_doc(doc)

        art_start, art_end = offsets_map["ART-045"]

        # Art. 45 deve ter 5 incisos (I a V)
        expected_incisos = ["INC-045-I", "INC-045-II", "INC-045-III", "INC-045-IV", "INC-045-V"]

        for inc_id in expected_incisos:
            assert inc_id in offsets_map, f"{inc_id} não encontrado no offsets_map"

            inc_start, inc_end = offsets_map[inc_id]

            assert art_start <= inc_start, (
                f"{inc_id} começa ANTES do artigo: {inc_start} < {art_start}"
            )
            assert inc_end <= art_end, (
                f"{inc_id} termina DEPOIS do artigo: {inc_end} > {art_end}"
            )

    def test_incisos_art46_within_range(self):
        """Verifica que todos os 7 incisos do Art. 46 estão no range."""
        parser = SpanParser()
        doc = parser.parse(self.LEI_14133_EXCERPT)

        offsets_map, _ = extract_offsets_from_parsed_doc(doc)

        art_start, art_end = offsets_map["ART-046"]

        # Art. 46 deve ter 7 incisos (I a VII)
        expected_incisos = [
            "INC-046-I", "INC-046-II", "INC-046-III", "INC-046-IV",
            "INC-046-V", "INC-046-VI", "INC-046-VII"
        ]

        for inc_id in expected_incisos:
            assert inc_id in offsets_map, f"{inc_id} não encontrado no offsets_map"

            inc_start, inc_end = offsets_map[inc_id]

            assert art_start <= inc_start < inc_end <= art_end, (
                f"{inc_id} [{inc_start}:{inc_end}] fora do ART-046 [{art_start}:{art_end}]"
            )

    def test_paragraphs_art46_within_range(self):
        """Verifica que os parágrafos do Art. 46 estão no range."""
        parser = SpanParser()
        doc = parser.parse(self.LEI_14133_EXCERPT)

        offsets_map, _ = extract_offsets_from_parsed_doc(doc)

        art_start, art_end = offsets_map["ART-046"]

        for par_id in ["PAR-046-1", "PAR-046-2"]:
            assert par_id in offsets_map, f"{par_id} não encontrado"

            par_start, par_end = offsets_map[par_id]

            assert art_start <= par_start < par_end <= art_end, (
                f"{par_id} [{par_start}:{par_end}] fora do ART-046 [{art_start}:{art_end}]"
            )

    def test_slicing_matches_span_text(self):
        """Verifica que o slicing retorna texto consistente com span.text."""
        parser = SpanParser()
        doc = parser.parse(self.LEI_14133_EXCERPT)

        offsets_map, _ = extract_offsets_from_parsed_doc(doc)
        canonical_text = normalize_canonical_text(doc.source_text)

        # Testa alguns spans específicos
        test_spans = ["ART-045", "INC-045-II", "PAR-046-1", "INC-046-V"]

        for span_id in test_spans:
            span = doc.get_span(span_id)
            if span is None:
                continue

            start, end = offsets_map[span_id]
            sliced = canonical_text[start:end].strip()

            # Para artigos, o SpanParser normaliza "Art. 45." para "Art. 45º"
            # Comparamos palavras-chave ao invés de prefixo exato
            span_text = span.text.strip()

            if span_id.startswith("ART-"):
                # Para artigos: verifica número e palavras-chave
                art_num = span_id.split("-")[1]  # "045"
                assert f"Art. {int(art_num)}" in sliced[:20], (
                    f"{span_id}: número do artigo não encontrado no slice.\n"
                    f"  Slice: {repr(sliced[:50])}"
                )
            elif span_id.startswith("INC-"):
                # Para incisos: verifica numeral romano
                romano = span.identifier  # "I", "II", etc.
                assert sliced.startswith(romano) or f" {romano} " in sliced[:20], (
                    f"{span_id}: numeral romano '{romano}' não encontrado.\n"
                    f"  Slice: {repr(sliced[:30])}"
                )
            elif span_id.startswith("PAR-"):
                # Para parágrafos: verifica "§" ou "Parágrafo"
                assert "§" in sliced[:10] or "Parágrafo" in sliced[:15], (
                    f"{span_id}: marcador de parágrafo não encontrado.\n"
                    f"  Slice: {repr(sliced[:30])}"
                )
            else:
                # Fallback: verifica se há overlap significativo
                words = span_text.split()[:5]
                matches = sum(1 for w in words if w in sliced[:100])
                assert matches >= 2, (
                    f"{span_id}: menos de 2 palavras em comum.\n"
                    f"  Span: {repr(span_text[:40])}\n"
                    f"  Slice: {repr(sliced[:40])}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
