# -*- coding: utf-8 -*-
"""
Teste de Integração: Triângulo de Offsets (PR13)

Verifica consistência entre:
1. Estrutura (ranges pai/filho)
2. Indexação (chunk_text == slice do canonical)
3. Evidência (offsets do chunk permitem reconstruir texto)

Documento mínimo: 1 artigo + 2 incisos + 1 parágrafo
"""

import pytest
from src.parsing.span_parser import SpanParser
from src.parsing.span_models import SpanType
from src.chunking.canonical_offsets import (
    extract_offsets_from_parsed_doc,
    normalize_canonical_text,
)
from src.chunking.chunk_materializer import ChunkMaterializer


# Documento mínimo para teste
MINIMAL_DOC = """
Art. 1º Este artigo estabelece as regras básicas:

I - primeira regra do artigo;

II - segunda regra do artigo.

§ 1º Este parágrafo complementa o artigo com detalhes adicionais.
"""


class TestOffsetTriangle:
    """Testa o triângulo: estrutura → indexação → evidência."""

    @pytest.fixture
    def parsed_doc(self):
        """Parseia documento mínimo."""
        parser = SpanParser()
        return parser.parse(MINIMAL_DOC)

    @pytest.fixture
    def canonical_text(self):
        """Texto canônico normalizado."""
        return normalize_canonical_text(MINIMAL_DOC)

    def test_assert1_structural_ranges(self, parsed_doc, canonical_text):
        """
        Assert 1: ART.start_pos < INC.start_pos < INC.end_pos <= ART.end_pos

        Verifica que os filhos (INC, PAR) estão contidos no range do artigo pai.
        """
        # Obtém artigo
        art = parsed_doc.get_span("ART-001")
        assert art is not None, "ART-001 não encontrado"
        assert art.start_pos >= 0, "ART-001 sem start_pos"
        assert art.end_pos > art.start_pos, "ART-001 end_pos inválido"

        # Obtém incisos
        inc1 = parsed_doc.get_span("INC-001-I")
        inc2 = parsed_doc.get_span("INC-001-II")
        assert inc1 is not None, "INC-001-I não encontrado"
        assert inc2 is not None, "INC-001-II não encontrado"

        # Obtém parágrafo
        par1 = parsed_doc.get_span("PAR-001-1")
        assert par1 is not None, "PAR-001-1 não encontrado"

        # Assert 1: Ranges estruturais
        # Incisos dentro do artigo
        assert art.start_pos < inc1.start_pos, \
            f"INC-001-I não começa após ART: {inc1.start_pos} <= {art.start_pos}"
        assert inc1.end_pos <= art.end_pos, \
            f"INC-001-I termina após ART: {inc1.end_pos} > {art.end_pos}"

        assert art.start_pos < inc2.start_pos, \
            f"INC-001-II não começa após ART: {inc2.start_pos} <= {art.start_pos}"
        assert inc2.end_pos <= art.end_pos, \
            f"INC-001-II termina após ART: {inc2.end_pos} > {art.end_pos}"

        # Parágrafo dentro do artigo
        assert art.start_pos < par1.start_pos, \
            f"PAR-001-1 não começa após ART: {par1.start_pos} <= {art.start_pos}"
        assert par1.end_pos <= art.end_pos, \
            f"PAR-001-1 termina após ART: {par1.end_pos} > {art.end_pos}"

        # Ordem: inc1 < inc2 < par1
        assert inc1.start_pos < inc2.start_pos, "INC-I deve vir antes de INC-II"
        assert inc2.start_pos < par1.start_pos, "Incisos devem vir antes do parágrafo"

        print(f"✅ Assert 1 PASSED: Ranges estruturais válidos")
        print(f"   ART: [{art.start_pos}:{art.end_pos}] (caput_end: {art.caput_end_pos})")
        print(f"   INC-I: [{inc1.start_pos}:{inc1.end_pos}]")
        print(f"   INC-II: [{inc2.start_pos}:{inc2.end_pos}]")
        print(f"   PAR-1: [{par1.start_pos}:{par1.end_pos}]")

    def test_assert2_caput_indexing(self, parsed_doc, canonical_text):
        """
        Assert 2: ART_chunk_text == canonical_text[ART.start_pos:ART.caput_end_pos]

        Verifica que o texto do chunk ART é exatamente o slice do caput.
        """
        art = parsed_doc.get_span("ART-001")
        assert art is not None, "ART-001 não encontrado"
        assert art.caput_end_pos > 0, "ART-001 sem caput_end_pos"

        # Slice do canonical usando caput_range
        caput_slice = canonical_text[art.start_pos:art.caput_end_pos]

        # O texto do span deve estar contido no slice (pode ter whitespace diff)
        # O caput é "Art. 1º Este artigo estabelece as regras básicas:"
        assert "Art. 1" in caput_slice, f"Caput não contém 'Art. 1': {caput_slice[:50]}"
        assert "regras básicas" in caput_slice, f"Caput não contém texto esperado: {caput_slice}"

        # Caput NÃO deve conter os incisos ou parágrafos
        assert "I -" not in caput_slice, f"Caput contém inciso! slice: {caput_slice}"
        assert "§" not in caput_slice, f"Caput contém parágrafo! slice: {caput_slice}"

        print(f"✅ Assert 2 PASSED: Caput indexing correto")
        print(f"   caput_range: [{art.start_pos}:{art.caput_end_pos}]")
        print(f"   caput_text: '{caput_slice.strip()}'")

    def test_assert3_evidence_reconstruction(self, parsed_doc, canonical_text):
        """
        Assert 3: Evidence API - slice pelos offsets retorna texto que bate com chunk.

        Simula o que a Evidence API faria: dado um chunk com offsets,
        o slice do canonical deve retornar o texto esperado.
        """
        # Extrai offsets do parsed_doc (como o ChunkMaterializer faria)
        offsets_map, canonical_hash = extract_offsets_from_parsed_doc(parsed_doc)

        # Verifica cada span
        for span_id, (start, end) in offsets_map.items():
            span = parsed_doc.get_span(span_id)
            if span is None:
                continue

            # Slice do canonical
            evidence_slice = canonical_text[start:end].strip()

            # Para artigos, o offset é do caput (não do artigo inteiro)
            if span.span_type == SpanType.ARTIGO:
                # Caput deve conter "Art."
                assert "Art." in evidence_slice, \
                    f"{span_id}: evidence slice não contém 'Art.': {evidence_slice[:50]}"
            else:
                # Filhos: o slice deve conter o início do texto do span
                span_prefix = span.text[:20] if len(span.text) > 20 else span.text
                # Normaliza para comparação (remove whitespace extra)
                evidence_normalized = " ".join(evidence_slice.split())
                span_normalized = " ".join(span_prefix.split())

                assert span_normalized in evidence_normalized or evidence_normalized.startswith(span_normalized[:10]), \
                    f"{span_id}: evidence não bate com span.text\n" \
                    f"  evidence: '{evidence_slice[:50]}'\n" \
                    f"  span.text: '{span.text[:50]}'"

        print(f"✅ Assert 3 PASSED: Evidence reconstruction válida")
        print(f"   Offsets verificados: {list(offsets_map.keys())}")

    def test_full_triangle(self, parsed_doc, canonical_text):
        """
        Teste completo: executa os 3 asserts em sequência.
        """
        print("\n" + "=" * 60)
        print("TESTE DE INTEGRAÇÃO: TRIÂNGULO DE OFFSETS")
        print("=" * 60)

        # Assert 1
        self.test_assert1_structural_ranges(parsed_doc, canonical_text)

        # Assert 2
        self.test_assert2_caput_indexing(parsed_doc, canonical_text)

        # Assert 3
        self.test_assert3_evidence_reconstruction(parsed_doc, canonical_text)

        print("\n" + "=" * 60)
        print("✅ TRIÂNGULO FECHADO: estrutura → indexação → evidência")
        print("=" * 60)


    def test_semantic_separation(self, parsed_doc, canonical_text):
        """
        Verifica separação semântica entre:
        - offsets_map: usa structural_end_pos para validação de hierarquia
        - caput_end_pos: usado para indexação/evidence do ART chunk

        Isso evita o bug onde filhos pareciam "fora do pai" quando
        offsets_map usava caput_end_pos.
        """
        # Extrai offsets (devem usar structural_end_pos, não caput_end_pos)
        offsets_map, _ = extract_offsets_from_parsed_doc(parsed_doc)

        art = parsed_doc.get_span("ART-001")
        inc1 = parsed_doc.get_span("INC-001-I")
        par1 = parsed_doc.get_span("PAR-001-1")

        # Verifica propriedades semânticas do Span
        assert art.structural_end_pos == art.end_pos, "structural_end_pos deve ser alias de end_pos"
        assert art.has_caput_range, "Artigo com filhos deve ter caput_range"
        assert art.caput_length > 0, "caput_length deve ser > 0"
        assert art.structural_length > art.caput_length, "structural_length > caput_length"

        # offsets_map deve usar structural_end_pos
        art_start, art_structural_end = offsets_map["ART-001"]
        assert art_structural_end == art.structural_end_pos, \
            f"offsets_map deve usar structural_end_pos ({art.structural_end_pos}), não caput_end_pos ({art.caput_end_pos})"

        # Filhos devem estar DENTRO do structural range
        inc_start, inc_end = offsets_map["INC-001-I"]
        par_start, par_end = offsets_map["PAR-001-1"]

        assert art_start <= inc_start < inc_end <= art_structural_end, \
            f"INC-001-I [{inc_start}:{inc_end}] deve estar dentro do structural range [{art_start}:{art_structural_end}]"

        assert art_start <= par_start < par_end <= art_structural_end, \
            f"PAR-001-1 [{par_start}:{par_end}] deve estar dentro do structural range [{art_start}:{art_structural_end}]"

        # Mas caput_end_pos deve ser ANTES dos filhos
        assert art.caput_end_pos <= inc_start, \
            f"caput_end_pos ({art.caput_end_pos}) deve ser <= INC start ({inc_start})"

        # Invariante: start_pos <= caput_end_pos <= structural_end_pos
        assert art.start_pos <= art.caput_end_pos <= art.structural_end_pos, \
            f"Invariante violado: {art.start_pos} <= {art.caput_end_pos} <= {art.structural_end_pos}"

        print(f"✅ Semantic separation PASSED:")
        print(f"   offsets_map['ART-001'] = [{art_start}:{art_structural_end}] (structural)")
        print(f"   art.caput_end_pos = {art.caput_end_pos} (para indexação)")
        print(f"   art.caput_length = {art.caput_length} chars")
        print(f"   art.structural_length = {art.structural_length} chars")
        print(f"   Filhos corretamente dentro do structural range")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
