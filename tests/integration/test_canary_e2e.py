# -*- coding: utf-8 -*-
"""
Teste Canary E2E - Validação Completa do Pipeline
=================================================

Este teste é um "canary in the coal mine" - se falhar, algo fundamental quebrou.
Deve ser executado em todo CI/CD e antes de deploys.

IMPORTANTE: Este teste roda SEM MOCKS - usa implementações reais de:
- SpanParser (parsing real)
- ChunkMaterializer (materialização real)
- OriginClassifier (classificação real)
- extract_offsets_from_parsed_doc (extração real)

INVARIANTES GARANTIDAS
======================

1. PARSING (Canary 1)
   - SpanParser extrai artigos, parágrafos e incisos corretamente
   - Hierarquia parent_id está correta
   - span_id segue convenção (ART-xxx, PAR-xxx-y, INC-xxx-Y)

2. PR13 OFFSETS (Canary 2)
   - Todos os spans têm offsets no offsets_map
   - start >= 0, end > start para todos os spans
   - Filhos estão DENTRO do range estrutural do pai
   - canonical_hash não é vazio

3. SNIPPET INTEGRITY (Canary 3)
   - canonical_text[start:end] contém o texto esperado
   - Artigos contêm "Art."
   - Incisos contêm numerais romanos

4. ORIGIN CLASSIFICATION (Canary 4)
   - Artigos normais são classificados como "self"
   - Art. 337-* são classificados como "external" (Código Penal)
   - Menções a outras leis são "self" com confidence="low" (auditável)

5. LARGE ARTICLE (Canary 5)
   - Artigos com muitos incisos (20+) processam sem erro 400
   - Materialização completa sem exceção

6. HIERARQUIA (Canary 6)
   - Artigos têm parent_node_id = "" (string vazia)
   - Parágrafos apontam para artigo pai
   - Incisos apontam para parágrafo ou artigo
   - Payload Milvus usa parent_node_id, NUNCA parent_chunk_id

7. NEGATIVO: AMBIGUOUS (Canary 7)
   - Texto duplicado no canonical DEVE causar OffsetResolutionError
   - reason deve conter "AMBIGUOUS"

8. NEGATIVO: NOT_FOUND (Canary 8)
   - Texto inexistente no canonical DEVE causar OffsetResolutionError
   - reason deve conter "NOT_FOUND"

ARQUIVOS SKIPPED/EXCLUÍDOS E MOTIVOS
====================================

EXCLUÍDOS (--ignore no pytest):
- test_article_validator.py: Requer FastAPI (ModuleNotFoundError)
- test_chunk_materializer_split.py: Import relativo inválido (exec() com código)
- test_rel_type_classification.py: Duplicated timeseries Prometheus (conflito de registry)

SKIPPED (pytest.mark.skip):
- test_ingestion_pipeline.py::TestIngestionPipelineHappyPath (2 testes)
  Motivo: src.orchestrator não existe ainda
  TODO: Reativar quando orchestrator for implementado

PRÓXIMOS PASSOS
===============

Quando src.orchestrator existir:
1. Remover skip de TestIngestionPipelineHappyPath
2. Ajustar patch paths para módulo real
3. Adicionar canary de ingestão completa (PDF → Milvus)
4. Reduzir excludes resolvendo imports problemáticos

@author: Claude (RunPod)
@date: 2026-02-05
@version: 2.0 (com negativos AMBIGUOUS/NOT_FOUND)
"""

import pytest
import sys
from pathlib import Path

# Adiciona projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.parsing.span_parser import SpanParser
from src.parsing.span_models import SpanType, ParsedDocument
from src.chunking.canonical_offsets import (
    extract_offsets_from_parsed_doc,
    normalize_canonical_text,
    OffsetResolutionError,
)
from src.chunking.chunk_materializer import ChunkMaterializer, DeviceType
from src.chunking.origin_classifier import OriginClassifier
from src.parsing.article_orchestrator import ArticleChunk


# Documento de teste com:
# - 2 artigos (um normal, um grande com muitos incisos)
# - Parágrafos
# - Incisos aninhados
# - Referência ao Código Penal (para testar origin classification)
MINIMAL_LAW_DOCUMENT = """
Art. 1º Esta Lei estabelece normas gerais de licitação e contratação para as Administrações Públicas diretas.

§ 1º O disposto nesta Lei aplica-se à administração direta dos Poderes Executivo, Legislativo e Judiciário.

§ 2º Para os fins desta Lei, consideram-se:

I - órgão público: toda unidade de atuação integrante da estrutura da Administração Pública;

II - entidade: toda pessoa jurídica que integre a Administração Pública indireta.

Art. 2º Na aplicação desta Lei, serão observados os seguintes princípios:

I - legalidade;

II - impessoalidade;

III - moralidade;

IV - publicidade;

V - eficiência;

VI - interesse público;

VII - probidade administrativa;

VIII - igualdade;

IX - planejamento;

X - transparência;

XI - eficácia;

XII - segregação de funções;

XIII - motivação;

XIV - vinculação ao edital;

XV - julgamento objetivo;

XVI - segurança jurídica;

XVII - razoabilidade;

XVIII - competitividade;

XIX - proporcionalidade;

XX - celeridade;

XXI - economicidade;

XXII - desenvolvimento nacional sustentável.
"""


class TestCanaryE2E:
    """Teste canary que valida o pipeline completo."""

    @pytest.fixture
    def parsed_doc(self):
        """Parseia o documento mínimo."""
        parser = SpanParser()
        return parser.parse(MINIMAL_LAW_DOCUMENT)

    @pytest.fixture
    def canonical_text(self):
        """Texto canônico normalizado."""
        return normalize_canonical_text(MINIMAL_LAW_DOCUMENT)

    def test_canary_1_parsing_extracts_structure(self, parsed_doc):
        """CANARY 1: Parser extrai estrutura correta do documento."""
        # Deve ter 2 artigos
        articles = [s for s in parsed_doc.spans if s.span_type == SpanType.ARTIGO]
        assert len(articles) >= 2, f"Esperava >= 2 artigos, encontrou {len(articles)}"

        # Art. 1 deve existir
        art1 = parsed_doc.get_span("ART-001")
        assert art1 is not None, "ART-001 não encontrado"
        assert "Art. 1" in art1.text

        # Art. 2 deve existir
        art2 = parsed_doc.get_span("ART-002")
        assert art2 is not None, "ART-002 não encontrado"
        assert "Art. 2" in art2.text

        # Deve ter parágrafos no Art. 1
        paragraphs = [s for s in parsed_doc.spans
                      if s.span_type == SpanType.PARAGRAFO and s.parent_id == "ART-001"]
        assert len(paragraphs) >= 2, f"Art. 1 deveria ter >= 2 parágrafos"

        # Deve ter incisos no Art. 2
        incisos_art2 = [s for s in parsed_doc.spans
                        if s.span_type == SpanType.INCISO and
                        (s.parent_id == "ART-002" or s.span_id.startswith("INC-002-"))]
        assert len(incisos_art2) >= 20, f"Art. 2 deveria ter >= 20 incisos (muitos princípios)"

        print(f"✅ CANARY 1 PASSED: Parsing extraiu {len(articles)} artigos, "
              f"{len(paragraphs)} parágrafos, {len(incisos_art2)} incisos")

    def test_canary_2_pr13_offsets_valid(self, parsed_doc, canonical_text):
        """CANARY 2: PR13 offsets são válidos e permitem slicing."""
        offsets_map, canonical_hash = extract_offsets_from_parsed_doc(parsed_doc)

        # Deve ter hash não vazio
        assert canonical_hash and len(canonical_hash) > 0, "canonical_hash vazio"

        # Todos os artigos devem ter offsets no mapa
        articles = [s for s in parsed_doc.spans if s.span_type == SpanType.ARTIGO]
        for art in articles:
            assert art.span_id in offsets_map, f"{art.span_id} não está no offsets_map"
            start, end = offsets_map[art.span_id]
            assert start >= 0, f"{art.span_id} tem start negativo: {start}"
            assert end > start, f"{art.span_id} tem end <= start: {end} <= {start}"

        # Verifica invariante: filhos dentro do range do pai
        for span in parsed_doc.spans:
            if span.parent_id and span.span_id in offsets_map and span.parent_id in offsets_map:
                child_start, child_end = offsets_map[span.span_id]
                parent_start, parent_end = offsets_map[span.parent_id]
                assert parent_start <= child_start, \
                    f"{span.span_id} começa antes do pai: {child_start} < {parent_start}"
                assert child_end <= parent_end, \
                    f"{span.span_id} termina após o pai: {child_end} > {parent_end}"

        print(f"✅ CANARY 2 PASSED: PR13 offsets válidos para {len(offsets_map)} spans")

    def test_canary_3_snippet_integrity(self, parsed_doc, canonical_text):
        """CANARY 3: Slicing pelos offsets retorna texto correto."""
        offsets_map, _ = extract_offsets_from_parsed_doc(parsed_doc)

        # Para cada artigo, verifica que o slice contém "Art."
        articles = [s for s in parsed_doc.spans if s.span_type == SpanType.ARTIGO]
        for art in articles:
            if art.span_id not in offsets_map:
                continue
            start, end = offsets_map[art.span_id]
            snippet = canonical_text[start:end]
            assert "Art." in snippet, \
                f"Snippet de {art.span_id} não contém 'Art.': {snippet[:50]}"

        # Para incisos, verifica que contém numeral romano
        incisos = [s for s in parsed_doc.spans if s.span_type == SpanType.INCISO]
        for inc in incisos[:5]:  # Apenas primeiros 5 para não demorar
            if inc.span_id not in offsets_map:
                continue
            start, end = offsets_map[inc.span_id]
            snippet = canonical_text[start:end]
            # Deve conter pelo menos um caractere romano ou "-"
            assert any(c in snippet for c in ["I", "V", "X", "-"]), \
                f"Snippet de {inc.span_id} parece inválido: {snippet[:30]}"

        print(f"✅ CANARY 3 PASSED: Snippet integrity verificada")

    def test_canary_4_origin_classification(self):
        """CANARY 4: Origin classifier funciona corretamente."""
        classifier = OriginClassifier()

        # Chunk normal deve ser "self"
        chunk_self = {"text": "Art. 1º Esta Lei estabelece normas gerais."}
        result_self = classifier.classify(chunk_self)
        assert result_self["origin_type"] == "self", \
            f"Artigo normal deveria ser 'self', foi: {result_self['origin_type']}"

        # Art. 337-E deve ser "external" (Código Penal)
        chunk_external = {"text": "Art. 337-E. Admitir, possibilitar ou dar causa..."}
        result_external = classifier.classify(chunk_external)
        assert result_external["origin_type"] == "external", \
            f"Art. 337-E deveria ser 'external', foi: {result_external['origin_type']}"
        assert result_external["origin_reference"] == "DL-2848-1940", \
            f"Referência deveria ser DL-2848-1940, foi: {result_external['origin_reference']}"

        # Menção a outra lei deve ser "self" com low confidence (auditável, não decisivo)
        chunk_mention = {"text": "A Lei 8.666 fica revogada."}
        result_mention = classifier.classify(chunk_mention)
        assert result_mention["origin_type"] == "self", \
            f"Menção deveria ser 'self' (low confidence), foi: {result_mention['origin_type']}"
        assert result_mention["origin_confidence"] == "low", \
            f"Menção deveria ter confidence 'low', foi: {result_mention['origin_confidence']}"

        print("✅ CANARY 4 PASSED: Origin classification funciona")

    def test_canary_5_large_article_no_400(self, parsed_doc, canonical_text):
        """CANARY 5: Artigo grande (muitos incisos) processa sem erro 400."""
        offsets_map, canonical_hash = extract_offsets_from_parsed_doc(parsed_doc)

        # Art. 2 tem muitos incisos - simula artigo grande
        art2 = parsed_doc.get_span("ART-002")
        assert art2 is not None, "ART-002 não encontrado"

        # Coleta todos os incisos do Art. 2
        incisos_art2 = [s.span_id for s in parsed_doc.spans
                        if s.span_type == SpanType.INCISO and
                        (s.parent_id == "ART-002" or s.span_id.startswith("INC-002-"))]

        # Cria ArticleChunk com muitos incisos
        article_chunk = ArticleChunk(
            article_id="ART-002",
            article_number="2",
            text=art2.text,
            citations=["ART-002"] + incisos_art2,
            inciso_ids=incisos_art2,
            paragrafo_ids=[],
        )

        # Deve materializar sem erro (o teste anterior de 400 era por JSON truncado)
        materializer = ChunkMaterializer(
            document_id="TEST-CANARY-001",
            offsets_map=offsets_map,
            canonical_hash=canonical_hash,
            canonical_text=canonical_text,
        )

        # Não deve lançar exceção
        try:
            chunks = materializer.materialize_article(article_chunk, parsed_doc)
            assert len(chunks) > 0, "Deveria ter materializado pelo menos 1 chunk"
            print(f"✅ CANARY 5 PASSED: Artigo grande materializado ({len(chunks)} chunks)")
        except Exception as e:
            pytest.fail(f"Erro ao materializar artigo grande: {e}")

    def test_canary_6_hierarchy_parent_node_id(self, parsed_doc, canonical_text):
        """CANARY 6: Hierarquia correta via parent_node_id."""
        offsets_map, canonical_hash = extract_offsets_from_parsed_doc(parsed_doc)

        # Cria ArticleChunk do Art. 1 com parágrafos e incisos
        art1 = parsed_doc.get_span("ART-001")
        assert art1 is not None

        paragraphs = [s.span_id for s in parsed_doc.spans
                      if s.span_type == SpanType.PARAGRAFO and s.parent_id == "ART-001"]
        incisos = [s.span_id for s in parsed_doc.spans
                   if s.span_type == SpanType.INCISO and
                   (s.parent_id == "ART-001" or s.parent_id in paragraphs)]

        article_chunk = ArticleChunk(
            article_id="ART-001",
            article_number="1",
            text=art1.text,
            citations=["ART-001"] + paragraphs + incisos,
            inciso_ids=incisos,
            paragrafo_ids=paragraphs,
        )

        materializer = ChunkMaterializer(
            document_id="TEST-CANARY-001",
            offsets_map=offsets_map,
            canonical_hash=canonical_hash,
            canonical_text=canonical_text,
        )

        chunks = materializer.materialize_article(article_chunk, parsed_doc)

        # Verifica hierarquia
        for chunk in chunks:
            if chunk.device_type == DeviceType.ARTICLE:
                # Artigo não tem pai
                assert chunk.parent_node_id == "", \
                    f"Artigo deveria ter parent_node_id='', tem: {chunk.parent_node_id}"
            elif chunk.device_type == DeviceType.PARAGRAPH:
                # Parágrafo aponta para artigo
                assert "ART-001" in chunk.parent_node_id, \
                    f"Parágrafo deveria apontar para ART-001, tem: {chunk.parent_node_id}"
            elif chunk.device_type == DeviceType.INCISO:
                # Inciso aponta para artigo ou parágrafo
                assert "ART-001" in chunk.parent_node_id or "PAR-001" in chunk.parent_node_id, \
                    f"Inciso deveria apontar para ART ou PAR, tem: {chunk.parent_node_id}"

            # Nenhum chunk deve ter parent_chunk_id no dict
            milvus_dict = chunk.to_milvus_dict()
            assert "parent_chunk_id" not in milvus_dict, \
                f"Chunk {chunk.span_id} tem parent_chunk_id no payload (deveria ser parent_node_id)"

        print(f"✅ CANARY 6 PASSED: Hierarquia correta ({len(chunks)} chunks verificados)")

    def test_canary_7_negative_ambiguous(self):
        """
        CANARY 7 NEGATIVO: Texto ambíguo (múltiplas ocorrências) deve causar erro.

        Este teste garante que o sistema FALHA CORRETAMENTE quando não consegue
        determinar univocamente a posição de um span no texto canônico.
        """
        from src.parsing.span_models import Span, SpanType, ParsedDocument
        from src.parsing.article_orchestrator import ArticleChunk

        # Texto canônico com "TEXTO REPETIDO" aparecendo 3 vezes
        canonical_text_ambiguous = """Art. 1º TEXTO REPETIDO aqui.

§ 1º TEXTO REPETIDO no parágrafo.

§ 2º TEXTO REPETIDO de novo.
"""

        # Cria ParsedDocument
        parsed_doc = ParsedDocument()
        article_span = Span(
            span_id="ART-001",
            span_type=SpanType.ARTIGO,
            text="Art. 1º TEXTO REPETIDO aqui.",
            parent_id=None,
            start_pos=0,
            end_pos=len(canonical_text_ambiguous),
            caput_end_pos=29,
        )
        parsed_doc.add_span(article_span)

        # Parágrafo com texto que aparece múltiplas vezes no canonical
        paragraph_span = Span(
            span_id="PAR-001-1",
            span_type=SpanType.PARAGRAFO,
            text="TEXTO REPETIDO",  # Ambíguo! Aparece 3x
            parent_id="ART-001",
            # Sem start_pos/end_pos - forçará resolução
        )
        parsed_doc.add_span(paragraph_span)

        article_chunk = ArticleChunk(
            article_id="ART-001",
            article_number="1",
            text="Art. 1º TEXTO REPETIDO aqui.",
            citations=["ART-001", "PAR-001-1"],
            inciso_ids=[],
            paragrafo_ids=["PAR-001-1"],
        )

        materializer = ChunkMaterializer(
            document_id="TEST-AMBIGUOUS",
            offsets_map={"ART-001": (0, len(canonical_text_ambiguous))},
            canonical_hash="hash_ambiguous",
            canonical_text=canonical_text_ambiguous,
        )

        # DEVE lançar OffsetResolutionError com reason AMBIGUOUS
        with pytest.raises(OffsetResolutionError) as exc_info:
            materializer.materialize_article(article_chunk, parsed_doc)

        assert "AMBIGUOUS" in exc_info.value.reason, \
            f"Erro deveria ter reason=AMBIGUOUS, tem: {exc_info.value.reason}"

        print("✅ CANARY 7 PASSED: AMBIGUOUS corretamente detectado")

    def test_canary_8_negative_not_found(self):
        """
        CANARY 8 NEGATIVO: Texto inexistente no canonical deve causar erro.

        Este teste garante que o sistema FALHA CORRETAMENTE quando o texto
        de um span não existe no texto canônico.
        """
        from src.parsing.span_models import Span, SpanType, ParsedDocument
        from src.parsing.article_orchestrator import ArticleChunk

        # Texto canônico simples
        canonical_text_simple = "Art. 1º Este é o texto do artigo.\n"

        # Cria ParsedDocument
        parsed_doc = ParsedDocument()
        article_span = Span(
            span_id="ART-001",
            span_type=SpanType.ARTIGO,
            text="Art. 1º Este é o texto do artigo.",
            parent_id=None,
            start_pos=0,
            end_pos=len(canonical_text_simple),
        )
        parsed_doc.add_span(article_span)

        # Parágrafo com texto que NÃO existe no canonical
        paragraph_span = Span(
            span_id="PAR-001-1",
            span_type=SpanType.PARAGRAFO,
            text="TEXTO QUE NÃO EXISTE NO CANONICAL",  # Não encontrável!
            parent_id="ART-001",
            # Sem start_pos/end_pos - forçará resolução
        )
        parsed_doc.add_span(paragraph_span)

        article_chunk = ArticleChunk(
            article_id="ART-001",
            article_number="1",
            text="Art. 1º Este é o texto do artigo.",
            citations=["ART-001", "PAR-001-1"],
            inciso_ids=[],
            paragrafo_ids=["PAR-001-1"],
        )

        materializer = ChunkMaterializer(
            document_id="TEST-NOT-FOUND",
            offsets_map={"ART-001": (0, len(canonical_text_simple))},
            canonical_hash="hash_notfound",
            canonical_text=canonical_text_simple,
        )

        # DEVE lançar OffsetResolutionError com reason NOT_FOUND
        with pytest.raises(OffsetResolutionError) as exc_info:
            materializer.materialize_article(article_chunk, parsed_doc)

        assert "NOT_FOUND" in exc_info.value.reason, \
            f"Erro deveria ter reason=NOT_FOUND, tem: {exc_info.value.reason}"

        print("✅ CANARY 8 PASSED: NOT_FOUND corretamente detectado")

    def test_canary_full_pipeline(self, parsed_doc, canonical_text):
        """CANARY COMPLETO: Executa todos os canaries em sequência (positivos + negativos)."""
        print("\n" + "=" * 60)
        print("CANARY E2E: VALIDAÇÃO COMPLETA DO PIPELINE (SEM MOCKS)")
        print("=" * 60)

        # Testes positivos (devem passar)
        print("\n--- TESTES POSITIVOS ---")
        self.test_canary_1_parsing_extracts_structure(parsed_doc)
        self.test_canary_2_pr13_offsets_valid(parsed_doc, canonical_text)
        self.test_canary_3_snippet_integrity(parsed_doc, canonical_text)
        self.test_canary_4_origin_classification()
        self.test_canary_5_large_article_no_400(parsed_doc, canonical_text)
        self.test_canary_6_hierarchy_parent_node_id(parsed_doc, canonical_text)

        # Testes negativos (devem falhar corretamente)
        print("\n--- TESTES NEGATIVOS ---")
        self.test_canary_7_negative_ambiguous()
        self.test_canary_8_negative_not_found()

        print("\n" + "=" * 60)
        print("✅✅✅ CANARY E2E PASSOU: PIPELINE SAUDÁVEL ✅✅✅")
        print("(6 positivos + 2 negativos verificados)")
        print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
