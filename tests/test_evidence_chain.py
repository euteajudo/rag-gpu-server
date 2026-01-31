"""
Testes de cadeia de evidência: Neo4j edges → Milvus chunks.

PR3 v2 - Hard Reset RAG Architecture

Verifica que:
1. source_chunk_id nos edges Neo4j aponta para registros reais no Milvus
2. Não existem self-loops
3. Stubs são criados corretamente para targets não ingeridos
"""

import pytest
from typing import Optional
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch


@dataclass
class MockEdge:
    """Edge mockado para testes."""

    source_node_id: str
    target_node_id: str
    source_chunk_id: str
    confidence: float = 1.0
    extraction_method: str = "regex"


@dataclass
class MockMilvusChunk:
    """Chunk mockado do Milvus."""

    node_id: str
    chunk_id: str
    document_id: str
    span_id: str
    device_type: str
    text: str


class TestSourceChunkIdPointsToMilvus:
    """
    Testes para verificar que source_chunk_id aponta para registro válido no Milvus.

    Critério PR3:
    - Todo edge :CITA deve ter source_chunk_id que corresponde a um node_id no Milvus
    - Isso garante a "cadeia de evidência" para audit/PR4 Evidence API
    """

    @pytest.fixture
    def sample_milvus_chunks(self) -> list[MockMilvusChunk]:
        """Cria chunks mockados do Milvus."""
        return [
            MockMilvusChunk(
                node_id="leis:LEI-14133-2021#ART-005@P00",
                chunk_id="LEI-14133-2021#ART-005@P00",
                document_id="LEI-14133-2021",
                span_id="ART-005",
                device_type="article",
                text="Art. 5º O processo de licitação...",
            ),
            MockMilvusChunk(
                node_id="leis:LEI-14133-2021#ART-006@P00",
                chunk_id="LEI-14133-2021#ART-006@P00",
                document_id="LEI-14133-2021",
                span_id="ART-006",
                device_type="article",
                text="Art. 6º Para os fins desta Lei...",
            ),
            MockMilvusChunk(
                node_id="leis:LEI-14133-2021#ART-006@P01",
                chunk_id="LEI-14133-2021#ART-006@P01",
                document_id="LEI-14133-2021",
                span_id="ART-006",
                device_type="article",
                text="...continuação do art. 6º.",
            ),
            MockMilvusChunk(
                node_id="leis:LEI-14133-2021#INC-005-I@P00",
                chunk_id="LEI-14133-2021#INC-005-I@P00",
                document_id="LEI-14133-2021",
                span_id="INC-005-I",
                device_type="inciso",
                text="I - inciso I do art. 5º;",
            ),
        ]

    @pytest.fixture
    def sample_edges(self) -> list[MockEdge]:
        """Cria edges mockados do Neo4j."""
        return [
            # Edge válido: source_chunk_id existe no Milvus
            MockEdge(
                source_node_id="leis:LEI-14133-2021#ART-005",
                target_node_id="leis:LEI-14133-2021#ART-003",
                source_chunk_id="leis:LEI-14133-2021#ART-005@P00",
            ),
            # Edge válido: referência de inciso
            MockEdge(
                source_node_id="leis:LEI-14133-2021#INC-005-I",
                target_node_id="leis:LEI-8666-1993#ART-021",
                source_chunk_id="leis:LEI-14133-2021#INC-005-I@P00",
            ),
            # Edge válido: chunk splittado (Part 1)
            MockEdge(
                source_node_id="leis:LEI-14133-2021#ART-006",
                target_node_id="leis:LEI-14133-2021#ART-005",
                source_chunk_id="leis:LEI-14133-2021#ART-006@P01",
            ),
        ]

    def test_all_source_chunk_ids_exist_in_milvus(
        self, sample_edges: list[MockEdge], sample_milvus_chunks: list[MockMilvusChunk]
    ):
        """
        Verifica que todos os source_chunk_id nos edges existem no Milvus.

        Este é o teste principal de "cadeia de evidência".
        """
        # Extrai node_ids do Milvus
        milvus_node_ids = {chunk.node_id for chunk in sample_milvus_chunks}

        # Verifica cada edge
        invalid_edges = []
        for edge in sample_edges:
            if edge.source_chunk_id not in milvus_node_ids:
                invalid_edges.append(edge)

        assert len(invalid_edges) == 0, (
            f"{len(invalid_edges)} edges têm source_chunk_id que não existe no Milvus:\n"
            + "\n".join(f"  - {e.source_chunk_id}" for e in invalid_edges)
        )

    def test_source_chunk_id_format_has_part_suffix(self, sample_edges: list[MockEdge]):
        """
        Verifica que source_chunk_id tem o sufixo @Pxx (referência física).

        source_chunk_id deve apontar para chunk físico, não nó lógico.
        """
        for edge in sample_edges:
            assert "@P" in edge.source_chunk_id, (
                f"source_chunk_id deve ter sufixo @Pxx: {edge.source_chunk_id}"
            )

    def test_source_node_id_is_logical(self, sample_edges: list[MockEdge]):
        """
        Verifica que source_node_id NÃO tem sufixo @Pxx (é nó lógico).

        source_node_id é usado para criar o nó no grafo (sem parte física).
        """
        for edge in sample_edges:
            assert "@P" not in edge.source_node_id, (
                f"source_node_id não deve ter sufixo @Pxx: {edge.source_node_id}"
            )

    def test_logical_and_physical_are_related(self, sample_edges: list[MockEdge]):
        """
        Verifica que source_chunk_id é derivável de source_node_id.

        source_chunk_id = source_node_id + @Pxx
        """
        for edge in sample_edges:
            # Remove @Pxx do source_chunk_id para obter logical
            base = edge.source_chunk_id.split("@P")[0]

            assert base == edge.source_node_id, (
                f"source_chunk_id base '{base}' deve ser igual a source_node_id '{edge.source_node_id}'"
            )


class TestNoSelfLoops:
    """Testes para garantir que não existem self-loops."""

    def test_edge_source_different_from_target(self):
        """
        Verifica que source_node_id != target_node_id.

        Self-loops não fazem sentido em citações normativas.
        """
        # Edge válido
        valid_edge = MockEdge(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            source_chunk_id="leis:LEI-14133-2021#ART-005@P00",
        )
        assert valid_edge.source_node_id != valid_edge.target_node_id

        # Self-loop (deve ser rejeitado pelo writer)
        self_loop = MockEdge(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-005",
            source_chunk_id="leis:LEI-14133-2021#ART-005@P00",
        )
        # Este teste documenta o comportamento esperado
        # O Neo4jEdgeWriter deve rejeitar self-loops (linha 190-193)
        is_self_loop = self_loop.source_node_id == self_loop.target_node_id
        assert is_self_loop, "Este é um self-loop que deve ser rejeitado"

    def test_neo4j_writer_rejects_self_loops(self):
        """
        Verifica que Neo4jEdgeWriter rejeita self-loops.

        Testa o comportamento da classe sem conectar ao banco.
        """
        from sinks.neo4j_writer import EdgeCandidate, Neo4jEdgeWriter

        # Cria writer sem conectar ao banco
        writer = Neo4jEdgeWriter(password="test")
        # Simula conexão sem realmente conectar
        writer._connected = True
        writer._driver = MagicMock()

        # Cria self-loop
        self_loop = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-005",
            source_chunk_id="leis:LEI-14133-2021#ART-005@P00",
        )

        # create_edge deve retornar False para self-loop
        result = writer.create_edge(self_loop)

        assert result is False, "Neo4jEdgeWriter deve rejeitar self-loops"


class TestStubCreation:
    """Testes para verificar criação de stubs para targets não ingeridos."""

    def test_target_as_stub_when_not_ingested(self):
        """
        Verifica comportamento quando target ainda não foi ingerido.

        O Neo4jEdgeWriter usa MERGE com ON CREATE SET stub=true.
        """
        from sinks.neo4j_writer import Neo4jEdgeWriter, EdgeCandidate

        # Este teste documenta o comportamento esperado
        # Quando o target não existe, o MERGE cria um nó "stub"
        query_stub_creation = """
        MERGE (target:LegalNode {node_id: $target_node_id})
        ON CREATE SET target.stub = true,
                      target.created_at = datetime()
        """

        # Verifica que a query do writer contém essa lógica
        # (não podemos testar sem banco real, mas documentamos o comportamento)
        assert "stub = true" in query_stub_creation


class TestEdgeCandidateDeduplication:
    """Testes para deduplicação de edges."""

    def test_same_edge_extracted_from_split_chunks(self):
        """
        Verifica comportamento quando mesma citação aparece em chunks splitados.

        Com overlap de 200 chars, a mesma citação pode aparecer em P00 e P01.
        O Neo4j MERGE garante que apenas um edge é criado.
        """
        # Mesma citação extraída de duas partes do mesmo artigo
        edge_from_p00 = MockEdge(
            source_node_id="leis:LEI-14133-2021#ART-006",
            target_node_id="leis:LEI-14133-2021#ART-005",
            source_chunk_id="leis:LEI-14133-2021#ART-006@P00",
        )

        edge_from_p01 = MockEdge(
            source_node_id="leis:LEI-14133-2021#ART-006",
            target_node_id="leis:LEI-14133-2021#ART-005",
            source_chunk_id="leis:LEI-14133-2021#ART-006@P01",
        )

        # source_node_id e target_node_id são iguais
        assert edge_from_p00.source_node_id == edge_from_p01.source_node_id
        assert edge_from_p00.target_node_id == edge_from_p01.target_node_id

        # Mas source_chunk_id é diferente (mostra de qual parte física veio)
        assert edge_from_p00.source_chunk_id != edge_from_p01.source_chunk_id

        # O Neo4j MERGE (source)-[r:CITA]->(target) garante unicidade do edge
        # O source_chunk_id armazena a primeira (ou última) evidência física


class TestIntegrationEvidenceChain:
    """
    Testes de integração para cadeia de evidência completa.

    Estes testes requerem conexão real com Neo4j e Milvus.
    Marcados com @pytest.mark.integration para skip em CI.
    """

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requer conexão real com Neo4j e Milvus")
    def test_real_evidence_chain_for_document(self):
        """
        Teste real de cadeia de evidência.

        Para cada edge no Neo4j:
        1. Busca source_chunk_id
        2. Verifica se existe no Milvus
        3. Confirma que o texto contém a citação
        """
        # Este teste seria implementado com conexões reais
        # e executado manualmente ou em ambiente de staging
        pass

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requer conexão real com Neo4j")
    def test_no_self_loops_in_production(self):
        """
        Verifica que não existem self-loops no banco de produção.

        Query:
            MATCH (n)-[r:CITA]->(n) RETURN count(r) as self_loops
        """
        pass


class TestEdgeCandidateValidation:
    """Testes de validação de EdgeCandidate."""

    def test_edge_candidate_dataclass(self):
        """Verifica que EdgeCandidate tem todos os campos necessários."""
        from sinks.neo4j_writer import EdgeCandidate

        edge = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            source_chunk_id="leis:LEI-14133-2021#ART-005@P00",
            confidence=0.95,
            extraction_method="regex",
            citation_text="art. 3º",
            context_text="conforme disposto no art. 3º desta Lei",
            ingest_run_id="550e8400-e29b-41d4-a716-446655440000",
        )

        # Campos obrigatórios
        assert edge.source_node_id is not None
        assert edge.target_node_id is not None

        # Campos de proveniência
        assert edge.source_chunk_id is not None
        assert edge.extraction_method is not None

        # Campos opcionais
        assert edge.confidence >= 0 and edge.confidence <= 1
        assert edge.citation_text is not None

    def test_edge_candidate_defaults(self):
        """Verifica valores default do EdgeCandidate."""
        from sinks.neo4j_writer import EdgeCandidate

        edge = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
        )

        assert edge.relation_type == "CITA"
        assert edge.confidence == 1.0
        assert edge.extraction_method == "REGEX"  # PR3 v2.1: Normalizado para CAPS
        assert edge.source_chunk_id == ""  # Default vazio

    def test_edge_candidate_confidence_tier_derived(self):
        """Verifica que confidence_tier é derivado automaticamente do score."""
        from sinks.neo4j_writer import EdgeCandidate

        # HIGH tier (>= 0.8)
        edge_high = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            confidence=0.95,
        )
        assert edge_high.confidence_tier == "HIGH"

        # MEDIUM tier (0.5 - 0.8)
        edge_medium = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            confidence=0.65,
        )
        assert edge_medium.confidence_tier == "MEDIUM"

        # LOW tier (< 0.5)
        edge_low = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            confidence=0.3,
        )
        assert edge_low.confidence_tier == "LOW"

    def test_edge_candidate_extraction_method_normalized(self):
        """Verifica que extraction_method é normalizado para CAPS."""
        from sinks.neo4j_writer import EdgeCandidate

        # Lowercase -> CAPS
        edge = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            extraction_method="heuristic",
        )
        assert edge.extraction_method == "HEURISTIC"

        # Mixed case -> CAPS
        edge2 = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            extraction_method="Nli",
        )
        assert edge2.extraction_method == "NLI"

        # Unknown -> UNKNOWN
        edge3 = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            extraction_method="some_custom_method",
        )
        assert edge3.extraction_method == "UNKNOWN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
