# -*- coding: utf-8 -*-
"""
Testes de integração: GraphRetriever com PR6 (rel_type_confidence + allowed_rel_types).

PR6 - Camada 3: Validação do GraphRetriever com:
1. Filtragem por min_rel_type_confidence
2. Filtragem por allowed_rel_types (presets IMPACT_ONLY, BROAD)
3. Compatibilidade com min_confidence_tier do PR4
4. Retorno de TODAS as edges do path com metadados completos

@author: Equipe VectorGov
@since: 30/01/2025
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# Adiciona src ao path para imports
extracao_src_path = Path(__file__).parent.parent.parent / "extracao" / "src"
if str(extracao_src_path) not in sys.path:
    sys.path.insert(0, str(extracao_src_path))

# Tenta importar módulos necessários
try:
    from graph.graph_retriever import (
        GraphRetriever,
        CitedNode,
        EdgeInfo,
        DirectEvidence,
    )
    from graph.constants import (
        VALID_REL_TYPES,
        REL_TYPES_IMPACT_ONLY,
        REL_TYPES_BROAD,
        REL_TYPE_MIN_CONFIDENCE,
        DEFAULT_MIN_REL_TYPE_CONFIDENCE,
    )
    HAS_GRAPH_RETRIEVER = True
except ImportError:
    HAS_GRAPH_RETRIEVER = False
    GraphRetriever = None
    CitedNode = None
    EdgeInfo = None
    DirectEvidence = None
    VALID_REL_TYPES = None
    REL_TYPES_IMPACT_ONLY = None
    REL_TYPES_BROAD = None
    REL_TYPE_MIN_CONFIDENCE = None
    DEFAULT_MIN_REL_TYPE_CONFIDENCE = None


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_neo4j_session():
    """Mock de sessão Neo4j para testes unitários."""
    session_mock = MagicMock()
    return session_mock


@pytest.fixture
def mock_graph_retriever():
    """Mock completo do GraphRetriever para testes unitários."""
    if not HAS_GRAPH_RETRIEVER:
        pytest.skip("GraphRetriever não disponível")

    with patch.object(GraphRetriever, '__init__', lambda self, *args, **kwargs: None):
        retriever = GraphRetriever.__new__(GraphRetriever)
        # Configura atributos mínimos
        retriever.neo4j_driver = MagicMock()
        retriever._parent_cache = MagicMock()
        return retriever


# =============================================================================
# TESTES: Presets de allowed_rel_types
# =============================================================================


@pytest.mark.skipif(not HAS_GRAPH_RETRIEVER, reason="GraphRetriever não disponível")
class TestRelTypePresets:
    """Testa os presets REL_TYPES_IMPACT_ONLY e REL_TYPES_BROAD."""

    def test_impact_only_preset_contains_high_impact_types(self):
        """REL_TYPES_IMPACT_ONLY deve conter apenas tipos de alto impacto."""
        expected = {
            "REVOGA_EXPRESSAMENTE",
            "ALTERA_EXPRESSAMENTE",
            "REGULAMENTA",
            "EXCEPCIONA",
            "DEPENDE_DE",
        }
        assert REL_TYPES_IMPACT_ONLY == expected

    def test_broad_preset_includes_cita_and_referencia(self):
        """REL_TYPES_BROAD deve incluir CITA e REFERENCIA além do alto impacto."""
        assert "CITA" in REL_TYPES_BROAD
        assert "REFERENCIA" in REL_TYPES_BROAD
        # Também deve incluir os de alto impacto
        for rt in REL_TYPES_IMPACT_ONLY:
            assert rt in REL_TYPES_BROAD

    def test_presets_are_subsets_of_valid_rel_types(self):
        """Todos os presets devem ser subconjuntos de VALID_REL_TYPES."""
        for rt in REL_TYPES_IMPACT_ONLY:
            assert rt in VALID_REL_TYPES
        for rt in REL_TYPES_BROAD:
            assert rt in VALID_REL_TYPES


# =============================================================================
# TESTES: min_rel_type_confidence filtering
# =============================================================================


@pytest.mark.skipif(not HAS_GRAPH_RETRIEVER, reason="GraphRetriever não disponível")
class TestMinRelTypeConfidenceFiltering:
    """Testa filtragem por min_rel_type_confidence no GraphRetriever."""

    def test_default_min_confidence_for_cita(self):
        """CITA deve ter min_confidence 0.0 (aceita qualquer)."""
        assert REL_TYPE_MIN_CONFIDENCE.get("CITA") == 0.0

    def test_default_min_confidence_for_referencia(self):
        """REFERENCIA deve ter min_confidence 0.0 (aceita qualquer)."""
        assert REL_TYPE_MIN_CONFIDENCE.get("REFERENCIA") == 0.0

    def test_default_min_confidence_for_high_impact_types(self):
        """Tipos de alto impacto devem exigir min_confidence >= 0.80."""
        high_impact = [
            "REVOGA_EXPRESSAMENTE",
            "ALTERA_EXPRESSAMENTE",
            "REGULAMENTA",
            "EXCEPCIONA",
            "DEPENDE_DE",
        ]
        for rel_type in high_impact:
            assert REL_TYPE_MIN_CONFIDENCE.get(rel_type, 0.0) >= 0.80, \
                f"{rel_type} deve exigir min_confidence >= 0.80"

    def test_default_min_rel_type_confidence_is_080(self):
        """DEFAULT_MIN_REL_TYPE_CONFIDENCE deve ser 0.80."""
        assert DEFAULT_MIN_REL_TYPE_CONFIDENCE == 0.80


# =============================================================================
# TESTES: EdgeInfo dataclass
# =============================================================================


@pytest.mark.skipif(not HAS_GRAPH_RETRIEVER, reason="GraphRetriever não disponível")
class TestEdgeInfoDataclass:
    """Testa a dataclass EdgeInfo do PR6."""

    def test_edge_info_has_required_fields(self):
        """EdgeInfo deve ter todos os campos obrigatórios."""
        edge = EdgeInfo(
            source_node_id="leis:LEI-14133-2021#ART-100",
            target_node_id="leis:LEI-14133-2021#ART-005",
            rel_type="REVOGA_EXPRESSAMENTE",
            rel_type_confidence=0.95,
            confidence_tier="HIGH",
        )
        assert edge.source_node_id == "leis:LEI-14133-2021#ART-100"
        assert edge.target_node_id == "leis:LEI-14133-2021#ART-005"
        assert edge.rel_type == "REVOGA_EXPRESSAMENTE"
        assert edge.rel_type_confidence == 0.95
        assert edge.confidence_tier == "HIGH"

    def test_edge_info_defaults(self):
        """EdgeInfo deve ter defaults corretos."""
        edge = EdgeInfo(
            source_node_id="a",
            target_node_id="b",
        )
        assert edge.rel_type == "CITA"
        assert edge.rel_type_confidence == 0.0
        assert edge.confidence_tier == "LOW"
        assert edge.source_chunk_id == ""
        assert edge.extraction_method == ""


# =============================================================================
# TESTES: CitedNode com lista de edges
# =============================================================================


@pytest.mark.skipif(not HAS_GRAPH_RETRIEVER, reason="GraphRetriever não disponível")
class TestCitedNodeEdges:
    """Testa CitedNode com lista de edges (PR6)."""

    def test_cited_node_has_edges_list(self):
        """CitedNode deve ter campo edges como lista."""
        node = CitedNode(
            node_id="leis:LEI-14133-2021#ART-005",
            text="Texto do artigo",
            document_id="LEI-14133-2021",
            span_id="ART-005",
            device_type="article",
            hop=1,
            frequency=3,
        )
        assert hasattr(node, "edges")
        assert isinstance(node.edges, list)

    def test_cited_node_edges_are_edge_info(self):
        """CitedNode.edges deve conter objetos EdgeInfo."""
        edge1 = EdgeInfo(
            source_node_id="leis:LEI-14133-2021#ART-100",
            target_node_id="leis:LEI-14133-2021#ART-050",
            rel_type="CITA",
            rel_type_confidence=0.5,
        )
        edge2 = EdgeInfo(
            source_node_id="leis:LEI-14133-2021#ART-050",
            target_node_id="leis:LEI-14133-2021#ART-005",
            rel_type="REGULAMENTA",
            rel_type_confidence=0.88,
        )

        node = CitedNode(
            node_id="leis:LEI-14133-2021#ART-005",
            text="Texto do artigo",
            document_id="LEI-14133-2021",
            span_id="ART-005",
            device_type="article",
            hop=2,
            frequency=1,
            edges=[edge1, edge2],
        )

        assert len(node.edges) == 2
        assert all(isinstance(e, EdgeInfo) for e in node.edges)
        assert node.edges[0].rel_type == "CITA"
        assert node.edges[1].rel_type == "REGULAMENTA"

    def test_cited_node_backwards_compat_fields(self):
        """CitedNode deve manter campos legados para backwards compatibility."""
        node = CitedNode(
            node_id="leis:LEI-14133-2021#ART-005",
            text="Texto",
            document_id="LEI-14133-2021",
            span_id="ART-005",
            device_type="article",
            hop=1,
            frequency=1,
            source_chunk_id="leis:LEI-14133-2021#ART-100@P00",
            extraction_method="REGEX",
            confidence_tier="HIGH",
        )
        assert node.source_chunk_id == "leis:LEI-14133-2021#ART-100@P00"
        assert node.extraction_method == "REGEX"
        assert node.confidence_tier == "HIGH"


# =============================================================================
# TESTES: _expand_via_graph com PR6 parameters
# =============================================================================


@pytest.mark.skipif(not HAS_GRAPH_RETRIEVER, reason="GraphRetriever não disponível")
class TestExpandViaGraphPR6:
    """Testa _expand_via_graph com parâmetros PR6."""

    def test_expand_with_allowed_rel_types_impact_only(self, mock_graph_retriever):
        """Teste que allowed_rel_types=IMPACT_ONLY filtra tipos genéricos."""
        # Mock do session e resultado
        mock_session = MagicMock()
        mock_session.run.return_value = []  # Sem resultados para simplificar
        mock_graph_retriever.neo4j_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_graph_retriever.neo4j_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Chama com IMPACT_ONLY
        result = mock_graph_retriever._expand_via_graph(
            seed_node_ids=["leis:LEI-14133-2021#ART-100"],
            hops=2,
            topK_graph=10,
            min_confidence_tier="LOW",
            allowed_rel_types=list(REL_TYPES_IMPACT_ONLY),
        )

        # Verifica que a query foi executada
        assert mock_session.run.called
        call_args = mock_session.run.call_args
        query = call_args[0][0]

        # Verifica que o pattern de relacionamento inclui apenas alto impacto
        assert "REVOGA_EXPRESSAMENTE" in query or "ALTERA_EXPRESSAMENTE" in query
        # CITA não deve estar (a menos que esteja nos allowed_rel_types)
        # Como passamos IMPACT_ONLY, CITA não deve aparecer no pattern
        assert "CITA" not in query.split("*")[0]  # Antes do *1..N

    def test_expand_with_min_rel_type_confidence(self, mock_graph_retriever):
        """Teste que min_rel_type_confidence é passado corretamente."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_graph_retriever.neo4j_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_graph_retriever.neo4j_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Chama com min_rel_type_confidence explícito
        result = mock_graph_retriever._expand_via_graph(
            seed_node_ids=["leis:LEI-14133-2021#ART-100"],
            hops=1,
            topK_graph=10,
            min_rel_type_confidence=0.90,
        )

        # Verifica que o parâmetro foi passado
        call_kwargs = mock_session.run.call_args[1]
        assert call_kwargs["min_rel_type_confidence"] == 0.90

    def test_expand_default_min_rel_type_confidence_for_cita(self, mock_graph_retriever):
        """Teste que CITA usa min_rel_type_confidence 0.0 por default."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_graph_retriever.neo4j_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_graph_retriever.neo4j_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Chama sem min_rel_type_confidence (usa default por tipo)
        result = mock_graph_retriever._expand_via_graph(
            seed_node_ids=["leis:LEI-14133-2021#ART-100"],
            hops=1,
            topK_graph=10,
            allowed_rel_types=["CITA"],
            min_rel_type_confidence=None,  # Usa default
        )

        # Para CITA, o default é 0.0
        call_kwargs = mock_session.run.call_args[1]
        assert call_kwargs["min_rel_type_confidence"] == 0.0


# =============================================================================
# TESTES: Compatibilidade com min_confidence_tier (PR4)
# =============================================================================


@pytest.mark.skipif(not HAS_GRAPH_RETRIEVER, reason="GraphRetriever não disponível")
class TestPR4Compatibility:
    """Testa que PR6 não quebra funcionalidades do PR4."""

    def test_min_confidence_tier_still_works(self, mock_graph_retriever):
        """min_confidence_tier deve continuar funcionando com PR6."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_graph_retriever.neo4j_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_graph_retriever.neo4j_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Chama com min_confidence_tier HIGH
        result = mock_graph_retriever._expand_via_graph(
            seed_node_ids=["leis:LEI-14133-2021#ART-100"],
            hops=1,
            topK_graph=10,
            min_confidence_tier="HIGH",
        )

        # Verifica que min_tier_value foi passado corretamente
        call_kwargs = mock_session.run.call_args[1]
        assert call_kwargs["min_tier_value"] == 3  # HIGH = 3

    def test_both_filters_can_be_used_together(self, mock_graph_retriever):
        """min_confidence_tier e min_rel_type_confidence podem ser usados juntos."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_graph_retriever.neo4j_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_graph_retriever.neo4j_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Usa ambos os filtros
        result = mock_graph_retriever._expand_via_graph(
            seed_node_ids=["leis:LEI-14133-2021#ART-100"],
            hops=2,
            topK_graph=10,
            min_confidence_tier="MEDIUM",  # PR4
            min_rel_type_confidence=0.85,  # PR6
        )

        call_kwargs = mock_session.run.call_args[1]
        assert call_kwargs["min_tier_value"] == 2  # MEDIUM = 2
        assert call_kwargs["min_rel_type_confidence"] == 0.85


# =============================================================================
# TESTES: Validação de rel_types
# =============================================================================


@pytest.mark.skipif(not HAS_GRAPH_RETRIEVER, reason="GraphRetriever não disponível")
class TestRelTypeValidation:
    """Testa validação de allowed_rel_types."""

    def test_invalid_rel_type_fallback_to_cita(self, mock_graph_retriever):
        """Tipos inválidos devem fazer fallback para CITA."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_graph_retriever.neo4j_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_graph_retriever.neo4j_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Tipos inválidos
        result = mock_graph_retriever._expand_via_graph(
            seed_node_ids=["leis:LEI-14133-2021#ART-100"],
            hops=1,
            topK_graph=10,
            allowed_rel_types=["INVALID_TYPE", "DROP TABLE"],
        )

        # Query deve usar CITA como fallback
        query = mock_session.run.call_args[0][0]
        assert "CITA" in query

    def test_cypher_injection_prevention(self, mock_graph_retriever):
        """Tentativas de injeção Cypher devem ser bloqueadas."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        mock_graph_retriever.neo4j_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_graph_retriever.neo4j_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Tentativa de injeção
        result = mock_graph_retriever._expand_via_graph(
            seed_node_ids=["leis:LEI-14133-2021#ART-100"],
            hops=1,
            topK_graph=10,
            allowed_rel_types=["CITA]->(n) DETACH DELETE n//"],
        )

        # Query deve usar CITA (fallback seguro)
        query = mock_session.run.call_args[0][0]
        assert "DETACH DELETE" not in query


# =============================================================================
# TESTES: retrieve() com PR6 parameters
# =============================================================================


@pytest.mark.skipif(not HAS_GRAPH_RETRIEVER, reason="GraphRetriever não disponível")
class TestRetrieveWithPR6:
    """Testa método retrieve() expondo parâmetros PR6."""

    def test_retrieve_signature_has_pr6_params(self):
        """retrieve() deve ter parâmetros allowed_rel_types e min_rel_type_confidence."""
        import inspect
        sig = inspect.signature(GraphRetriever.retrieve)
        params = sig.parameters

        assert "allowed_rel_types" in params
        assert "min_rel_type_confidence" in params

    def test_retrieve_passes_pr6_params_to_expand(self, mock_graph_retriever):
        """retrieve() deve passar parâmetros PR6 para _expand_via_graph."""
        # Mock completo do retriever
        mock_graph_retriever._search_milvus_hybrid = MagicMock(return_value=[
            DirectEvidence(
                node_id="leis:LEI-14133-2021#ART-100@P00",
                logical_node_id="leis:LEI-14133-2021#ART-100",
                score=0.9,
                text="Texto",
                retrieval_text="[CONTEXTO] Texto",
                parent_text=None,
                document_id="LEI-14133-2021",
                span_id="ART-100",
                device_type="article",
            )
        ])
        mock_graph_retriever._expand_via_graph = MagicMock(return_value=[])
        mock_graph_retriever._apply_token_budget = MagicMock(return_value=([], [], False))
        mock_graph_retriever._count_total_tokens = MagicMock(return_value=0)
        mock_graph_retriever.use_reranker = False
        mock_graph_retriever.stage1_limit = 10
        mock_graph_retriever._parent_cache = MagicMock()
        mock_graph_retriever._parent_cache.stats = {}

        # Chama retrieve com parâmetros PR6
        mock_graph_retriever.retrieve(
            query="O que é ETP?",
            collection_name="leis_v4",
            allowed_rel_types=list(REL_TYPES_IMPACT_ONLY),
            min_rel_type_confidence=0.85,
        )

        # Verifica que _expand_via_graph foi chamado com parâmetros PR6
        call_kwargs = mock_graph_retriever._expand_via_graph.call_args[1]
        assert call_kwargs["allowed_rel_types"] == list(REL_TYPES_IMPACT_ONLY)
        assert call_kwargs["min_rel_type_confidence"] == 0.85
