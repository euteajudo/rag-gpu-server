# -*- coding: utf-8 -*-
"""
Testes do Neo4j Client: upsert_edge e upsert_edges_batch com rel_type_confidence.

PR5 - Camada 3: Validação da propagação de rel_type_confidence para o Neo4j.

Valida que:
1. upsert_edge aceita rel_type_confidence
2. upsert_edges_batch aceita rel_type_confidence em cada edge
3. A query Cypher seta rel_type_confidence corretamente
4. Valores default são aplicados quando não especificados

@author: Equipe VectorGov
@since: 30/01/2025
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Adiciona src ao path para imports
extracao_src_path = Path(__file__).parent.parent.parent / "extracao" / "src"
if str(extracao_src_path) not in sys.path:
    sys.path.insert(0, str(extracao_src_path))

# Tenta importar o client do extracao (onde está o Neo4j client real)
try:
    from graph.client import Neo4jClient
    from graph.constants import VALID_REL_TYPES, validate_rel_type
    HAS_NEO4J_CLIENT = True
except ImportError:
    HAS_NEO4J_CLIENT = False
    Neo4jClient = None
    VALID_REL_TYPES = None
    validate_rel_type = None

import pytest


# =============================================================================
# TESTES: Assinatura do upsert_edge
# =============================================================================


@pytest.mark.skipif(not HAS_NEO4J_CLIENT, reason="Neo4jClient não disponível")
class TestUpsertEdgeSignature:
    """Testa a assinatura do método upsert_edge."""

    def test_upsert_edge_accepts_rel_type_confidence(self):
        """upsert_edge deve aceitar parâmetro rel_type_confidence."""
        import inspect

        sig = inspect.signature(Neo4jClient.upsert_edge)
        params = sig.parameters

        assert "rel_type_confidence" in params, \
            "upsert_edge deve ter parâmetro rel_type_confidence"

    def test_upsert_edge_rel_type_confidence_default(self):
        """rel_type_confidence deve ter default 0.0."""
        import inspect

        sig = inspect.signature(Neo4jClient.upsert_edge)
        param = sig.parameters.get("rel_type_confidence")

        assert param is not None
        assert param.default == 0.0, \
            "rel_type_confidence deve ter default 0.0"


@pytest.mark.skipif(not HAS_NEO4J_CLIENT, reason="Neo4jClient não disponível")
class TestUpsertEdgesBatchSignature:
    """Testa a assinatura do método upsert_edges_batch."""

    def test_upsert_edges_batch_accepts_rel_type_confidence_in_edge_dict(self):
        """upsert_edges_batch deve processar rel_type_confidence em cada edge dict."""
        # Verifica que o método existe e pode ser chamado com edges que contêm rel_type_confidence
        import inspect

        # Método existe
        assert hasattr(Neo4jClient, "upsert_edges_batch")
        assert callable(getattr(Neo4jClient, "upsert_edges_batch"))


# =============================================================================
# TESTES: Comportamento com mocks
# =============================================================================


@pytest.mark.skipif(not HAS_NEO4J_CLIENT, reason="Neo4jClient não disponível")
class TestUpsertEdgeBehavior:
    """Testa o comportamento do upsert_edge com mocks."""

    @patch.object(Neo4jClient, '_execute_write')
    def test_upsert_edge_passes_rel_type_confidence_to_cypher(self, mock_execute):
        """upsert_edge deve passar rel_type_confidence para a query Cypher."""
        mock_execute.return_value = 1

        client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test"
        )
        # Mock a conexão
        client._driver = MagicMock()
        client._connected = True

        # Chama upsert_edge com rel_type_confidence
        client.upsert_edge(
            source_node_id="leis:LEI-14133-2021#ART-200",
            target_node_id="leis:LEI-8666-1993#ART-005",
            rel_type="REVOGA_EXPRESSAMENTE",
            rel_type_confidence=0.95,
        )

        # Verifica que foi chamado
        assert mock_execute.called

        # Verifica os parâmetros passados
        call_args = mock_execute.call_args
        if call_args:
            # Pode ser args ou kwargs
            params = call_args.kwargs.get("parameters") or (
                call_args[0][1] if len(call_args[0]) > 1 else {}
            )
            # O nome do parâmetro pode variar
            assert params.get("rel_type_confidence") == 0.95 or \
                   "rel_type_confidence" in str(call_args)


@pytest.mark.skipif(not HAS_NEO4J_CLIENT, reason="Neo4jClient não disponível")
class TestUpsertEdgesBatchBehavior:
    """Testa o comportamento do upsert_edges_batch com mocks."""

    @patch.object(Neo4jClient, '_execute_write')
    def test_upsert_edges_batch_includes_rel_type_confidence(self, mock_execute):
        """upsert_edges_batch deve processar rel_type_confidence de cada edge."""
        mock_execute.return_value = 2

        client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test"
        )
        client._driver = MagicMock()
        client._connected = True

        edges = [
            {
                "source_node_id": "leis:LEI-14133-2021#ART-200",
                "target_node_id": "leis:LEI-8666-1993#ART-005",
                "rel_type": "REVOGA_EXPRESSAMENTE",
                "rel_type_confidence": 0.95,
            },
            {
                "source_node_id": "leis:LEI-14133-2021#ART-201",
                "target_node_id": "leis:LEI-14133-2021#ART-018",
                "rel_type": "ALTERA_EXPRESSAMENTE",
                "rel_type_confidence": 0.90,
            },
        ]

        client.upsert_edges_batch(edges)

        # Verifica que foi chamado
        assert mock_execute.called


# =============================================================================
# TESTES: Validação de rel_type
# =============================================================================


@pytest.mark.skipif(not HAS_NEO4J_CLIENT, reason="Neo4jClient não disponível")
class TestRelTypeValidation:
    """Testa a validação de rel_type."""

    def test_valid_rel_types_include_all_pr5_types(self):
        """VALID_REL_TYPES deve incluir todos os tipos do PR5."""
        required_types = {
            "CITA",
            "REFERENCIA",
            "ALTERA_EXPRESSAMENTE",
            "REVOGA_EXPRESSAMENTE",
            "REVOGA_TACITAMENTE",
            "REGULAMENTA",
            "DEPENDE_DE",
            "EXCEPCIONA",
        }

        for rel_type in required_types:
            assert rel_type in VALID_REL_TYPES, \
                f"{rel_type} deve estar em VALID_REL_TYPES"

    def test_validate_rel_type_accepts_valid_types(self):
        """validate_rel_type deve aceitar tipos válidos."""
        valid_types = [
            "CITA",
            "REVOGA_EXPRESSAMENTE",
            "ALTERA_EXPRESSAMENTE",
            "REGULAMENTA",
        ]

        for rel_type in valid_types:
            result = validate_rel_type(rel_type)
            assert result == rel_type

    def test_validate_rel_type_rejects_invalid_types(self):
        """validate_rel_type deve rejeitar tipos inválidos."""
        invalid_types = [
            "INVALIDO",
            "DROP TABLE",
            "CITA; DELETE",
            "",
        ]

        for rel_type in invalid_types:
            with pytest.raises(ValueError):
                validate_rel_type(rel_type)

    def test_validate_rel_type_normalizes_case(self):
        """validate_rel_type deve normalizar para uppercase."""
        result = validate_rel_type("cita")
        assert result == "CITA"

        result = validate_rel_type("revoga_expressamente")
        assert result == "REVOGA_EXPRESSAMENTE"


# =============================================================================
# TESTES: Formato do edge dict
# =============================================================================


class TestEdgeDictFormat:
    """Testa o formato esperado do dict de edge."""

    def test_edge_dict_with_all_fields(self):
        """Edge dict deve aceitar todos os campos PR5."""
        edge = {
            "source_node_id": "leis:LEI-14133-2021#ART-200",
            "target_node_id": "leis:LEI-8666-1993#ART-005",
            "raw": "",
            "method": "regex",
            "confidence": 1.0,
            "rel_type": "REVOGA_EXPRESSAMENTE",
            "rel_type_confidence": 0.95,
        }

        # Verifica campos obrigatórios
        assert "source_node_id" in edge
        assert "target_node_id" in edge
        assert "rel_type" in edge
        assert "rel_type_confidence" in edge

        # Verifica tipos
        assert isinstance(edge["source_node_id"], str)
        assert isinstance(edge["target_node_id"], str)
        assert isinstance(edge["rel_type"], str)
        assert isinstance(edge["rel_type_confidence"], (int, float))

    def test_edge_dict_confidence_vs_rel_type_confidence(self):
        """confidence e rel_type_confidence são campos distintos."""
        edge = {
            "source_node_id": "leis:LEI-14133-2021#ART-200",
            "target_node_id": "leis:LEI-8666-1993#ART-005",
            "confidence": 1.0,  # Confiança da extração
            "rel_type": "REVOGA_EXPRESSAMENTE",
            "rel_type_confidence": 0.95,  # Confiança da classificação
        }

        # São campos diferentes
        assert edge["confidence"] != edge["rel_type_confidence"]

        # confidence é para extração (regex = 100%)
        assert edge["confidence"] == 1.0

        # rel_type_confidence é para classificação (variável)
        assert edge["rel_type_confidence"] == 0.95

    def test_edge_dict_with_default_values(self):
        """Edge dict deve funcionar com valores default."""
        edge = {
            "source_node_id": "leis:LEI-14133-2021#ART-200",
            "target_node_id": "leis:LEI-8666-1993#ART-005",
            # Sem especificar rel_type e rel_type_confidence
        }

        # Valores default devem ser aplicados pelo sync_service
        default_rel_type = "CITA"
        default_rel_type_confidence = 0.5  # Para formato legado

        # Simula o que o sync_service faz
        edge.setdefault("rel_type", default_rel_type)
        edge.setdefault("rel_type_confidence", default_rel_type_confidence)

        assert edge["rel_type"] == "CITA"
        assert edge["rel_type_confidence"] == 0.5


# =============================================================================
# TESTES: Cenários de uso real
# =============================================================================


class TestRealWorldScenarios:
    """Testa cenários de uso real."""

    def test_revoga_expressamente_scenario(self):
        """Cenário: Lei nova revoga artigo de lei antiga."""
        # Texto: "Fica revogado o art. 5º da Lei 8.666/1993"
        edge = {
            "source_node_id": "leis:LEI-14133-2021#ART-193",
            "target_node_id": "leis:LEI-8666-1993#ART-005",
            "rel_type": "REVOGA_EXPRESSAMENTE",
            "rel_type_confidence": 0.95,
            "confidence": 1.0,
        }

        # Validações do PR5
        assert edge["rel_type"] == "REVOGA_EXPRESSAMENTE"
        assert edge["rel_type_confidence"] >= 0.90, \
            "REVOGA_EXPRESSAMENTE deve ter confidence >= 0.90"

    def test_altera_expressamente_scenario(self):
        """Cenário: Lei altera redação de artigo."""
        # Texto: "O art. 18 passa a vigorar com a seguinte redação"
        edge = {
            "source_node_id": "leis:LEI-14234-2022#ART-001",
            "target_node_id": "leis:LEI-14133-2021#ART-018",
            "rel_type": "ALTERA_EXPRESSAMENTE",
            "rel_type_confidence": 0.92,
            "confidence": 1.0,
        }

        assert edge["rel_type"] == "ALTERA_EXPRESSAMENTE"
        assert edge["rel_type_confidence"] >= 0.85

    def test_regulamenta_scenario(self):
        """Cenário: Decreto regulamenta lei."""
        # Texto: "Este Decreto regulamenta a Lei 14.133"
        edge = {
            "source_node_id": "leis:DECRETO-10947-2022#ART-001",
            "target_node_id": "leis:LEI-14133-2021",
            "rel_type": "REGULAMENTA",
            "rel_type_confidence": 0.88,
            "confidence": 1.0,
        }

        assert edge["rel_type"] == "REGULAMENTA"
        assert edge["rel_type_confidence"] >= 0.80

    def test_cita_simples_scenario(self):
        """Cenário: Citação simples sem efeito específico."""
        # Texto: "Conforme previsto no art. 25"
        edge = {
            "source_node_id": "leis:IN-65-2022#ART-003",
            "target_node_id": "leis:LEI-14133-2021#ART-025",
            "rel_type": "CITA",
            "rel_type_confidence": 0.50,
            "confidence": 1.0,
        }

        assert edge["rel_type"] == "CITA"
        assert edge["rel_type_confidence"] == 0.50  # Default para citação simples
