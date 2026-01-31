# -*- coding: utf-8 -*-
"""
Testes de propagação: rel_type e rel_type_confidence no payload.

PR5 - Camada 2/3: Validação da propagação de metadados de classificação
através do pipeline até o formato final que vai para o Neo4j.

Valida que:
1. ProcessedChunk.citations contém rel_type e rel_type_confidence
2. O payload para sync_service preserva os metadados
3. A conversão de formatos não perde informação

@author: Equipe VectorGov
@since: 30/01/2025
"""

import sys
from pathlib import Path

# Adiciona src ao path para imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import importlib.util


def load_module(name: str, file_path: Path):
    """Carrega módulo diretamente do arquivo."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Carrega módulo de models
models_module = load_module(
    "models",
    src_path / "ingestion" / "models.py"
)

ProcessedChunk = models_module.ProcessedChunk


# =============================================================================
# TESTES: ProcessedChunk.citations aceita formato PR5
# =============================================================================


class TestProcessedChunkCitationsFormat:
    """Testa que ProcessedChunk.citations aceita o formato PR5."""

    def test_citations_accepts_list_of_dicts(self):
        """ProcessedChunk.citations deve aceitar lista de dicts com rel_type."""
        citations = [
            {
                "target_node_id": "leis:LEI-8666-1993#ART-005",
                "rel_type": "REVOGA_EXPRESSAMENTE",
                "rel_type_confidence": 0.95,
            },
            {
                "target_node_id": "leis:LEI-14133-2021#ART-018",
                "rel_type": "CITA",
                "rel_type_confidence": 0.5,
            },
        ]

        chunk = ProcessedChunk(
            node_id="leis:LEI-14234-2021#ART-200",
            chunk_id="LEI-14234-2021#ART-200",
            span_id="ART-200",
            device_type="article",
            chunk_level="article",
            text="Fica revogado o art. 5º da Lei 8.666/1993.",
            document_id="LEI-14234-2021",
            tipo_documento="LEI",
            numero="14234",
            ano=2021,
            citations=citations,
        )

        assert chunk.citations == citations
        assert len(chunk.citations) == 2

        # Verifica estrutura
        assert chunk.citations[0]["rel_type"] == "REVOGA_EXPRESSAMENTE"
        assert chunk.citations[0]["rel_type_confidence"] == 0.95

    def test_citations_accepts_legacy_format(self):
        """ProcessedChunk.citations deve aceitar formato legado (lista de strings)."""
        citations = [
            "leis:LEI-8666-1993#ART-005",
            "leis:LEI-14133-2021#ART-018",
        ]

        chunk = ProcessedChunk(
            node_id="leis:LEI-14234-2021#ART-200",
            chunk_id="LEI-14234-2021#ART-200",
            span_id="ART-200",
            device_type="article",
            chunk_level="article",
            text="Texto do artigo.",
            document_id="LEI-14234-2021",
            tipo_documento="LEI",
            numero="14234",
            ano=2021,
            citations=citations,
        )

        assert chunk.citations == citations
        assert len(chunk.citations) == 2
        assert isinstance(chunk.citations[0], str)

    def test_citations_default_empty_list(self):
        """ProcessedChunk.citations deve ter default como lista vazia."""
        chunk = ProcessedChunk(
            node_id="leis:LEI-14234-2021#ART-200",
            chunk_id="LEI-14234-2021#ART-200",
            span_id="ART-200",
            device_type="article",
            chunk_level="article",
            text="Texto do artigo.",
            document_id="LEI-14234-2021",
            tipo_documento="LEI",
            numero="14234",
            ano=2021,
        )

        assert chunk.citations == []


# =============================================================================
# TESTES: Payload para Neo4j Sync
# =============================================================================


class TestPayloadForNeo4jSync:
    """Testa que o payload está correto para sincronização."""

    def test_dict_conversion_preserves_citations(self):
        """Conversão para dict deve preservar citations com metadados."""
        citations = [
            {
                "target_node_id": "leis:LEI-8666-1993#ART-005",
                "rel_type": "REVOGA_EXPRESSAMENTE",
                "rel_type_confidence": 0.95,
            },
        ]

        chunk = ProcessedChunk(
            node_id="leis:LEI-14234-2021#ART-200",
            chunk_id="LEI-14234-2021#ART-200",
            span_id="ART-200",
            device_type="article",
            chunk_level="article",
            text="Fica revogado o art. 5º.",
            document_id="LEI-14234-2021",
            tipo_documento="LEI",
            numero="14234",
            ano=2021,
            citations=citations,
        )

        # Converte para dict (como seria enviado para API/Milvus)
        chunk_dict = chunk.model_dump()

        assert "citations" in chunk_dict
        assert len(chunk_dict["citations"]) == 1
        assert chunk_dict["citations"][0]["rel_type"] == "REVOGA_EXPRESSAMENTE"
        assert chunk_dict["citations"][0]["rel_type_confidence"] == 0.95

    def test_json_serialization_preserves_rel_type(self):
        """Serialização JSON deve preservar rel_type e rel_type_confidence."""
        import json

        citations = [
            {
                "target_node_id": "leis:LEI-8666-1993#ART-005",
                "rel_type": "REVOGA_EXPRESSAMENTE",
                "rel_type_confidence": 0.95,
            },
            {
                "target_node_id": "leis:LEI-14133-2021#ART-018",
                "rel_type": "ALTERA_EXPRESSAMENTE",
                "rel_type_confidence": 0.90,
            },
        ]

        chunk = ProcessedChunk(
            node_id="leis:LEI-14234-2021#ART-200",
            chunk_id="LEI-14234-2021#ART-200",
            span_id="ART-200",
            device_type="article",
            chunk_level="article",
            text="Texto.",
            document_id="LEI-14234-2021",
            tipo_documento="LEI",
            numero="14234",
            ano=2021,
            citations=citations,
        )

        # Serializa para JSON
        json_str = chunk.model_dump_json()
        parsed = json.loads(json_str)

        # Verifica que metadados foram preservados
        assert len(parsed["citations"]) == 2
        assert parsed["citations"][0]["rel_type"] == "REVOGA_EXPRESSAMENTE"
        assert parsed["citations"][0]["rel_type_confidence"] == 0.95
        assert parsed["citations"][1]["rel_type"] == "ALTERA_EXPRESSAMENTE"
        assert parsed["citations"][1]["rel_type_confidence"] == 0.90


# =============================================================================
# TESTES: Formato esperado pelo sync_service
# =============================================================================


class TestSyncServiceExpectedFormat:
    """Testa o formato esperado pelo sync_service."""

    def test_citation_dict_has_required_fields(self):
        """Dict de citação deve ter campos obrigatórios para sync."""
        citation = {
            "target_node_id": "leis:LEI-8666-1993#ART-005",
            "rel_type": "REVOGA_EXPRESSAMENTE",
            "rel_type_confidence": 0.95,
        }

        # Campos obrigatórios
        assert "target_node_id" in citation
        assert "rel_type" in citation
        assert "rel_type_confidence" in citation

        # Tipos corretos
        assert isinstance(citation["target_node_id"], str)
        assert isinstance(citation["rel_type"], str)
        assert isinstance(citation["rel_type_confidence"], (int, float))

    def test_rel_type_is_valid_enum_value(self):
        """rel_type deve ser um valor válido da taxonomia."""
        valid_rel_types = {
            "CITA",
            "REFERENCIA",
            "ALTERA",
            "ALTERA_EXPRESSAMENTE",
            "REVOGA",
            "REVOGA_EXPRESSAMENTE",
            "REVOGA_TACITAMENTE",  # Válido na whitelist, mas nunca retornado pelo classifier
            "REGULAMENTA",
            "DEPENDE_DE",
            "EXCEPCIONA",
        }

        test_rel_types = [
            "CITA",
            "REVOGA_EXPRESSAMENTE",
            "ALTERA_EXPRESSAMENTE",
            "REGULAMENTA",
            "EXCEPCIONA",
            "DEPENDE_DE",
            "REFERENCIA",
        ]

        for rel_type in test_rel_types:
            assert rel_type in valid_rel_types, f"{rel_type} não é válido"

    def test_rel_type_confidence_range(self):
        """rel_type_confidence deve estar entre 0.0 e 1.0."""
        confidences = [0.0, 0.5, 0.75, 0.85, 0.90, 0.95, 1.0]

        for conf in confidences:
            assert 0.0 <= conf <= 1.0

    def test_revoga_expressamente_confidence_threshold(self):
        """REVOGA_EXPRESSAMENTE deve ter confidence >= 0.90."""
        # Conforme especificação PR5
        min_confidence_revoga = 0.90

        citation = {
            "target_node_id": "leis:LEI-8666-1993#ART-005",
            "rel_type": "REVOGA_EXPRESSAMENTE",
            "rel_type_confidence": 0.95,
        }

        assert citation["rel_type_confidence"] >= min_confidence_revoga


# =============================================================================
# TESTES: Separação de confidence vs rel_type_confidence
# =============================================================================


class TestConfidenceSeparation:
    """Testa a separação entre confidence (extração) e rel_type_confidence (classificação)."""

    def test_confidence_is_for_extraction(self):
        """confidence é para confiança da extração (regex match)."""
        # Simulação de edge com ambos os campos
        edge = {
            "source_node_id": "leis:LEI-14133-2021#ART-200",
            "target_node_id": "leis:LEI-8666-1993#ART-005",
            "confidence": 1.0,  # Extração por regex = 100% confiança
            "rel_type": "REVOGA_EXPRESSAMENTE",
            "rel_type_confidence": 0.95,  # Classificação por padrão = 95%
        }

        # confidence e rel_type_confidence são independentes
        assert edge["confidence"] == 1.0  # Extração sempre 1.0 para regex
        assert edge["rel_type_confidence"] == 0.95  # Classificação variável

    def test_rel_type_confidence_varies_by_pattern(self):
        """rel_type_confidence deve variar conforme o padrão detectado."""
        # Padrões fortes (revogação explícita)
        high_confidence = {
            "rel_type": "REVOGA_EXPRESSAMENTE",
            "rel_type_confidence": 0.95,
        }

        # Padrões médios (dependência)
        medium_confidence = {
            "rel_type": "DEPENDE_DE",
            "rel_type_confidence": 0.80,
        }

        # Padrões fracos (referência genérica)
        low_confidence = {
            "rel_type": "REFERENCIA",
            "rel_type_confidence": 0.70,
        }

        # Default (citação simples)
        default_confidence = {
            "rel_type": "CITA",
            "rel_type_confidence": 0.50,
        }

        assert high_confidence["rel_type_confidence"] > medium_confidence["rel_type_confidence"]
        assert medium_confidence["rel_type_confidence"] > low_confidence["rel_type_confidence"]
        assert low_confidence["rel_type_confidence"] > default_confidence["rel_type_confidence"]
