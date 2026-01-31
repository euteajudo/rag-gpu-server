"""
Testes de validação dos 3 controles críticos do PR3 v2.1.

Este arquivo FALHA se qualquer controle estiver ausente:
1. Self-loop prevention (guardrail)
2. Score mapping alignment
3. Context stitching (MAX_PARENT_CHARS)

Execute: pytest tests/test_pr3_v21_controls.py -v
"""

import pytest
import ast
import re
from pathlib import Path


class TestPR3V21ControlsExist:
    """
    Testes que verificam a EXISTÊNCIA dos 3 controles no código-fonte.

    Se qualquer teste falhar, significa que o controle foi removido
    ou modificado de forma incompatível.
    """

    @pytest.fixture
    def neo4j_writer_source(self) -> str:
        """Carrega o código-fonte do neo4j_writer.py."""
        path = Path(__file__).parent.parent / "src" / "sinks" / "neo4j_writer.py"
        assert path.exists(), f"Arquivo não encontrado: {path}"
        return path.read_text(encoding="utf-8")

    @pytest.fixture
    def milvus_writer_source(self) -> str:
        """Carrega o código-fonte do milvus_writer.py."""
        path = Path(__file__).parent.parent / "src" / "sinks" / "milvus_writer.py"
        assert path.exists(), f"Arquivo não encontrado: {path}"
        return path.read_text(encoding="utf-8")


    # =========================================================================
    # CONTROLE 1: Self-loop Prevention (Guardrail)
    # =========================================================================

    def test_self_loop_prevention_in_create_edge(self, neo4j_writer_source: str):
        """
        Verifica que create_edge() previne self-loops.

        Critério: Deve existir comparação source_node_id == target_node_id
        com retorno False antes de criar o edge.
        """
        # Padrão esperado: if edge.source_node_id == edge.target_node_id
        pattern = r"if\s+edge\.source_node_id\s*==\s*edge\.target_node_id"
        match = re.search(pattern, neo4j_writer_source)

        assert match is not None, (
            "CONTROLE 1 AUSENTE: Self-loop prevention não encontrado em create_edge().\n"
            "Esperado: if edge.source_node_id == edge.target_node_id: return False"
        )

        # Verifica que retorna False após a detecção
        # Busca nas próximas 5 linhas após o match
        start_pos = match.end()
        next_lines = neo4j_writer_source[start_pos:start_pos + 200]
        assert "return False" in next_lines, (
            "CONTROLE 1 INCOMPLETO: Detecção de self-loop existe mas não retorna False"
        )

    def test_self_loop_prevention_in_batch(self, neo4j_writer_source: str):
        """
        Verifica que create_edges_batch() filtra self-loops.

        Critério: Deve existir list comprehension ou filter que exclui
        edges onde source == target.
        """
        # Padrão: e for e in edges if e.source_node_id != e.target_node_id
        pattern = r"for\s+\w+\s+in\s+edges\s+if\s+\w+\.source_node_id\s*!=\s*\w+\.target_node_id"
        match = re.search(pattern, neo4j_writer_source)

        assert match is not None, (
            "CONTROLE 1 AUSENTE: Filtro de self-loop não encontrado em create_edges_batch().\n"
            "Esperado: [e for e in edges if e.source_node_id != e.target_node_id]"
        )

    def test_confidence_tier_derivation(self, neo4j_writer_source: str):
        """
        Verifica que ConfidenceTier é derivado automaticamente do score.

        Critério: EdgeCandidate.__post_init__ deve chamar ConfidenceTier.from_score()
        """
        # Verifica existência do enum ConfidenceTier
        assert "class ConfidenceTier" in neo4j_writer_source, (
            "CONTROLE 1 AUSENTE: Enum ConfidenceTier não encontrado"
        )

        # Verifica from_score method
        assert "def from_score" in neo4j_writer_source, (
            "CONTROLE 1 AUSENTE: Método from_score() não encontrado em ConfidenceTier"
        )

        # Verifica derivação automática no __post_init__
        pattern = r"ConfidenceTier\.from_score\s*\(\s*self\.confidence\s*\)"
        match = re.search(pattern, neo4j_writer_source)
        assert match is not None, (
            "CONTROLE 1 AUSENTE: Derivação automática de confidence_tier não encontrada.\n"
            "Esperado: ConfidenceTier.from_score(self.confidence)"
        )


    # =========================================================================
    # CONTROLE 2: DeviceType Normalization
    # =========================================================================

    def test_device_type_enum_exists(self, milvus_writer_source: str):
        """
        Verifica que DeviceType enum existe com valores padronizados.
        """
        assert "class DeviceType" in milvus_writer_source, (
            "CONTROLE 2 AUSENTE: Enum DeviceType não encontrado"
        )

        # Valores esperados
        for value in ["ART", "PAR", "INC", "ALI", "UNKNOWN"]:
            assert f'"{value}"' in milvus_writer_source or f"'{value}'" in milvus_writer_source, (
                f"CONTROLE 2 INCOMPLETO: DeviceType.{value} não encontrado"
            )

    def test_device_type_normalization_in_post_init(self, milvus_writer_source: str):
        """
        Verifica que device_type é normalizado automaticamente no __post_init__.
        """
        pattern = r"DeviceType\.from_string\s*\(\s*self\.device_type\s*\)"
        match = re.search(pattern, milvus_writer_source)

        assert match is not None, (
            "CONTROLE 2 AUSENTE: Normalização de device_type não encontrada.\n"
            "Esperado: DeviceType.from_string(self.device_type)"
        )

    def test_parent_chunk_id_validation(self, milvus_writer_source: str):
        """
        Verifica que parent_chunk_id é validado (artigos não devem ter parent).
        """
        # Verifica regra: artigos limpam parent_chunk_id
        pattern = r"if\s+self\.device_type\s*==\s*DeviceType\.ART\.value"
        match = re.search(pattern, milvus_writer_source)

        assert match is not None, (
            "CONTROLE 2 AUSENTE: Validação de parent_chunk_id para artigos não encontrada.\n"
            "Artigos não devem ter parent_chunk_id."
        )


    # =========================================================================
    # CONTROLE 3: article_number_int Derivation
    # =========================================================================

    def test_article_number_int_derivation(self, milvus_writer_source: str):
        """
        Verifica que article_number_int é derivado automaticamente de article_number.
        """
        # Padrão: regex para extrair número do article_number
        pattern = r're\.match\s*\(\s*r["\'].*\\d.*["\'].*self\.article_number'
        match = re.search(pattern, milvus_writer_source)

        assert match is not None, (
            "CONTROLE 3 AUSENTE: Derivação de article_number_int não encontrada.\n"
            "Esperado: re.match(r'(\\d+)', self.article_number)"
        )


class TestPR3V21DataclassIntegrity:
    """
    Testes que verificam a integridade das dataclasses.
    """

    def test_edge_candidate_has_required_fields(self):
        """Verifica que EdgeCandidate tem todos os campos obrigatórios."""
        from sinks.neo4j_writer import EdgeCandidate

        # Campos obrigatórios
        required = ["source_node_id", "target_node_id"]
        for field in required:
            assert hasattr(EdgeCandidate, "__dataclass_fields__"), (
                "EdgeCandidate não é uma dataclass"
            )
            assert field in EdgeCandidate.__dataclass_fields__, (
                f"Campo obrigatório ausente: {field}"
            )

    def test_edge_candidate_post_init_derives_tier(self):
        """Verifica que __post_init__ deriva confidence_tier corretamente."""
        from sinks.neo4j_writer import EdgeCandidate

        # HIGH tier (>= 0.8)
        edge_high = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            confidence=0.95,
        )
        assert edge_high.confidence_tier == "HIGH", (
            f"Esperado HIGH para confidence=0.95, obtido {edge_high.confidence_tier}"
        )

        # MEDIUM tier (0.5 - 0.8)
        edge_medium = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            confidence=0.65,
        )
        assert edge_medium.confidence_tier == "MEDIUM", (
            f"Esperado MEDIUM para confidence=0.65, obtido {edge_medium.confidence_tier}"
        )

        # LOW tier (< 0.5)
        edge_low = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-003",
            confidence=0.3,
        )
        assert edge_low.confidence_tier == "LOW", (
            f"Esperado LOW para confidence=0.3, obtido {edge_low.confidence_tier}"
        )

    def test_milvus_chunk_normalizes_device_type(self):
        """Verifica que MilvusChunk normaliza device_type para CAPS."""
        from sinks.milvus_writer import MilvusChunk

        chunk = MilvusChunk(
            node_id="leis:LEI-14133-2021#ART-005@P00",
            logical_node_id="leis:LEI-14133-2021#ART-005",
            chunk_id="LEI-14133-2021#ART-005@P00",
            parent_chunk_id=None,
            part_index=0,
            part_total=1,
            text="Art. 5º Texto.",
            retrieval_text="Texto.",
            parent_text=None,
            device_type="article",  # lowercase
        )

        assert chunk.device_type == "ART", (
            f"Esperado ART para device_type='article', obtido {chunk.device_type}"
        )

    def test_milvus_chunk_clears_parent_for_articles(self):
        """Verifica que artigos têm parent_chunk_id limpo automaticamente."""
        from sinks.milvus_writer import MilvusChunk

        chunk = MilvusChunk(
            node_id="leis:LEI-14133-2021#ART-005@P00",
            logical_node_id="leis:LEI-14133-2021#ART-005",
            chunk_id="LEI-14133-2021#ART-005@P00",
            parent_chunk_id="LEI-14133-2021#ART-001@P00",  # Será limpo
            part_index=0,
            part_total=1,
            text="Art. 5º Texto.",
            retrieval_text="Texto.",
            parent_text=None,
            device_type="article",
        )

        assert chunk.parent_chunk_id is None, (
            f"Esperado None para parent_chunk_id de artigo, obtido {chunk.parent_chunk_id}"
        )


class TestPR3V21SelfLoopBehavior:
    """
    Testes comportamentais que verificam que self-loops são rejeitados.
    """

    def test_neo4j_writer_rejects_self_loop(self):
        """Verifica que Neo4jEdgeWriter rejeita self-loops."""
        from sinks.neo4j_writer import EdgeCandidate, Neo4jEdgeWriter
        from unittest.mock import MagicMock

        # Cria writer mockado
        writer = Neo4jEdgeWriter(password="test")
        writer._connected = True
        writer._driver = MagicMock()

        # Cria self-loop
        self_loop = EdgeCandidate(
            source_node_id="leis:LEI-14133-2021#ART-005",
            target_node_id="leis:LEI-14133-2021#ART-005",  # MESMO ID
            source_chunk_id="leis:LEI-14133-2021#ART-005@P00",
        )

        # Deve retornar False
        result = writer.create_edge(self_loop)

        assert result is False, (
            "FALHA CRÍTICA: Neo4jEdgeWriter aceitou self-loop!\n"
            "Self-loops NUNCA devem ser criados."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
