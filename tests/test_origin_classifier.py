"""
Testes para o OriginClassifier.

Verifica a classificacao de chunks por origem material:
- "self": Material da propria lei
- "external": Material de outras leis citadas/modificadas

@author: Claude (RunPod)
@date: 2026-02-04
"""

import pytest
import sys
sys.path.insert(0, '/workspace/rag-gpu-server')

from src.chunking.origin_classifier import (
    OriginClassifier,
    OriginRule,
    classify_chunk_origins,
    is_external_material,
    get_origin_warning,
)
from src.chunking.chunk_materializer import MaterializedChunk


class TestOriginClassifierBasic:
    """Testes basicos do OriginClassifier."""

    @pytest.fixture
    def classifier(self):
        return OriginClassifier()

    def test_init_loads_default_rules(self, classifier):
        """Deve carregar regras default na inicializacao."""
        assert len(classifier.rules) > 0
        assert all(isinstance(r, OriginRule) for r in classifier.rules)

    def test_self_origin_regular_article(self, classifier):
        """Artigo normal da lei deve ser classificado como 'self'."""
        chunk = {"text": "Art. 1º Esta Lei estabelece normas gerais de licitação."}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "self"
        assert result["is_external_material"] == False
        assert result["origin_reference"] is None
        assert result["origin_confidence"] == "high"

    def test_self_origin_inciso(self, classifier):
        """Inciso normal deve ser classificado como 'self'."""
        chunk = {"text": "I - principio da legalidade;"}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "self"
        assert result["is_external_material"] == False

    def test_empty_text(self, classifier):
        """Texto vazio deve retornar 'self' como default."""
        chunk = {"text": ""}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "self"
        assert result["is_external_material"] == False


class TestCodigoPenalDetection:
    """Testes de deteccao do Codigo Penal."""

    @pytest.fixture
    def classifier(self):
        return OriginClassifier()

    def test_art337e_high_confidence(self, classifier):
        """Art. 337-E deve ser detectado como Codigo Penal com alta confianca."""
        chunk = {"text": "Art. 337-E. Admitir, possibilitar ou dar causa à contratação direta..."}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "DL-2848-1940"
        assert result["origin_reference_name"] == "Codigo Penal"
        assert result["is_external_material"] == True
        assert result["origin_confidence"] == "high"
        assert "codigo_penal_art337" in result["origin_reason"]

    def test_art337f_to_337p(self, classifier):
        """Todos os Art. 337-* devem ser detectados."""
        letters = ["E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
        for letter in letters:
            chunk = {"text": f"Art. 337-{letter}. Texto do artigo..."}
            result = classifier.classify(chunk)
            assert result["origin_type"] == "external", f"Art. 337-{letter} nao detectado"
            assert result["origin_reference"] == "DL-2848-1940"

    def test_decreto_lei_2848_mention(self, classifier):
        """Mencao ao Decreto-Lei 2.848 deve ser detectada."""
        chunk = {"text": "O Decreto-Lei nº 2.848, de 1940 (Código Penal), passa a vigorar..."}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "DL-2848-1940"
        assert result["origin_confidence"] == "high"

    def test_codigo_penal_mention_medium_confidence(self, classifier):
        """Mencao generica ao 'Codigo Penal' deve ter confianca media."""
        chunk = {"text": "conforme o Código Penal estabelece..."}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "DL-2848-1940"
        assert result["origin_confidence"] == "medium"


class TestOtherLawsDetection:
    """Testes de deteccao de outras leis."""

    @pytest.fixture
    def classifier(self):
        return OriginClassifier()

    def test_cpc_lei_13105(self, classifier):
        """Lei 13.105 (CPC) deve ser detectada."""
        chunk = {"text": "conforme a Lei nº 13.105, de 2015..."}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "LEI-13105-2015"

    def test_cpc_mention(self, classifier):
        """Mencao ao 'Codigo de Processo Civil' deve ser detectada."""
        chunk = {"text": "nos termos do Código de Processo Civil..."}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "LEI-13105-2015"

    def test_lei_8666_low_confidence(self, classifier):
        """Lei 8.666 deve ser detectada com baixa confianca (apenas mencao)."""
        chunk = {"text": "A Lei nº 8.666, de 1993, fica revogada."}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "LEI-8666-1993"
        assert result["origin_confidence"] == "low"

    def test_lei_10520(self, classifier):
        """Lei 10.520 (Pregao) deve ser detectada."""
        chunk = {"text": "revoga-se a Lei nº 10.520, de 2002..."}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "LEI-10520-2002"

    def test_lindb(self, classifier):
        """LINDB (Decreto-Lei 4.657) deve ser detectada."""
        chunk = {"text": "conforme o Decreto-Lei nº 4.657, de 1942..."}
        result = classifier.classify(chunk)

        assert result["origin_type"] == "external"
        assert result["origin_reference"] == "DL-4657-1942"


class TestBatchClassification:
    """Testes de classificacao em batch."""

    @pytest.fixture
    def classifier(self):
        return OriginClassifier()

    def test_batch_mixed_chunks(self, classifier):
        """Batch com chunks mistos deve retornar estatisticas corretas."""
        chunks = [
            {"text": "Art. 1º Esta Lei estabelece normas gerais."},
            {"text": "Art. 337-E. Admitir, possibilitar..."},
            {"text": "Art. 2º Aplicam-se as disposições."},
            {"text": "Art. 337-F. Frustrar ou fraudar..."},
            {"text": "A Lei 8.666 fica revogada."},
        ]

        results, stats = classifier.classify_batch(chunks)

        assert stats["total"] == 5
        assert stats["self"] == 2
        assert stats["external"] == 3
        assert "DL-2848-1940" in stats["external_refs"]
        assert stats["external_refs"]["DL-2848-1940"] == 2

    def test_batch_all_self(self, classifier):
        """Batch com todos chunks 'self'."""
        chunks = [
            {"text": "Art. 1º Texto normal."},
            {"text": "Art. 2º Outro texto normal."},
            {"text": "I - inciso comum;"},
        ]

        results, stats = classifier.classify_batch(chunks)

        assert stats["total"] == 3
        assert stats["self"] == 3
        assert stats["external"] == 0
        assert len(stats["external_refs"]) == 0


class TestMaterializedChunkClassification:
    """Testes com MaterializedChunk (dataclass)."""

    @pytest.fixture
    def classifier(self):
        return OriginClassifier()

    def test_classify_materialized_chunk_self(self, classifier):
        """MaterializedChunk normal deve ser classificado como 'self'."""
        chunk = MaterializedChunk(
            node_id="leis:LEI-14133#ART-001",
            chunk_id="LEI-14133#ART-001",
            parent_node_id="",
            span_id="ART-001",
            device_type="article",
            chunk_level="article",
            text="Art. 1º Esta Lei estabelece normas gerais de licitação."
        )

        classifier.classify_materialized_chunk(chunk)

        assert chunk.origin_type == "self"
        assert chunk.is_external_material == False
        assert chunk.origin_reference is None

    def test_classify_materialized_chunk_external(self, classifier):
        """MaterializedChunk com Art. 337-E deve ser classificado como 'external'."""
        chunk = MaterializedChunk(
            node_id="leis:LEI-14133#CIT-337E",
            chunk_id="LEI-14133#CIT-337E",
            parent_node_id="",
            span_id="CIT-337E",
            device_type="article",
            chunk_level="article",
            text="Art. 337-E. Admitir, possibilitar ou dar causa à contratação direta..."
        )

        classifier.classify_materialized_chunk(chunk)

        assert chunk.origin_type == "external"
        assert chunk.is_external_material == True
        assert chunk.origin_reference == "DL-2848-1940"
        assert chunk.origin_reference_name == "Codigo Penal"

    def test_classify_materialized_batch(self, classifier):
        """Batch de MaterializedChunk deve ser classificado corretamente."""
        chunks = [
            MaterializedChunk(
                node_id="test1", chunk_id="test1", parent_node_id="",
                span_id="ART-001", device_type="article", chunk_level="article",
                text="Art. 1º Texto normal."
            ),
            MaterializedChunk(
                node_id="test2", chunk_id="test2", parent_node_id="",
                span_id="CIT-337E", device_type="article", chunk_level="article",
                text="Art. 337-E. Texto do Codigo Penal."
            ),
        ]

        stats = classifier.classify_materialized_batch(chunks)

        assert stats["self"] == 1
        assert stats["external"] == 1
        assert chunks[0].origin_type == "self"
        assert chunks[1].origin_type == "external"


class TestPriorityOrder:
    """Testes de ordem de prioridade das regras."""

    @pytest.fixture
    def classifier(self):
        return OriginClassifier()

    def test_specific_rule_wins_over_generic(self, classifier):
        """Regra especifica (Art. 337-*) deve ter prioridade sobre generica (Codigo Penal)."""
        # Texto que matcharia ambas as regras
        chunk = {"text": "Art. 337-E. Segundo o Código Penal..."}
        result = classifier.classify(chunk)

        # Deve usar a regra mais especifica (art337), nao a generica (mention)
        assert "art337" in result["origin_reason"]
        assert result["origin_confidence"] == "high"  # art337 e high, mention e medium


class TestHelperFunctions:
    """Testes das funcoes auxiliares."""

    def test_classify_chunk_origins(self):
        """Funcao utilitaria deve funcionar corretamente."""
        chunks = [
            {"text": "Art. 1º Normal."},
            {"text": "Art. 337-E. External."},
        ]

        results, stats = classify_chunk_origins(chunks)

        assert stats["self"] == 1
        assert stats["external"] == 1

    def test_is_external_material_true(self):
        """is_external_material deve retornar True para chunks externos."""
        chunk = {"origin_type": "external", "is_external_material": True}
        assert is_external_material(chunk) == True

    def test_is_external_material_false(self):
        """is_external_material deve retornar False para chunks self."""
        chunk = {"origin_type": "self", "is_external_material": False}
        assert is_external_material(chunk) == False

    def test_get_origin_warning_external(self):
        """get_origin_warning deve retornar mensagem para chunks externos."""
        chunk = {
            "origin_type": "external",
            "is_external_material": True,
            "origin_reference": "DL-2848-1940",
            "origin_reference_name": "Codigo Penal",
        }

        warning = get_origin_warning(chunk)

        assert warning is not None
        assert "Codigo Penal" in warning
        assert "DL-2848-1940" in warning

    def test_get_origin_warning_self(self):
        """get_origin_warning deve retornar None para chunks self."""
        chunk = {"origin_type": "self", "is_external_material": False}

        warning = get_origin_warning(chunk)

        assert warning is None


class TestEdgeCases:
    """Testes de casos de borda."""

    @pytest.fixture
    def classifier(self):
        return OriginClassifier()

    def test_case_insensitive(self, classifier):
        """Deteccao deve ser case-insensitive."""
        variations = [
            "ART. 337-E. Texto...",
            "art. 337-e. Texto...",
            "Art. 337-E. Texto...",
        ]

        for text in variations:
            chunk = {"text": text}
            result = classifier.classify(chunk)
            assert result["origin_type"] == "external", f"Falhou para: {text}"

    def test_codigo_with_accent(self, classifier):
        """Deve detectar 'Código' com acento."""
        chunk = {"text": "conforme o Código Penal estabelece..."}
        result = classifier.classify(chunk)
        assert result["origin_type"] == "external"

    def test_codigo_without_accent(self, classifier):
        """Deve detectar 'Codigo' sem acento."""
        chunk = {"text": "conforme o Codigo Penal estabelece..."}
        result = classifier.classify(chunk)
        assert result["origin_type"] == "external"

    def test_partial_match_not_triggered(self, classifier):
        """Texto parcial nao deve disparar regra incorretamente."""
        # "3378" nao deve ser confundido com "337-*"
        chunk = {"text": "Art. 3378. Texto qualquer..."}
        result = classifier.classify(chunk)
        # Deve ser self (numero 3378 nao existe, mas nao e 337-*)
        # Na verdade vai matchear como self porque nao tem "-"
        assert result["origin_type"] == "self"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
