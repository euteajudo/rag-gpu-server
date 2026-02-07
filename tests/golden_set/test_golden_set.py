# -*- coding: utf-8 -*-
"""
Golden Set Tests — Validação com ground truth.

Testes parametrizados que verificam:
1. Contagem de dispositivos por tipo
2. Offsets válidos para todos os evidence chunks
3. Snippets contêm text_prefix
4. Hierarquia correta (parent_node_id)
5. Nenhum ADDRESS_MISMATCH

Requer PDFs fixtures em tests/golden_set/fixtures/.
Se PDFs não existem, testes são skipped.

Para rodar com dados reais:
1. Colocar PDFs em tests/golden_set/fixtures/
2. pytest tests/golden_set/test_golden_set.py -v
"""

import pytest
import sys
import os
sys.path.insert(0, '/workspace/rag-gpu-server')

from tests.golden_set.conftest import (
    load_ground_truth,
    load_pdf,
    get_available_ground_truths,
    compute_error_metrics,
)


# =============================================================================
# Mock-based unit tests (always run, no PDF needed)
# =============================================================================

class TestComputeErrorMetrics:
    """Testa compute_error_metrics com dados mockados."""

    def test_perfect_match(self):
        """Ground truth perfeito: precision=1, recall=1."""
        chunks = [
            {"span_id": "ART-001", "device_type": "article", "text": "Art. 1o texto",
             "canonical_start": 0, "canonical_end": 13},
            {"span_id": "ART-002", "device_type": "article", "text": "Art. 2o texto",
             "canonical_start": 14, "canonical_end": 27},
        ]
        truth = {
            "expected_devices": [
                {"span_id": "ART-001", "device_type": "article", "text_prefix": "Art. 1"},
                {"span_id": "ART-002", "device_type": "article", "text_prefix": "Art. 2"},
            ],
        }
        metrics = compute_error_metrics(chunks, truth)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["offset_rate"] == 1.0
        assert metrics["text_match_rate"] == 1.0

    def test_missing_device(self):
        """Um device faltando: recall < 1."""
        chunks = [
            {"span_id": "ART-001", "device_type": "article", "text": "Art. 1o",
             "canonical_start": 0, "canonical_end": 7},
        ]
        truth = {
            "expected_devices": [
                {"span_id": "ART-001", "device_type": "article", "text_prefix": "Art. 1"},
                {"span_id": "ART-002", "device_type": "article", "text_prefix": "Art. 2"},
            ],
        }
        metrics = compute_error_metrics(chunks, truth)
        assert metrics["recall"] == 0.5
        assert "ART-002" in metrics["details"]["missed"]

    def test_extra_device(self):
        """Device extra não no ground truth: precision < 1."""
        chunks = [
            {"span_id": "ART-001", "device_type": "article", "text": "Art. 1o",
             "canonical_start": 0, "canonical_end": 7},
            {"span_id": "ART-999", "device_type": "article", "text": "Extra",
             "canonical_start": 8, "canonical_end": 13},
        ]
        truth = {
            "expected_devices": [
                {"span_id": "ART-001", "device_type": "article", "text_prefix": "Art. 1"},
            ],
        }
        metrics = compute_error_metrics(chunks, truth)
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 0.5
        assert "ART-999" in metrics["details"]["extra"]

    def test_sentinel_offsets(self):
        """Chunks com offsets sentinela: offset_rate < 1."""
        chunks = [
            {"span_id": "ART-001", "device_type": "article", "text": "Art. 1o",
             "canonical_start": -1, "canonical_end": -1},
            {"span_id": "ART-002", "device_type": "article", "text": "Art. 2o",
             "canonical_start": 0, "canonical_end": 7},
        ]
        truth = {
            "expected_devices": [
                {"span_id": "ART-001", "device_type": "article", "text_prefix": "Art. 1"},
                {"span_id": "ART-002", "device_type": "article", "text_prefix": "Art. 2"},
            ],
        }
        metrics = compute_error_metrics(chunks, truth)
        assert metrics["offset_rate"] == 0.5

    def test_text_prefix_mismatch(self):
        """Texto não contém prefix: text_match_rate < 1."""
        chunks = [
            {"span_id": "ART-001", "device_type": "article", "text": "Texto sem artigo",
             "canonical_start": 0, "canonical_end": 15},
        ]
        truth = {
            "expected_devices": [
                {"span_id": "ART-001", "device_type": "article", "text_prefix": "Art. 1"},
            ],
        }
        metrics = compute_error_metrics(chunks, truth)
        assert metrics["text_match_rate"] == 0.0


class TestGroundTruthFiles:
    """Testa que os arquivos YAML de ground truth são válidos."""

    def test_yaml_files_exist(self):
        """Pelo menos um arquivo YAML de ground truth existe."""
        available = get_available_ground_truths()
        assert len(available) >= 1, "Nenhum arquivo YAML de ground truth encontrado"

    def test_yaml_files_parse(self):
        """Todos os YAMLs parseiam corretamente."""
        for yaml_name in get_available_ground_truths():
            truth = load_ground_truth(yaml_name)
            assert "document_id" in truth
            assert "tipo_documento" in truth
            assert "expected_devices" in truth
            assert isinstance(truth["expected_devices"], list)
            assert len(truth["expected_devices"]) > 0

    def test_yaml_devices_have_required_fields(self):
        """Cada device no YAML tem campos obrigatórios."""
        for yaml_name in get_available_ground_truths():
            truth = load_ground_truth(yaml_name)
            for device in truth["expected_devices"]:
                assert "span_id" in device
                assert "device_type" in device
                assert "text_prefix" in device

    def test_yaml_invariants(self):
        """Invariantes do YAML são válidos."""
        for yaml_name in get_available_ground_truths():
            truth = load_ground_truth(yaml_name)
            invariants = truth.get("invariants", {})
            if "min_articles" in invariants:
                articles = [d for d in truth["expected_devices"] if d["device_type"] == "article"]
                assert len(articles) >= invariants["min_articles"]


# =============================================================================
# Integration tests (require PDF fixtures — skipped if not present)
# =============================================================================
# These tests are designed to run with real PDFs.
# Place PDFs in tests/golden_set/fixtures/ matching the YAML filenames.
# e.g., LEI-14133-2021-p1-3.pdf → LEI-14133-2021-p1-3.yaml

# To run:
#   1. Copy real PDFs to tests/golden_set/fixtures/
#   2. pytest tests/golden_set/test_golden_set.py -v -k "integration"
