# -*- coding: utf-8 -*-
"""
Testes para DriftDetector (Feature 7).

3 casos:
1. Primeiro run — sem drift
2. Repeat (mesmo pdf_hash + pipeline_version + canonical_hash) — sem drift
3. Drift detectado (mesmo pdf_hash + pipeline_version, canonical_hash diferente)
"""

import pytest
import os
import tempfile
import sys
sys.path.insert(0, '/workspace/rag-gpu-server')

from src.utils.drift_detector import DriftDetector, DriftCheckResult


@pytest.fixture
def drift_detector(tmp_path):
    """DriftDetector com registry em diretório temporário."""
    registry_path = str(tmp_path / "drift_registry.json")
    return DriftDetector(registry_path=registry_path)


class TestDriftDetector:

    def test_first_run_no_drift(self, drift_detector):
        """Primeiro run: has_previous_run=False, is_drifted=False."""
        result = drift_detector.check(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            current_canonical_hash="hash_v1",
        )
        assert result.has_previous_run is False
        assert result.is_drifted is False
        assert "Primeiro run" in result.message

    def test_repeat_no_drift(self, drift_detector):
        """Repeat com mesmo canonical_hash: sem drift."""
        # Register first run
        drift_detector.register_run(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            canonical_hash="hash_v1",
            ingest_run_id="run-001",
        )

        # Check again with same hash
        result = drift_detector.check(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            current_canonical_hash="hash_v1",
        )
        assert result.has_previous_run is True
        assert result.is_drifted is False
        assert result.previous_canonical_hash == "hash_v1"
        assert result.previous_run_id == "run-001"

    def test_drift_detected(self, drift_detector):
        """Drift: mesmo pdf_hash + pipeline_version, canonical_hash diferente."""
        # Register first run
        drift_detector.register_run(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            canonical_hash="hash_v1",
            ingest_run_id="run-001",
        )

        # Check with different canonical_hash
        result = drift_detector.check(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            current_canonical_hash="hash_v2_different",
        )
        assert result.has_previous_run is True
        assert result.is_drifted is True
        assert result.previous_canonical_hash == "hash_v1"
        assert result.current_canonical_hash == "hash_v2_different"
        assert "DRIFT" in result.message

    def test_different_pipeline_version_no_drift(self, drift_detector):
        """Diferente pipeline_version: sem drift (chave diferente)."""
        # Register with v1.0.0
        drift_detector.register_run(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.0.0",
            canonical_hash="hash_v1",
            ingest_run_id="run-001",
        )

        # Check with v1.1.0 and different hash — should be first run (new key)
        result = drift_detector.check(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            current_canonical_hash="hash_v2",
        )
        assert result.has_previous_run is False
        assert result.is_drifted is False

    def test_different_pdf_hash_no_drift(self, drift_detector):
        """Diferente pdf_hash: sem drift (PDF diferente)."""
        drift_detector.register_run(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            canonical_hash="hash_v1",
            ingest_run_id="run-001",
        )

        result = drift_detector.check(
            document_id="LEI-14133-2021",
            pdf_hash="def456_different_pdf",
            pipeline_version="1.1.0",
            current_canonical_hash="hash_v2",
        )
        assert result.has_previous_run is False
        assert result.is_drifted is False

    def test_registry_persistence(self, drift_detector):
        """Registry persiste entre instâncias."""
        drift_detector.register_run(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            canonical_hash="hash_v1",
            ingest_run_id="run-001",
        )

        # Create new instance with same path
        detector2 = DriftDetector(registry_path=drift_detector._registry_path)
        result = detector2.check(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            current_canonical_hash="hash_v1",
        )
        assert result.has_previous_run is True
        assert result.is_drifted is False

    def test_corrupted_registry_recovers(self, drift_detector):
        """Registry corrompido: recria sem erro."""
        # Write garbage
        with open(drift_detector._registry_path, "w") as f:
            f.write("{corrupt json!!!")

        # Should not raise, returns first-run result
        result = drift_detector.check(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            current_canonical_hash="hash_v1",
        )
        assert result.has_previous_run is False
        assert result.is_drifted is False


# =============================================================================
# Redis backend tests (using FakeRedis mock)
# =============================================================================

class FakeRedis:
    """Minimal Redis mock for testing DriftDetector Redis backend."""

    def __init__(self):
        self._store = {}

    def get(self, key):
        entry = self._store.get(key)
        if entry is None:
            return None
        # Check TTL
        if entry["expires_at"] and time.time() > entry["expires_at"]:
            del self._store[key]
            return None
        return entry["value"]

    def setex(self, key, ttl, value):
        self._store[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
        }


import time


class TestDriftDetectorRedis:

    @pytest.fixture
    def redis_detector(self):
        """DriftDetector backed by FakeRedis."""
        fake = FakeRedis()
        return DriftDetector(redis_client=fake)

    def test_first_run_redis(self, redis_detector):
        """Redis backend: primeiro run sem drift."""
        result = redis_detector.check(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            current_canonical_hash="hash_v1",
        )
        assert result.has_previous_run is False
        assert result.is_drifted is False

    def test_repeat_no_drift_redis(self, redis_detector):
        """Redis backend: repeat sem drift."""
        redis_detector.register_run(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            canonical_hash="hash_v1",
            ingest_run_id="run-001",
        )
        result = redis_detector.check(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            current_canonical_hash="hash_v1",
        )
        assert result.has_previous_run is True
        assert result.is_drifted is False

    def test_drift_detected_redis(self, redis_detector):
        """Redis backend: drift detected."""
        redis_detector.register_run(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            canonical_hash="hash_v1",
            ingest_run_id="run-001",
        )
        result = redis_detector.check(
            document_id="LEI-14133-2021",
            pdf_hash="abc123",
            pipeline_version="1.1.0",
            current_canonical_hash="hash_v2_different",
        )
        assert result.has_previous_run is True
        assert result.is_drifted is True
        assert "DRIFT" in result.message

    def test_redis_fallback_on_error(self):
        """Redis backend: error during read falls back gracefully."""
        class BrokenRedis:
            def get(self, key):
                raise ConnectionError("Redis down")
            def setex(self, key, ttl, value):
                raise ConnectionError("Redis down")

        detector = DriftDetector(redis_client=BrokenRedis())
        # Should not raise — returns first-run (None entry)
        result = detector.check(
            document_id="TEST",
            pdf_hash="abc",
            pipeline_version="1.0",
            current_canonical_hash="hash1",
        )
        assert result.has_previous_run is False
        assert result.is_drifted is False
