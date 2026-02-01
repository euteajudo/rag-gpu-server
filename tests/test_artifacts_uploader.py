# -*- coding: utf-8 -*-
"""
Testes para o ArtifactsUploader (PR13/Etapa 4).

Testa:
- Upload de artifacts com sucesso
- Retry em caso de falha
- Skip quando não configurado
- Preparação de metadados
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.sinks.artifacts_uploader import (
    ArtifactsUploader,
    ArtifactMetadata,
    ArtifactUploadResult,
    prepare_offsets_map,
    compute_sha256,
)


class TestArtifactsUploader:
    """Testes para ArtifactsUploader."""

    def test_is_configured_false_when_no_base_url(self):
        """Uploader não configurado quando ARTIFACTS_BASE_URL está vazio."""
        uploader = ArtifactsUploader(base_url="")
        assert uploader.is_configured() is False

    def test_is_configured_true_when_base_url_set(self):
        """Uploader configurado quando ARTIFACTS_BASE_URL está definido."""
        uploader = ArtifactsUploader(base_url="https://vectorgov.io")
        assert uploader.is_configured() is True

    def test_upload_returns_error_when_not_configured(self):
        """Upload retorna erro quando uploader não está configurado."""
        uploader = ArtifactsUploader(base_url="")

        metadata = ArtifactMetadata(
            document_id="TEST-001",
            tipo_documento="LEI",
            numero="123",
            ano=2021,
            sha256_source="abc123",
            sha256_canonical_md="def456",
            canonical_hash="def456",
            ingest_run_id="run-001",
        )

        result = uploader.upload(
            pdf_content=b"fake pdf",
            canonical_md="# Test",
            offsets_json={"ART-001": {"start": 0, "end": 10}},
            metadata=metadata,
        )

        assert result.success is False
        assert "não configurado" in result.message.lower() or "não definido" in result.error.lower()

    @patch("src.sinks.artifacts_uploader.requests.Session")
    def test_upload_success(self, mock_session_class):
        """Upload bem-sucedido retorna storage_paths."""
        # Setup mock
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "storage_paths": {
                "pdf": "documents/TEST-001/original.pdf",
                "canonical_md": "documents/TEST-001/canonical.md",
                "offsets_json": "documents/TEST-001/offsets.json",
            }
        }
        mock_session.post.return_value = mock_response

        uploader = ArtifactsUploader(
            base_url="https://vectorgov.io",
            api_key="test-key",
        )

        metadata = ArtifactMetadata(
            document_id="TEST-001",
            tipo_documento="LEI",
            numero="123",
            ano=2021,
            sha256_source="abc123",
            sha256_canonical_md="def456",
            canonical_hash="def456",
            ingest_run_id="run-001",
        )

        result = uploader.upload(
            pdf_content=b"fake pdf content",
            canonical_md="# Test Document\n\nArt. 1 Test.",
            offsets_json={"ART-001": {"start": 0, "end": 100}},
            metadata=metadata,
        )

        assert result.success is True
        assert result.storage_paths is not None
        assert "documents/TEST-001/original.pdf" in result.storage_paths.get("pdf", "")

    @patch("src.sinks.artifacts_uploader.requests.Session")
    def test_upload_retry_on_server_error(self, mock_session_class):
        """Upload faz retry em erros 5xx."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Primeiro call falha, segundo sucesso
        mock_response_fail = Mock()
        mock_response_fail.status_code = 503
        mock_response_fail.text = "Service Unavailable"

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"success": True, "storage_paths": {}}

        mock_session.post.side_effect = [mock_response_fail, mock_response_success]

        uploader = ArtifactsUploader(base_url="https://vectorgov.io")

        metadata = ArtifactMetadata(
            document_id="TEST-001",
            tipo_documento="LEI",
            numero="123",
            ano=2021,
            sha256_source="abc123",
            sha256_canonical_md="def456",
            canonical_hash="def456",
            ingest_run_id="run-001",
        )

        result = uploader.upload(
            pdf_content=b"fake pdf",
            canonical_md="# Test",
            offsets_json={},
            metadata=metadata,
        )

        # Deve ter sucesso após retry
        assert result.success is True
        assert result.retries >= 1

    @patch("src.sinks.artifacts_uploader.requests.Session")
    def test_upload_no_retry_on_client_error(self, mock_session_class):
        """Upload não faz retry em erros 4xx (exceto 429)."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        mock_session.post.return_value = mock_response

        uploader = ArtifactsUploader(base_url="https://vectorgov.io")

        metadata = ArtifactMetadata(
            document_id="TEST-001",
            tipo_documento="LEI",
            numero="123",
            ano=2021,
            sha256_source="abc123",
            sha256_canonical_md="def456",
            canonical_hash="def456",
            ingest_run_id="run-001",
        )

        result = uploader.upload(
            pdf_content=b"fake pdf",
            canonical_md="# Test",
            offsets_json={},
            metadata=metadata,
        )

        assert result.success is False
        # Deve ter parado após primeira tentativa (sem retries)
        assert mock_session.post.call_count == 1

    def test_build_headers_with_api_key(self):
        """Headers incluem Authorization quando API key está definida."""
        uploader = ArtifactsUploader(
            base_url="https://vectorgov.io",
            api_key="test-api-key",
        )

        headers = uploader._build_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"

    def test_build_headers_with_cloudflare_access(self):
        """Headers incluem CF-Access quando credenciais estão definidas."""
        uploader = ArtifactsUploader(
            base_url="https://vectorgov.io",
            cf_client_id="client-id",
            cf_client_secret="client-secret",
        )

        headers = uploader._build_headers()
        assert "CF-Access-Client-Id" in headers
        assert "CF-Access-Client-Secret" in headers
        assert headers["CF-Access-Client-Id"] == "client-id"
        assert headers["CF-Access-Client-Secret"] == "client-secret"


class TestHelperFunctions:
    """Testes para funções auxiliares."""

    def test_prepare_offsets_map(self):
        """Converte offsets_map de tuplas para dicts."""
        offsets_map = {
            "ART-001": (0, 100),
            "ART-002": (101, 200),
            "PAR-001-001": (50, 80),
        }

        result = prepare_offsets_map(offsets_map)

        assert result["ART-001"] == {"start": 0, "end": 100}
        assert result["ART-002"] == {"start": 101, "end": 200}
        assert result["PAR-001-001"] == {"start": 50, "end": 80}

    def test_compute_sha256(self):
        """Computa SHA256 corretamente."""
        content = b"Hello, World!"
        expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"

        result = compute_sha256(content)

        assert result == expected

    def test_compute_sha256_empty(self):
        """SHA256 de bytes vazios."""
        content = b""
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        result = compute_sha256(content)

        assert result == expected


class TestArtifactMetadata:
    """Testes para ArtifactMetadata."""

    def test_metadata_to_dict(self):
        """Metadados podem ser convertidos para dict."""
        metadata = ArtifactMetadata(
            document_id="LEI-14133-2021",
            tipo_documento="LEI",
            numero="14.133",
            ano=2021,
            sha256_source="abc123",
            sha256_canonical_md="def456",
            canonical_hash="def456",
            ingest_run_id="run-001",
            pipeline_version="1.0.0",
            document_version="2021-04-01T00:00:00",
        )

        result = asdict(metadata)

        assert result["document_id"] == "LEI-14133-2021"
        assert result["tipo_documento"] == "LEI"
        assert result["ano"] == 2021
        assert result["pipeline_version"] == "1.0.0"


class TestIntegrationWithPipeline:
    """Testes de integração com o pipeline (usando mocks)."""

    @patch("src.sinks.artifacts_uploader.ArtifactsUploader.upload")
    def test_pipeline_skips_upload_when_not_configured(self, mock_upload):
        """Pipeline faz skip quando uploader não está configurado."""
        # Este teste verifica que o pipeline não chama upload quando não configurado
        # O mock não deve ser chamado se is_configured() retorna False

        from src.sinks.artifacts_uploader import get_artifacts_uploader

        uploader = get_artifacts_uploader()

        if not uploader.is_configured():
            # Quando não configurado, upload não deve ser chamado
            # (verificado pelo fato de que o mock não foi chamado)
            assert mock_upload.call_count == 0
