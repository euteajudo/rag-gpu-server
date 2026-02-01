# -*- coding: utf-8 -*-
"""
Artifacts Uploader - Envia artefatos de evidência para a VPS (PR13/Etapa 4).

Este módulo é responsável por enviar os artefatos de ingestão para a VPS
via POST /api/v1/ingest/artifacts, garantindo que evidências estejam
armazenadas antes dos chunks irem para o Milvus.

Artefatos enviados:
- PDF original (original.pdf)
- Markdown canônico (canonical.md)
- Mapa de offsets (offsets.json)
- Metadados do documento

Configuração via env vars:
- ARTIFACTS_BASE_URL: URL base da VPS (ex: https://vectorgov.io)
- ARTIFACTS_INGEST_ENDPOINT: Endpoint (default: /api/v1/ingest/artifacts)
- VPS_API_KEY: API key para autenticação
- CF_ACCESS_CLIENT_ID: Cloudflare Access Client ID (opcional)
- CF_ACCESS_CLIENT_SECRET: Cloudflare Access Client Secret (opcional)
"""

import os
import json
import hashlib
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class ArtifactMetadata:
    """Metadados do documento para upload de artifacts."""
    document_id: str
    tipo_documento: str
    numero: str
    ano: int
    sha256_source: str  # Hash do PDF original
    sha256_canonical_md: str  # Hash do markdown canônico
    canonical_hash: str  # Hash para validação de offsets (PR13)
    ingest_run_id: str
    pipeline_version: str = "1.0.0"
    document_version: Optional[str] = None


@dataclass
class ArtifactUploadResult:
    """Resultado do upload de artifacts."""
    success: bool
    document_id: str
    message: str
    storage_paths: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    retries: int = 0


class ArtifactsUploader:
    """
    Cliente para upload de artifacts para a VPS.

    Implementa retry com backoff exponencial e validação de resposta.
    """

    DEFAULT_ENDPOINT = "/api/v1/ingest/artifacts"
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 0.5
    TIMEOUT = 60  # segundos

    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        cf_client_id: Optional[str] = None,
        cf_client_secret: Optional[str] = None,
    ):
        """
        Inicializa o uploader.

        Args:
            base_url: URL base da VPS (env: ARTIFACTS_BASE_URL)
            endpoint: Endpoint de upload (env: ARTIFACTS_INGEST_ENDPOINT)
            api_key: API key para autenticação (env: VPS_API_KEY)
            cf_client_id: Cloudflare Access Client ID (env: CF_ACCESS_CLIENT_ID)
            cf_client_secret: Cloudflare Access Client Secret (env: CF_ACCESS_CLIENT_SECRET)
        """
        self.base_url = base_url or os.getenv("ARTIFACTS_BASE_URL", "")
        self.endpoint = endpoint or os.getenv("ARTIFACTS_INGEST_ENDPOINT", self.DEFAULT_ENDPOINT)
        self.api_key = api_key or os.getenv("VPS_API_KEY", "")
        self.cf_client_id = cf_client_id or os.getenv("CF_ACCESS_CLIENT_ID", "")
        self.cf_client_secret = cf_client_secret or os.getenv("CF_ACCESS_CLIENT_SECRET", "")

        # Configura session com retry
        self.session = self._create_session()

        logger.info(
            f"ArtifactsUploader inicializado: base_url={self.base_url}, "
            f"endpoint={self.endpoint}, "
            f"cf_access={'enabled' if self.cf_client_id else 'disabled'}"
        )

    def _create_session(self) -> requests.Session:
        """Cria session HTTP com retry configurado."""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=self.BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _build_headers(self) -> Dict[str, str]:
        """Constrói headers de autenticação."""
        headers = {}

        # Ingest Service Key (header dedicado para /ingest/artifacts)
        if self.api_key:
            headers["X-Ingest-Key"] = self.api_key

        # Cloudflare Access (se configurado)
        if self.cf_client_id and self.cf_client_secret:
            headers["CF-Access-Client-Id"] = self.cf_client_id
            headers["CF-Access-Client-Secret"] = self.cf_client_secret

        return headers

    def is_configured(self) -> bool:
        """Verifica se o uploader está configurado."""
        return bool(self.base_url)

    def upload(
        self,
        pdf_content: bytes,
        canonical_md: str,
        offsets_json: Dict[str, Any],
        metadata: ArtifactMetadata,
    ) -> ArtifactUploadResult:
        """
        Faz upload dos artifacts para a VPS.

        Args:
            pdf_content: Bytes do PDF original
            canonical_md: Markdown canônico (string)
            offsets_json: Dicionário com mapa de offsets + canonical_hash
            metadata: Metadados do documento

        Returns:
            ArtifactUploadResult com status do upload
        """
        if not self.is_configured():
            logger.warning("ArtifactsUploader não configurado (ARTIFACTS_BASE_URL vazio)")
            return ArtifactUploadResult(
                success=False,
                document_id=metadata.document_id,
                message="Uploader não configurado",
                error="ARTIFACTS_BASE_URL não definido",
            )

        url = f"{self.base_url.rstrip('/')}{self.endpoint}"

        # Prepara offsets.json com canonical_hash e document_id
        offsets_payload = {
            "document_id": metadata.document_id,
            "canonical_hash": metadata.canonical_hash,
            "offsets": offsets_json,
            "pipeline_version": metadata.pipeline_version,
        }

        # Prepara arquivos para multipart
        files = {
            "pdf_file": ("original.pdf", pdf_content, "application/pdf"),
            "canonical_md_file": ("canonical.md", canonical_md.encode("utf-8"), "text/markdown"),
            "offsets_json_file": ("offsets.json", json.dumps(offsets_payload, ensure_ascii=False).encode("utf-8"), "application/json"),
        }

        # Prepara form data com metadados
        form_data = {
            "document_id": metadata.document_id,
            "tipo_documento": metadata.tipo_documento,
            "numero": metadata.numero,
            "ano": str(metadata.ano),
            "sha256_source": metadata.sha256_source,
            "sha256_canonical_md": metadata.sha256_canonical_md,
            "canonical_hash": metadata.canonical_hash,
            "ingest_run_id": metadata.ingest_run_id,
            "pipeline_version": metadata.pipeline_version,
        }

        if metadata.document_version:
            form_data["document_version"] = metadata.document_version

        headers = self._build_headers()

        logger.info(
            f"Uploading artifacts para {url}: "
            f"document_id={metadata.document_id}, "
            f"pdf={len(pdf_content)} bytes, "
            f"canonical_md={len(canonical_md)} chars, "
            f"offsets={len(offsets_json)} spans"
        )

        retries = 0
        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self.session.post(
                    url,
                    files=files,
                    data=form_data,
                    headers=headers,
                    timeout=self.TIMEOUT,
                )

                if response.status_code in (200, 201):
                    result_data = response.json()
                    logger.info(
                        f"Artifacts uploaded com sucesso: "
                        f"document_id={metadata.document_id}, "
                        f"response={result_data}"
                    )
                    return ArtifactUploadResult(
                        success=True,
                        document_id=metadata.document_id,
                        message="Artifacts uploaded com sucesso",
                        storage_paths=result_data.get("storage_paths"),
                        retries=retries,
                    )
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
                    logger.warning(f"Upload falhou (attempt {attempt + 1}): {error_msg}")
                    last_error = error_msg
                    retries += 1

                    # Não retry em erros 4xx (exceto 429)
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        break

                    # Backoff antes de retry
                    if attempt < self.MAX_RETRIES:
                        sleep_time = self.BACKOFF_FACTOR * (2 ** attempt)
                        time.sleep(sleep_time)

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Upload exception (attempt {attempt + 1}): {e}")
                retries += 1

                if attempt < self.MAX_RETRIES:
                    sleep_time = self.BACKOFF_FACTOR * (2 ** attempt)
                    time.sleep(sleep_time)

        logger.error(
            f"Upload de artifacts falhou após {retries} tentativas: "
            f"document_id={metadata.document_id}, error={last_error}"
        )

        return ArtifactUploadResult(
            success=False,
            document_id=metadata.document_id,
            message=f"Upload falhou após {retries} tentativas",
            error=last_error,
            retries=retries,
        )


def compute_sha256(content: bytes) -> str:
    """Computa SHA256 de bytes."""
    return hashlib.sha256(content).hexdigest()


def prepare_offsets_map(
    offsets_map: Dict[str, Tuple[int, int]],
) -> Dict[str, Dict[str, int]]:
    """
    Converte offsets_map para formato JSON-serializable.

    Args:
        offsets_map: dict[span_id, (start, end)]

    Returns:
        dict[span_id, {"start": int, "end": int}]
    """
    return {
        span_id: {"start": start, "end": end}
        for span_id, (start, end) in offsets_map.items()
    }


# Singleton
_uploader: Optional[ArtifactsUploader] = None


def get_artifacts_uploader() -> ArtifactsUploader:
    """Retorna instância singleton do uploader."""
    global _uploader
    if _uploader is None:
        _uploader = ArtifactsUploader()
    return _uploader
