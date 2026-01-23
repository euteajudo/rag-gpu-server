"""
Remote Embedder - Cliente para GPU Server (BGE-M3).

Chama o endpoint /embed do GPU Server para gerar embeddings
dense (1024d) e sparse.

Uso:
    from remote import RemoteEmbedder

    embedder = RemoteEmbedder()
    result = embedder.encode(["texto1", "texto2"])

    # Acessa resultados
    dense = result.dense_embeddings[0]  # list[float] 1024d
    sparse = result.sparse_embeddings[0]  # dict[int, float]
"""

import os
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RemoteEmbedderConfig:
    """Configuração do cliente de embeddings remoto."""

    gpu_server_url: str = "http://localhost:8000"
    gpu_api_key: str = ""  # API Key para autenticação no GPU Server
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "RemoteEmbedderConfig":
        """Carrega configuração de variáveis de ambiente."""
        return cls(
            gpu_server_url=os.getenv("GPU_SERVER_URL", "http://localhost:8000"),
            gpu_api_key=os.getenv("GPU_API_KEY", ""),
            timeout=float(os.getenv("GPU_SERVER_TIMEOUT", "60")),
            max_retries=int(os.getenv("GPU_SERVER_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("GPU_SERVER_RETRY_DELAY", "1.0")),
        )


@dataclass
class EmbeddingResult:
    """Resultado do embedding remoto."""

    dense_embeddings: list[list[float]]
    sparse_embeddings: list[dict[int, float]]
    latency_ms: float
    count: int


class RemoteEmbedder:
    """
    Cliente para GPU Server - endpoint /embed.

    Gera embeddings usando BGE-M3 no servidor GPU remoto.
    Compatível com a interface do BGEM3Embedder local.
    """

    def __init__(self, config: Optional[RemoteEmbedderConfig] = None):
        self.config = config or RemoteEmbedderConfig.from_env()
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Cliente HTTP com lazy initialization."""
        if self._client is None:
            headers = {}
            if self.config.gpu_api_key:
                headers["X-GPU-API-Key"] = self.config.gpu_api_key
            self._client = httpx.Client(
                timeout=self.config.timeout,
                limits=httpx.Limits(max_connections=10),
                headers=headers,
            )
        return self._client

    def encode(
        self,
        texts: list[str],
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> EmbeddingResult:
        """
        Gera embeddings para lista de textos via GPU Server.

        Args:
            texts: Lista de textos para embedding
            return_dense: Se retorna embeddings densos (1024d)
            return_sparse: Se retorna embeddings esparsos

        Returns:
            EmbeddingResult com dense e sparse embeddings

        Raises:
            httpx.HTTPError: Se falhar após retries
        """
        start_time = time.perf_counter()

        payload = {
            "texts": texts,
            "return_dense": return_dense,
            "return_sparse": return_sparse,
        }

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post(
                    f"{self.config.gpu_server_url}/embed",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()

                elapsed = (time.perf_counter() - start_time) * 1000
                logger.debug(
                    f"Embedding remoto: {len(texts)} textos em {elapsed:.2f}ms "
                    f"(server: {data.get('latency_ms', 0):.2f}ms)"
                )

                return EmbeddingResult(
                    dense_embeddings=data.get("dense_embeddings", []),
                    sparse_embeddings=data.get("sparse_embeddings", []),
                    latency_ms=elapsed,
                    count=data.get("count", len(texts)),
                )

            except httpx.HTTPError as e:
                last_error = e
                logger.warning(
                    f"Erro no embedding remoto (tentativa {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        raise last_error or Exception("Falha no embedding remoto")

    def encode_single(self, text: str) -> tuple[list[float], dict[int, float]]:
        """
        Gera embedding para um único texto.

        Returns:
            Tuple (dense_embedding, sparse_embedding)
        """
        result = self.encode([text])
        return result.dense_embeddings[0], result.sparse_embeddings[0]

    def encode_hybrid(self, texts: list[str]) -> dict:
        """
        Gera embeddings híbridos (compatível com BGEM3Embedder).

        Returns:
            Dict com 'dense_vecs' e 'lexical_weights'
        """
        result = self.encode(texts, return_dense=True, return_sparse=True)
        return {
            "dense": result.dense_embeddings,
            "sparse": result.sparse_embeddings,
        }

    def encode_hybrid_single(self, text: str) -> dict:
        """
        Gera embedding híbrido para um único texto.
        
        Compatível com HybridSearcher que espera:
            query_embedding["dense"] -> list[float]
            query_embedding["sparse"] -> dict[int, float]

        Returns:
            Dict com 'dense' e 'sparse'
        """
        result = self.encode([text], return_dense=True, return_sparse=True)
        return {
            "dense": result.dense_embeddings[0],
            "sparse": result.sparse_embeddings[0],
        }

    @property
    def embedding_dim(self) -> int:
        """Dimensão do embedding denso (BGE-M3 = 1024)."""
        return 1024

    def health_check(self) -> dict:
        """Verifica status do GPU Server."""
        try:
            response = self.client.get(
                f"{self.config.gpu_server_url}/health",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "status": "online" if data.get("status") == "healthy" else "degraded",
                "server_url": self.config.gpu_server_url,
                "embedder": data.get("embedder", {}),
                "latency_ms": data.get("uptime_seconds", 0),
            }

        except Exception as e:
            return {
                "status": "offline",
                "error": str(e),
                "server_url": self.config.gpu_server_url,
            }

    def close(self):
        """Fecha o cliente HTTP."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Singleton
_remote_embedder: Optional[RemoteEmbedder] = None


def get_remote_embedder(config: Optional[RemoteEmbedderConfig] = None) -> RemoteEmbedder:
    """Retorna instância singleton do RemoteEmbedder."""
    global _remote_embedder
    if _remote_embedder is None:
        _remote_embedder = RemoteEmbedder(config)
    return _remote_embedder
