"""
Remote Reranker - Cliente para GPU Server (BGE-Reranker).

Chama o endpoint /rerank do GPU Server para reordenar documentos
por relevância usando cross-encoder.

Uso:
    from remote import RemoteReranker

    reranker = RemoteReranker()
    result = reranker.rerank(
        query="O que é ETP?",
        documents=[{"text": "doc1"}, {"text": "doc2"}]
    )
"""

import os
import logging
import time
from dataclasses import dataclass
from typing import Optional, Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RemoteRerankerConfig:
    """Configuração do cliente de reranking remoto."""

    gpu_server_url: str = "http://localhost:8000"
    gpu_api_key: str = ""  # API Key para autenticação no GPU Server
    timeout: float = 120.0  # Reranking pode demorar mais
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "RemoteRerankerConfig":
        """Carrega configuração de variáveis de ambiente."""
        return cls(
            gpu_server_url=os.getenv("GPU_SERVER_URL", "http://localhost:8000"),
            gpu_api_key=os.getenv("GPU_API_KEY", ""),
            timeout=float(os.getenv("GPU_SERVER_TIMEOUT", "120")),
            max_retries=int(os.getenv("GPU_SERVER_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("GPU_SERVER_RETRY_DELAY", "1.0")),
        )


@dataclass
class RerankResult:
    """Resultado do reranking remoto."""

    scores: list[float]
    rankings: list[int]
    latency_ms: float


class RemoteReranker:
    """
    Cliente para GPU Server - endpoint /rerank.

    Reordena documentos usando BGE-Reranker no servidor GPU remoto.
    Compatível com a interface do BGEReranker local.
    """

    def __init__(self, config: Optional[RemoteRerankerConfig] = None):
        self.config = config or RemoteRerankerConfig.from_env()
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

    def _call_gpu_server(
        self,
        query: str,
        document_texts: list[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """Chama o GPU server para reranking."""
        if not document_texts:
            return RerankResult(scores=[], rankings=[], latency_ms=0)

        start_time = time.perf_counter()

        payload = {
            "query": query,
            "documents": document_texts,
        }
        if top_k:
            payload["top_k"] = top_k

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post(
                    f"{self.config.gpu_server_url}/rerank",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()

                elapsed = (time.perf_counter() - start_time) * 1000
                logger.debug(
                    f"Reranking remoto: {len(document_texts)} docs em {elapsed:.2f}ms "
                    f"(server: {data.get('latency_ms', 0):.2f}ms)"
                )

                return RerankResult(
                    scores=data.get("scores", []),
                    rankings=data.get("rankings", []),
                    latency_ms=elapsed,
                )

            except httpx.HTTPError as e:
                last_error = e
                logger.warning(
                    f"Erro no reranking remoto (tentativa {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        raise last_error or Exception("Falha no reranking remoto")

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        text_key: str = "text",
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Reordena documentos por relevância à query via GPU Server.
        
        Compatível com a interface do BGEReranker local.

        Args:
            query: Query de busca
            documents: Lista de dicts com documentos
            text_key: Chave do texto no dict
            top_k: Retorna apenas top K (None = todos)
            return_scores: Adiciona score ao resultado

        Returns:
            Lista de documentos reordenados (mesmo formato de entrada)
        """
        if not documents:
            return []

        # Extrai textos dos dicts
        texts = [doc.get(text_key, "") for doc in documents]
        
        # Chama GPU server
        result = self._call_gpu_server(query, texts, top_k=top_k)
        
        # Reordena documentos baseado nos rankings
        reordered = []
        for rank_idx in result.rankings:
            if rank_idx < len(documents):
                doc = documents[rank_idx].copy()
                if return_scores and rank_idx < len(result.scores):
                    doc["rerank_score"] = result.scores[rank_idx]
                reordered.append(doc)
        
        # Se não temos rankings, retorna na ordem original com scores
        if not reordered:
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                if return_scores and i < len(result.scores):
                    doc_copy["rerank_score"] = result.scores[i]
                reordered.append(doc_copy)
        
        return reordered

    def rerank_simple(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """
        Reordena lista simples de strings via GPU Server.

        Args:
            query: Query de busca
            documents: Lista de strings
            top_k: Retorna apenas top K (None = todos)

        Returns:
            RerankResult com scores e rankings
        """
        return self._call_gpu_server(query, documents, top_k)

    def compute_score(
        self,
        pairs: list[list[str]],
        normalize: bool = True,
    ) -> list[float]:
        """
        Calcula scores para pares query-documento.
        Compatível com a interface do FlagReranker.

        Args:
            pairs: Lista de [query, document]
            normalize: Se normaliza scores (0-1)

        Returns:
            Lista de scores
        """
        if not pairs:
            return []

        # Agrupa por query única para otimizar chamadas
        query = pairs[0][0]
        documents = [p[1] for p in pairs]

        result = self._call_gpu_server(query, documents)
        return result.scores

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
                "reranker": data.get("reranker", {}),
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
_remote_reranker: Optional[RemoteReranker] = None


def get_remote_reranker(config: Optional[RemoteRerankerConfig] = None) -> RemoteReranker:
    """Retorna instância singleton do RemoteReranker."""
    global _remote_reranker
    if _remote_reranker is None:
        _remote_reranker = RemoteReranker(config)
    return _remote_reranker
