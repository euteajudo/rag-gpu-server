"""
BGE-M3 Embedder - Gera embeddings dense + sparse.

Uso:
    from src.embedder import get_embedder

    embedder = get_embedder()
    result = embedder.encode(["texto 1", "texto 2"])
    # result.dense_embeddings: list[list[float]]
    # result.sparse_embeddings: list[dict[int, float]]
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch
from FlagEmbedding import BGEM3FlagModel

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Resultado do embedding."""

    dense_embeddings: list[list[float]]
    sparse_embeddings: list[dict[int, float]]
    latency_ms: float


class BGEM3Embedder:
    """
    Wrapper para BGE-M3.

    Gera embeddings:
    - Dense: 1024 dimensões (semântico)
    - Sparse: Learned sparse (keywords)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self._model: Optional[BGEM3FlagModel] = None

    def _ensure_loaded(self):
        """Carrega modelo se necessário."""
        if self._model is None:
            logger.info(f"Carregando BGE-M3: {self.model_name}")
            start = time.perf_counter()
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            elapsed = time.perf_counter() - start
            logger.info(f"BGE-M3 carregado em {elapsed:.2f}s")

    def encode(
        self,
        texts: list[str],
        return_dense: bool = True,
        return_sparse: bool = True,
    ) -> EmbeddingResult:
        """
        Gera embeddings para lista de textos.

        Args:
            texts: Lista de textos
            return_dense: Se retorna embeddings densos
            return_sparse: Se retorna embeddings esparsos

        Returns:
            EmbeddingResult com dense e sparse embeddings
        """
        self._ensure_loaded()

        start = time.perf_counter()

        result = self._model.encode(
            texts,
            return_dense=return_dense,
            return_sparse=return_sparse,
        )

        elapsed = (time.perf_counter() - start) * 1000

        # Converte dense para lista
        dense_embeddings = []
        if return_dense and "dense_vecs" in result:
            for vec in result["dense_vecs"]:
                if isinstance(vec, torch.Tensor):
                    dense_embeddings.append(vec.cpu().tolist())
                else:
                    dense_embeddings.append(vec.tolist())

        # Converte sparse para dict[int, float]
        sparse_embeddings = []
        if return_sparse and "lexical_weights" in result:
            for weights in result["lexical_weights"]:
                sparse_dict = {int(k): float(v) for k, v in weights.items()}
                sparse_embeddings.append(sparse_dict)

        return EmbeddingResult(
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            latency_ms=elapsed,
        )

    @property
    def embedding_dim(self) -> int:
        """Dimensão do embedding denso."""
        return 1024

    def health_check(self) -> dict:
        """Verifica status do modelo."""
        try:
            self._ensure_loaded()
            # Teste rápido
            result = self.encode(["test"], return_dense=True, return_sparse=False)
            return {
                "status": "online",
                "model": self.model_name,
                "embedding_dim": self.embedding_dim,
                "device": self.device,
                "latency_ms": round(result.latency_ms, 2),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


# Singleton
_embedder: Optional[BGEM3Embedder] = None


def get_embedder() -> BGEM3Embedder:
    """Retorna instância singleton do embedder."""
    global _embedder
    if _embedder is None:
        _embedder = BGEM3Embedder(
            model_name=config.embedding_model,
            use_fp16=config.use_fp16,
            device=config.device,
        )
    return _embedder
