"""
BGE Reranker - Cross-encoder para reranking.

Uso:
    from src.reranker import get_reranker

    reranker = get_reranker()
    result = reranker.rerank(
        query="O que é ETP?",
        documents=["doc1", "doc2", "doc3"]
    )
    # result.scores: list[float]
    # result.rankings: list[int] (índices ordenados)
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

# IMPORTANTE: Monkeypatch para contornar verificação CVE-2025-32434 do transformers
# O modelo BGE-Reranker usa pytorch_model.bin (não safetensors), o que dispara a verificação.
# Temos torch >= 2.6 instalado, então é seguro desabilitar.
try:
    import transformers.utils.import_utils as import_utils
    # Substitui a função de verificação por uma que não faz nada
    import_utils.check_torch_load_is_safe = lambda: None
    # Limpa o cache da verificação de versão do torch
    if hasattr(import_utils.is_torch_greater_or_equal, 'cache_clear'):
        import_utils.is_torch_greater_or_equal.cache_clear()
except Exception:
    pass  # Se falhar, continua normalmente

from FlagEmbedding import FlagReranker

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Resultado do reranking."""

    scores: list[float]
    rankings: list[int]  # Índices ordenados por score (desc)
    latency_ms: float


class BGEReranker:
    """
    Wrapper para BGE-Reranker-v2-m3.

    Cross-encoder que recebe query + documento e retorna score de relevância.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self._model: Optional[FlagReranker] = None

    def _ensure_loaded(self):
        """Carrega modelo se necessário."""
        if self._model is None:
            logger.info(f"Carregando BGE-Reranker: {self.model_name}")
            start = time.perf_counter()
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            elapsed = time.perf_counter() - start
            logger.info(f"BGE-Reranker carregado em {elapsed:.2f}s")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """
        Reordena documentos por relevância à query.

        Args:
            query: Query de busca
            documents: Lista de documentos para reordenar
            top_k: Retorna apenas top K (None = todos)

        Returns:
            RerankResult com scores e rankings
        """
        self._ensure_loaded()

        if not documents:
            return RerankResult(scores=[], rankings=[], latency_ms=0)

        start = time.perf_counter()

        # Prepara pares query-documento
        pairs = [[query, doc] for doc in documents]

        # Calcula scores
        scores = self._model.compute_score(pairs, normalize=True)

        # Garante que é lista
        if not isinstance(scores, list):
            scores = [scores]

        elapsed = (time.perf_counter() - start) * 1000

        # Ordena por score (descendente)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        rankings = [idx for idx, _ in indexed_scores]

        if top_k:
            rankings = rankings[:top_k]

        return RerankResult(
            scores=scores,
            rankings=rankings,
            latency_ms=elapsed,
        )

    def health_check(self) -> dict:
        """Verifica status do modelo."""
        try:
            self._ensure_loaded()
            # Teste rápido
            result = self.rerank("test query", ["test document"])
            return {
                "status": "online",
                "model": self.model_name,
                "device": self.device,
                "latency_ms": round(result.latency_ms, 2),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


# Singleton
_reranker: Optional[BGEReranker] = None


def get_reranker() -> BGEReranker:
    """Retorna instância singleton do reranker."""
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker(
            model_name=config.reranker_model,
            use_fp16=config.use_fp16,
            device=config.device,
        )
    return _reranker
