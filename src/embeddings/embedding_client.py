"""
Cliente de embeddings para BGE-M3.

PR3 v2 - Hard Reset RAG Architecture

Gera embeddings dense (1024d) e sparse a partir do retrieval_text.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuração do cliente de embeddings."""

    model_name: str = "BAAI/bge-m3"
    batch_size: int = 32
    max_length: int = 8192
    use_fp16: bool = True
    device: str = "cuda"
    return_sparse: bool = True
    return_dense: bool = True


@dataclass
class EmbeddingResult:
    """Resultado de embedding para um texto."""

    dense_vector: list[float] = field(default_factory=list)
    sparse_vector: dict[int, float] = field(default_factory=dict)
    text_length: int = 0
    truncated: bool = False


class EmbeddingClient:
    """
    Cliente para gerar embeddings usando BGE-M3.

    Suporta:
    - Dense embeddings (1024 dimensões)
    - Sparse embeddings (learned sparse)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Inicializa o cliente de embeddings.

        Args:
            config: Configuração do cliente
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-load do modelo BGE-M3."""
        if self._initialized:
            return

        try:
            from FlagEmbedding import BGEM3FlagModel

            logger.info(f"Carregando modelo {self.config.model_name}...")

            self._model = BGEM3FlagModel(
                self.config.model_name,
                use_fp16=self.config.use_fp16,
                device=self.config.device,
            )

            self._initialized = True
            logger.info("Modelo BGE-M3 carregado com sucesso")

        except ImportError as e:
            raise ImportError(
                "FlagEmbedding não instalado. Instale com: pip install FlagEmbedding"
            ) from e

    def embed(self, text: str) -> EmbeddingResult:
        """
        Gera embeddings para um texto.

        Args:
            text: Texto para embedding

        Returns:
            EmbeddingResult com vetores dense e sparse
        """
        self._ensure_initialized()

        text_length = len(text)
        truncated = text_length > self.config.max_length

        # Gera embeddings
        output = self._model.encode(
            [text],
            batch_size=1,
            max_length=self.config.max_length,
            return_dense=self.config.return_dense,
            return_sparse=self.config.return_sparse,
        )

        # Extrai vetores
        dense_vector = []
        sparse_vector = {}

        if self.config.return_dense and "dense_vecs" in output:
            dense_vector = output["dense_vecs"][0].tolist()

        if self.config.return_sparse and "lexical_weights" in output:
            sparse_weights = output["lexical_weights"][0]
            # Converte para dict[int, float]
            for token_id, weight in sparse_weights.items():
                sparse_vector[int(token_id)] = float(weight)

        return EmbeddingResult(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            text_length=text_length,
            truncated=truncated,
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Gera embeddings para múltiplos textos.

        Args:
            texts: Lista de textos

        Returns:
            Lista de EmbeddingResult
        """
        self._ensure_initialized()

        if not texts:
            return []

        # Gera embeddings em batch
        output = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=self.config.return_dense,
            return_sparse=self.config.return_sparse,
        )

        results = []
        for i, text in enumerate(texts):
            text_length = len(text)
            truncated = text_length > self.config.max_length

            dense_vector = []
            sparse_vector = {}

            if self.config.return_dense and "dense_vecs" in output:
                dense_vector = output["dense_vecs"][i].tolist()

            if self.config.return_sparse and "lexical_weights" in output:
                sparse_weights = output["lexical_weights"][i]
                for token_id, weight in sparse_weights.items():
                    sparse_vector[int(token_id)] = float(weight)

            results.append(
                EmbeddingResult(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    text_length=text_length,
                    truncated=truncated,
                )
            )

        return results

    @property
    def dense_dim(self) -> int:
        """Dimensão do vetor dense (1024 para BGE-M3)."""
        return 1024

    def is_available(self) -> bool:
        """Verifica se o modelo está disponível."""
        try:
            self._ensure_initialized()
            return True
        except Exception as e:
            logger.warning(f"Modelo não disponível: {e}")
            return False
