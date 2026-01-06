"""
Batch Collector - Agrupa requests para processamento em batch na GPU.

Arquitetura:
    Request 1 ──┐
    Request 2 ──┼──► BatchCollector ──► GPU (1 chamada) ──► Distribui resultados
    Request 3 ──┘        │
                   (espera max 50ms)

Benefícios:
    - GPU processa N textos em 1 chamada (vs N chamadas separadas)
    - Throughput 3-5x maior para requests concorrentes
    - Latência similar para requests individuais

Uso:
    collector = BatchCollector(processor_fn, max_batch_size=16, max_wait_ms=50)
    await collector.start()
    result = await collector.submit(item)  # Espera pelo batch
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Tipo do item
R = TypeVar("R")  # Tipo do resultado


@dataclass
class BatchItem(Generic[T]):
    """Item individual na fila do batch."""

    id: str
    data: T
    future: asyncio.Future = field(default_factory=asyncio.Future)
    timestamp: float = field(default_factory=time.time)


class BatchCollector(Generic[T, R]):
    """
    Coletor de requests para processamento em batch.

    Agrupa múltiplos requests em um batch e processa com uma única
    chamada GPU, distribuindo os resultados de volta.

    Args:
        processor_fn: Função que processa o batch (sync ou async)
        max_batch_size: Tamanho máximo do batch
        max_wait_ms: Tempo máximo de espera por mais items (em ms)
        name: Nome do collector (para logs)
    """

    def __init__(
        self,
        processor_fn: Callable[[list[T]], list[R]],
        max_batch_size: int = 16,
        max_wait_ms: float = 50,
        name: str = "batch",
    ):
        self.processor_fn = processor_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.name = name

        self._queue: asyncio.Queue[BatchItem[T]] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        # Métricas
        self._batches_processed = 0
        self._items_processed = 0
        self._total_wait_ms = 0

    async def start(self):
        """Inicia o loop de processamento de batches."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info(
            f"[{self.name}] BatchCollector iniciado "
            f"(max_batch={self.max_batch_size}, max_wait={self.max_wait_ms}ms)"
        )

    async def stop(self):
        """Para o loop de processamento."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"[{self.name}] BatchCollector parado")

    async def submit(self, data: T) -> R:
        """
        Submete um item para processamento em batch.

        Args:
            data: Dados a serem processados

        Returns:
            Resultado do processamento

        Raises:
            Exception: Se o processamento falhar
        """
        item = BatchItem(
            id=str(uuid.uuid4())[:8],
            data=data,
            future=asyncio.get_event_loop().create_future(),
        )

        await self._queue.put(item)
        logger.debug(f"[{self.name}] Item {item.id} adicionado à fila")

        # Espera pelo resultado
        return await item.future

    async def _process_loop(self):
        """Loop principal que coleta e processa batches."""
        while self._running:
            try:
                batch = await self._collect_batch()

                if not batch:
                    continue

                await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.name}] Erro no loop: {e}")
                await asyncio.sleep(0.1)

    async def _collect_batch(self) -> list[BatchItem[T]]:
        """
        Coleta items para o batch.

        Espera até:
        - Atingir max_batch_size, ou
        - Passar max_wait_ms após primeiro item
        """
        batch: list[BatchItem[T]] = []
        deadline: float | None = None

        while len(batch) < self.max_batch_size:
            # Calcula timeout
            if deadline is None:
                timeout = None  # Espera indefinidamente pelo primeiro item
            else:
                timeout = max(0, deadline - time.time())

            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                batch.append(item)

                # Define deadline após primeiro item
                if deadline is None:
                    deadline = time.time() + (self.max_wait_ms / 1000)

            except asyncio.TimeoutError:
                # Timeout atingido, processa o que temos
                break

        return batch

    async def _process_batch(self, batch: list[BatchItem[T]]):
        """Processa um batch de items."""
        if not batch:
            return

        batch_start = time.time()
        batch_ids = [item.id for item in batch]
        logger.info(f"[{self.name}] Processando batch de {len(batch)} items: {batch_ids}")

        try:
            # Extrai dados dos items
            data_list = [item.data for item in batch]

            # Processa (pode ser sync ou async)
            if asyncio.iscoroutinefunction(self.processor_fn):
                results = await self.processor_fn(data_list)
            else:
                # Roda função sync em thread
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, self.processor_fn, data_list)

            # Distribui resultados
            if len(results) != len(batch):
                raise ValueError(
                    f"Processor retornou {len(results)} resultados para {len(batch)} items"
                )

            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)

            # Métricas
            elapsed = (time.time() - batch_start) * 1000
            self._batches_processed += 1
            self._items_processed += len(batch)
            self._total_wait_ms += elapsed

            logger.info(
                f"[{self.name}] Batch processado em {elapsed:.1f}ms "
                f"({elapsed/len(batch):.1f}ms/item)"
            )

        except Exception as e:
            logger.error(f"[{self.name}] Erro no batch: {e}")
            # Propaga erro para todos os items
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)

    def stats(self) -> dict:
        """Retorna estatísticas do collector."""
        avg_batch_size = (
            self._items_processed / self._batches_processed
            if self._batches_processed > 0
            else 0
        )
        avg_latency = (
            self._total_wait_ms / self._batches_processed
            if self._batches_processed > 0
            else 0
        )

        return {
            "name": self.name,
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "batches_processed": self._batches_processed,
            "items_processed": self._items_processed,
            "avg_batch_size": round(avg_batch_size, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "queue_size": self._queue.qsize(),
        }


# =============================================================================
# SPECIALIZED COLLECTORS
# =============================================================================


@dataclass
class EmbedBatchItem:
    """Item para batch de embedding."""

    texts: list[str]
    return_dense: bool = True
    return_sparse: bool = True


@dataclass
class EmbedBatchResult:
    """Resultado de embedding para um item do batch."""

    dense_embeddings: list[list[float]] | None
    sparse_embeddings: list[dict[int, float]] | None
    latency_ms: float


@dataclass
class RerankBatchItem:
    """Item para batch de reranking."""

    query: str
    documents: list[str]
    top_k: int | None = None


@dataclass
class RerankBatchResult:
    """Resultado de reranking para um item do batch."""

    scores: list[float]
    rankings: list[int]
    latency_ms: float


def create_embed_batch_processor(embedder):
    """
    Cria processador de batch para embeddings.

    O batch agrupa múltiplos requests, concatena todos os textos,
    processa em uma chamada, e divide os resultados.
    """

    def process_batch(items: list[EmbedBatchItem]) -> list[EmbedBatchResult]:
        import time

        start = time.perf_counter()

        # Concatena todos os textos com índices de separação
        all_texts = []
        separators = [0]  # Índices onde cada item começa

        for item in items:
            all_texts.extend(item.texts)
            separators.append(len(all_texts))

        # Determina flags (usa OR - se qualquer um pedir, retorna)
        return_dense = any(item.return_dense for item in items)
        return_sparse = any(item.return_sparse for item in items)

        # Uma única chamada para todos os textos
        result = embedder.encode(
            texts=all_texts,
            return_dense=return_dense,
            return_sparse=return_sparse,
        )

        # Divide resultados de volta
        batch_results = []
        for i, item in enumerate(items):
            start_idx = separators[i]
            end_idx = separators[i + 1]

            dense = None
            sparse = None

            if item.return_dense and result.dense_embeddings:
                dense = result.dense_embeddings[start_idx:end_idx]
            if item.return_sparse and result.sparse_embeddings:
                sparse = result.sparse_embeddings[start_idx:end_idx]

            batch_results.append(
                EmbedBatchResult(
                    dense_embeddings=dense,
                    sparse_embeddings=sparse,
                    latency_ms=result.latency_ms / len(items),  # Divide latência
                )
            )

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"[embed] Batch de {len(items)} items processado em {elapsed:.1f}ms")

        return batch_results

    return process_batch


def create_rerank_batch_processor(reranker):
    """
    Cria processador de batch para reranking.

    O batch agrupa múltiplos requests, processa cada par query-docs,
    e retorna resultados separados.

    NOTA: Reranking é mais difícil de batchear porque cada query
    tem documentos diferentes. Aqui processamos sequencialmente
    mas em uma única chamada de thread.
    """

    def process_batch(items: list[RerankBatchItem]) -> list[RerankBatchResult]:
        import time

        start = time.perf_counter()
        results = []

        for item in items:
            result = reranker.rerank(
                query=item.query,
                documents=item.documents,
                top_k=item.top_k,
            )
            results.append(
                RerankBatchResult(
                    scores=result.scores,
                    rankings=result.rankings,
                    latency_ms=result.latency_ms,
                )
            )

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"[rerank] Batch de {len(items)} items processado em {elapsed:.1f}ms")

        return results

    return process_batch
