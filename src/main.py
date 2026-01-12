"""
RAG GPU Server - FastAPI para embeddings e reranking.

Endpoints:
    POST /embed         - Gera embeddings (dense + sparse) com batching automático
    POST /rerank        - Reordena documentos por relevância com batching
    GET  /health        - Health check
    GET  /healthz       - Liveness probe (Kubernetes)
    GET  /readyz        - Readiness probe (Kubernetes)

Arquitetura:
    - 1 worker uvicorn (modelos carregados 1x na GPU)
    - BatchCollector para agrupar requests e processar em batch
    - Micro-batching: espera até 50ms ou 16 items antes de processar
    - Throughput 3-5x maior para requests concorrentes

Uso:
    uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import config
from .embedder import get_embedder
from .reranker import get_reranker
from .batch_collector import (
    BatchCollector,
    EmbedBatchItem,
    EmbedBatchResult,
    RerankBatchItem,
    RerankBatchResult,
    create_embed_batch_processor,
    create_rerank_batch_processor,
)
from .ingestion import ingestion_router

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# THREAD POOL & BATCH COLLECTORS
# =============================================================================

# Pool de threads para operações GPU blocking
# max_workers=2: permite 2 operações GPU simultâneas (embedding + rerank)
GPU_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# Semáforo para limitar requests GPU simultâneos (fallback, se não usar batch)
GPU_SEMAPHORE = asyncio.Semaphore(4)  # Max 4 requests enfileirados

# Batch Collectors (inicializados no lifespan)
EMBED_COLLECTOR: BatchCollector | None = None
RERANK_COLLECTOR: BatchCollector | None = None

# Configuração de batching
BATCH_CONFIG = {
    "embed": {
        "max_batch_size": 16,  # Máximo de requests agrupados
        "max_wait_ms": 50,      # Espera máxima por mais requests
    },
    "rerank": {
        "max_batch_size": 8,    # Rerank é mais pesado
        "max_wait_ms": 30,      # Menor espera
    },
}


# =============================================================================
# MODELS
# =============================================================================


class EmbedRequest(BaseModel):
    """Request para embedding."""

    texts: list[str] = Field(..., min_length=1, max_length=100)
    return_dense: bool = True
    return_sparse: bool = True


class EmbedResponse(BaseModel):
    """Response com embeddings."""

    dense_embeddings: Optional[list[list[float]]] = None
    sparse_embeddings: Optional[list[dict[int, float]]] = None
    latency_ms: float
    count: int


class RerankRequest(BaseModel):
    """Request para reranking."""

    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1, max_length=100)
    top_k: Optional[int] = Field(None, ge=1, le=100)


class RerankResponse(BaseModel):
    """Response com scores de reranking."""

    scores: list[float]
    rankings: list[int]
    latency_ms: float


class HealthResponse(BaseModel):
    """Response do health check."""

    status: str
    embedder: dict
    reranker: dict
    uptime_seconds: float


# =============================================================================
# APP
# =============================================================================

# Tempo de início
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle do app - carrega modelos no startup, limpa no shutdown."""
    global EMBED_COLLECTOR, RERANK_COLLECTOR

    logger.info("=== RAG GPU Server iniciando ===")
    logger.info(f"Embedding model: {config.embedding_model}")
    logger.info(f"Reranker model: {config.reranker_model}")
    logger.info(f"Device: {config.device}")
    logger.info(f"GPU ThreadPool workers: {GPU_EXECUTOR._max_workers}")
    logger.info(f"Batch config: {BATCH_CONFIG}")

    # Pré-carrega modelos
    logger.info("Pré-carregando embedder...")
    embedder = get_embedder()
    embedder._ensure_loaded()

    logger.info("Pré-carregando reranker...")
    reranker = get_reranker()
    reranker._ensure_loaded()

    logger.info("=== Modelos carregados! ===")

    # Inicializa Batch Collectors
    logger.info("Iniciando Batch Collectors...")

    EMBED_COLLECTOR = BatchCollector(
        processor_fn=create_embed_batch_processor(embedder),
        max_batch_size=BATCH_CONFIG["embed"]["max_batch_size"],
        max_wait_ms=BATCH_CONFIG["embed"]["max_wait_ms"],
        name="embed",
    )
    await EMBED_COLLECTOR.start()

    RERANK_COLLECTOR = BatchCollector(
        processor_fn=create_rerank_batch_processor(reranker),
        max_batch_size=BATCH_CONFIG["rerank"]["max_batch_size"],
        max_wait_ms=BATCH_CONFIG["rerank"]["max_wait_ms"],
        name="rerank",
    )
    await RERANK_COLLECTOR.start()

    logger.info("=== Batch Collectors ativos! ===")

    yield

    # Shutdown: limpa recursos
    logger.info("=== GPU Server encerrando ===")

    logger.info("Parando Batch Collectors...")
    if EMBED_COLLECTOR:
        await EMBED_COLLECTOR.stop()
    if RERANK_COLLECTOR:
        await RERANK_COLLECTOR.stop()

    logger.info("Encerrando ThreadPoolExecutor...")
    GPU_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    logger.info("=== Shutdown completo ===")


app = FastAPI(
    title="RAG GPU Server",
    description="Servidor GPU para embeddings (BGE-M3) e reranking (BGE-Reranker)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ingestion_router)


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Gera embeddings para lista de textos.

    Retorna:
    - dense_embeddings: Vetores 1024d (semânticos)
    - sparse_embeddings: Dicts token_id -> weight (keywords)

    Nota: Usa BatchCollector para agrupar requests e processar em batch.
    Múltiplos requests simultâneos são agrupados para melhor throughput.
    """
    try:
        if EMBED_COLLECTOR is None:
            raise HTTPException(status_code=503, detail="Batch collector not initialized")

        # Submete para o batch collector
        batch_item = EmbedBatchItem(
            texts=request.texts,
            return_dense=request.return_dense,
            return_sparse=request.return_sparse,
        )

        result: EmbedBatchResult = await EMBED_COLLECTOR.submit(batch_item)

        return EmbedResponse(
            dense_embeddings=result.dense_embeddings if request.return_dense else None,
            sparse_embeddings=result.sparse_embeddings if request.return_sparse else None,
            latency_ms=round(result.latency_ms, 2),
            count=len(request.texts),
        )

    except Exception as e:
        logger.error(f"Erro no embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Reordena documentos por relevância à query.

    Retorna:
    - scores: Score de relevância para cada documento (0-1)
    - rankings: Índices dos documentos ordenados por relevância

    Nota: Usa BatchCollector para agrupar requests de diferentes clientes.
    """
    try:
        if RERANK_COLLECTOR is None:
            raise HTTPException(status_code=503, detail="Batch collector not initialized")

        # Submete para o batch collector
        batch_item = RerankBatchItem(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k,
        )

        result: RerankBatchResult = await RERANK_COLLECTOR.submit(batch_item)

        return RerankResponse(
            scores=result.scores,
            rankings=result.rankings,
            latency_ms=round(result.latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Erro no reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check completo com status dos modelos."""
    embedder = get_embedder()
    reranker = get_reranker()

    embedder_health = embedder.health_check()
    reranker_health = reranker.health_check()

    # Status geral
    all_online = (
        embedder_health.get("status") == "online"
        and reranker_health.get("status") == "online"
    )

    return HealthResponse(
        status="healthy" if all_online else "degraded",
        embedder=embedder_health,
        reranker=reranker_health,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/healthz")
async def healthz():
    """Liveness probe (Kubernetes)."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness probe (Kubernetes)."""
    try:
        # Verifica se modelos estão carregados
        embedder = get_embedder()
        reranker = get_reranker()

        if embedder._model is None or reranker._model is None:
            raise HTTPException(status_code=503, detail="Models not ready")

        return {"status": "ready"}

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/stats")
async def stats():
    """Estatísticas de concorrência, batching e uso."""
    return {
        "uptime_seconds": round(time.time() - _start_time, 2),
        "gpu_executor": {
            "max_workers": GPU_EXECUTOR._max_workers,
            "active_threads": len(GPU_EXECUTOR._threads),
        },
        "batch_collectors": {
            "embed": EMBED_COLLECTOR.stats() if EMBED_COLLECTOR else None,
            "rerank": RERANK_COLLECTOR.stats() if RERANK_COLLECTOR else None,
        },
    }


@app.get("/")
async def root():
    """Redirect para docs."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        reload=False,
    )
