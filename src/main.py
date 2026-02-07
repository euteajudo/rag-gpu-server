"""
RAG GPU Server - FastAPI para embeddings e reranking.

Versao async-safe: health checks usam asyncio.to_thread() para nao bloquear.

Endpoints:
    POST /embed         - Gera embeddings (dense + sparse) com batching automatico
    POST /rerank        - Reordena documentos por relevancia com batching
    GET  /health        - Health check (async-safe)
    GET  /healthz       - Liveness probe (Kubernetes)
    GET  /readyz        - Readiness probe (Kubernetes)
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
from pydantic import BaseModel, Field, field_validator

from .config import config
from .embedder import get_embedder
from .reranker import get_reranker
from .ingestion.pipeline import get_pipeline
from .batch_collector import (
    BatchCollector,
    EmbedBatchItem,
    EmbedBatchResult,
    RerankBatchItem,
    RerankBatchResult,
    create_embed_batch_processor,
    create_rerank_batch_processor,
)
from .auth import APIKeyAuthMiddleware, DISABLE_DOCS
from .ingestion.router import router as ingestion_router
from .inspection.router import router as inspection_router
from .middleware.rate_limit import RateLimitMiddleware, InMemoryRateLimiter

# GPU Metrics via pynvml
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# =============================================================================
# PR12: SCHEMA VALIDATION GUARDRAIL
# =============================================================================

def validate_milvus_schema_pr13(
    collection_name: str = "leis_v4",
    milvus_host: str = "localhost",
    milvus_port: int = 19530,
) -> dict:
    """
    Valida que a collection Milvus tem os campos PR13 (canonical offsets).

    PR13 requer:
    - canonical_start (INT64): offset início no canonical_text
    - canonical_end (INT64): offset fim no canonical_text
    - canonical_hash (VARCHAR 64): SHA256 anti-mismatch

    Args:
        collection_name: Nome da collection a validar
        milvus_host: Host do Milvus
        milvus_port: Porta do Milvus

    Returns:
        Dict com status da validação

    Raises:
        RuntimeError: Se collection existe mas falta campos PR13
    """
    try:
        from pymilvus import connections, Collection, utility
    except ImportError:
        # pymilvus não instalado - não é obrigatório para GPU Server
        return {
            "status": "skipped",
            "reason": "pymilvus not installed",
            "collection": collection_name,
        }

    PR13_REQUIRED_FIELDS = ["canonical_start", "canonical_end", "canonical_hash"]

    try:
        # Conecta ao Milvus
        connections.connect(
            alias="schema_check",
            host=milvus_host,
            port=milvus_port,
            timeout=10,
        )

        # Verifica se collection existe
        if not utility.has_collection(collection_name, using="schema_check"):
            connections.disconnect("schema_check")
            return {
                "status": "ok",
                "reason": "collection_not_exists",
                "collection": collection_name,
                "message": f"Collection '{collection_name}' não existe ainda. "
                           "Execute recreate_leis_v4_pr12.py para criar.",
            }

        # Collection existe - verifica campos
        col = Collection(collection_name, using="schema_check")
        field_names = [f.name for f in col.schema.fields]

        # Verifica campos PR13
        missing_fields = [f for f in PR13_REQUIRED_FIELDS if f not in field_names]

        connections.disconnect("schema_check")

        if missing_fields:
            # ERRO CRÍTICO: Collection existe mas falta campos PR13
            raise RuntimeError(
                f"\n"
                f"{'=' * 70}\n"
                f"ERRO PR12: Collection '{collection_name}' sem campos PR13!\n"
                f"{'=' * 70}\n"
                f"\n"
                f"Campos obrigatórios ausentes: {missing_fields}\n"
                f"\n"
                f"Para corrigir, execute:\n"
                f"  python scripts/recreate_leis_v4_pr12.py --dry-run  # Verificar\n"
                f"  python scripts/recreate_leis_v4_pr12.py --force    # Recriar (PERDE DADOS)\n"
                f"\n"
                f"Ou migre os dados manualmente antes de recriar.\n"
                f"{'=' * 70}\n"
            )

        return {
            "status": "ok",
            "reason": "schema_valid",
            "collection": collection_name,
            "pr13_fields": PR13_REQUIRED_FIELDS,
            "total_fields": len(field_names),
        }

    except RuntimeError:
        # Re-raise RuntimeError (é o erro intencional de schema inválido)
        raise
    except Exception as e:
        # Outros erros (conexão, timeout, etc) - não são fatais
        return {
            "status": "warning",
            "reason": "connection_error",
            "collection": collection_name,
            "error": str(e),
        }

def get_gpu_hardware_metrics() -> dict:
    """
    Obtém métricas de hardware da GPU via pynvml.

    Returns:
        Dict com utilização, memória, temperatura, etc.
    """
    if not PYNVML_AVAILABLE:
        return {"available": False, "error": "pynvml not installed"}

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            return {"available": False, "error": "No GPU found"}

        # Usa primeira GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Nome da GPU
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        # Utilização
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        # Memória
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Temperatura
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
        except pynvml.NVMLError:
            power = 0

        pynvml.nvmlShutdown()

        return {
            "available": True,
            "name": name,
            "utilization_percent": utilization.gpu,
            "memory_utilization_percent": utilization.memory,
            "memory_used_bytes": memory.used,
            "memory_total_bytes": memory.total,
            "memory_free_bytes": memory.free,
            "temperature_celsius": temperature,
            "power_draw_watts": round(power, 1),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# THREAD POOL & BATCH COLLECTORS
# =============================================================================

# Pool de threads para operacoes GPU blocking
# max_workers=2: permite 2 operacoes GPU simultaneas (embedding + rerank)
GPU_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# Semaforo para limitar requests GPU simultaneos (fallback, se nao usar batch)
GPU_SEMAPHORE = asyncio.Semaphore(4)  # Max 4 requests enfileirados

# Batch Collectors (inicializados no lifespan)
EMBED_COLLECTOR: BatchCollector | None = None
RERANK_COLLECTOR: BatchCollector | None = None

# Configuracao de batching
BATCH_CONFIG = {
    "embed": {
        "max_batch_size": 16,  # Maximo de requests agrupados
        "max_wait_ms": 50,      # Espera maxima por mais requests
    },
    "rerank": {
        "max_batch_size": 8,    # Rerank e mais pesado
        "max_wait_ms": 30,      # Menor espera
    },
}

# Rate Limiter (in-memory, sem Redis)
RATE_LIMITER = InMemoryRateLimiter(
    max_requests=config.gpu_rate_limit,
    window_seconds=60,
)


# =============================================================================
# MODELS
# =============================================================================


class EmbedRequest(BaseModel):
    """Request para embedding."""

    texts: list[str] = Field(..., min_length=1, max_length=100)
    return_dense: bool = True
    return_sparse: bool = True

    @field_validator("texts")
    @classmethod
    def validate_text_length(cls, texts: list[str]) -> list[str]:
        """Valida que cada texto nao excede o limite de caracteres."""
        max_len = config.max_text_length
        for i, text in enumerate(texts):
            if len(text) > max_len:
                raise ValueError(
                    f"Texto na posicao {i} excede o limite de {max_len} caracteres "
                    f"(tem {len(text)} caracteres)"
                )
        return texts


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

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, query: str) -> str:
        """Valida que a query nao excede o limite de caracteres."""
        max_len = config.max_text_length
        if len(query) > max_len:
            raise ValueError(
                f"Query excede o limite de {max_len} caracteres "
                f"(tem {len(query)} caracteres)"
            )
        return query

    @field_validator("documents")
    @classmethod
    def validate_documents_length(cls, documents: list[str]) -> list[str]:
        """Valida que cada documento nao excede o limite de caracteres."""
        max_len = config.max_text_length
        for i, doc in enumerate(documents):
            if len(doc) > max_len:
                raise ValueError(
                    f"Documento na posicao {i} excede o limite de {max_len} caracteres "
                    f"(tem {len(doc)} caracteres)"
                )
        return documents


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

# Tempo de inicio
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle do app - carrega modelos no startup, limpa no shutdown."""
    global EMBED_COLLECTOR, RERANK_COLLECTOR

    logger.info("=== RAG GPU Server iniciando ===")
    logger.info(f"Pipeline: VLM (Qwen3-VL + PyMuPDF)")
    logger.info(f"Embedding model: {config.embedding_model}")
    logger.info(f"Reranker model: {config.reranker_model}")
    logger.info(f"Device: {config.device}")
    logger.info(f"GPU ThreadPool workers: {GPU_EXECUTOR._max_workers}")
    logger.info(f"Batch config: {BATCH_CONFIG}")

    # Pre-carrega modelos
    logger.info("Pre-carregando embedder...")
    embedder = get_embedder()
    embedder._ensure_loaded()

    logger.info("Pre-carregando reranker...")
    reranker = get_reranker()
    reranker._ensure_loaded()

    logger.info("=== Modelos carregados! ===")

    # PR12: Valida schema Milvus (se disponível)
    logger.info("Validando schema Milvus PR13...")
    try:
        milvus_host = config.milvus_host if hasattr(config, "milvus_host") else "localhost"
        milvus_port = config.milvus_port if hasattr(config, "milvus_port") else 19530

        schema_result = validate_milvus_schema_pr13(
            collection_name="leis_v4",
            milvus_host=milvus_host,
            milvus_port=milvus_port,
        )

        if schema_result["status"] == "ok":
            logger.info(f"Schema PR13 válido: {schema_result.get('reason', 'ok')}")
        elif schema_result["status"] == "skipped":
            logger.info(f"Validação de schema pulada: {schema_result.get('reason', 'unknown')}")
        else:
            logger.warning(f"Validação de schema: {schema_result}")
    except RuntimeError as e:
        # Schema inválido - erro fatal
        logger.error(str(e))
        raise

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
    # Desabilita documentacao se DISABLE_DOCS=true
    docs_url=None if DISABLE_DOCS else "/docs",
    redoc_url=None if DISABLE_DOCS else "/redoc",
    openapi_url=None if DISABLE_DOCS else "/openapi.json",
)

# =============================================================================
# MIDDLEWARE (ordem importa: primeiro adicionado = ultimo executado)
# =============================================================================

# 1. CORS - Restrito apenas para dominios vectorgov.io e VPS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vectorgov.io",
        "https://www.vectorgov.io",
        "http://77.37.43.160",       # VPS Hostinger
        "http://localhost:3000",      # Dev local
        "http://127.0.0.1:3000",      # Dev local
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 2. Autenticacao por API Key
app.add_middleware(APIKeyAuthMiddleware)

# 3. Rate Limiting (in-memory, sem Redis)
app.add_middleware(RateLimitMiddleware, rate_limiter=RATE_LIMITER)

logger.info(
    f"Middleware de seguranca ativado: CORS restrito + API Key auth + "
    f"Rate Limit ({RATE_LIMITER.max_requests}/min)"
)

# Router de ingestao
app.include_router(ingestion_router)
app.include_router(inspection_router)

# NOTA: Endpoints /enrich/* ficam na VPS, não no RunPod
# Ver docs/VPS_ENRICH_INTEGRATION.md para detalhes


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Gera embeddings para lista de textos.

    Retorna:
    - dense_embeddings: Vetores 1024d (semanticos)
    - sparse_embeddings: Dicts token_id -> weight (keywords)

    Nota: Usa BatchCollector para agrupar requests e processar em batch.
    """
    try:
        if EMBED_COLLECTOR is None:
            raise HTTPException(status_code=503, detail="Batch collector not initialized")

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
    Reordena documentos por relevancia a query.

    Retorna:
    - scores: Score de relevancia para cada documento (0-1)
    - rankings: Indices dos documentos ordenados por relevancia
    """
    try:
        if RERANK_COLLECTOR is None:
            raise HTTPException(status_code=503, detail="Batch collector not initialized")

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
    """
    Health check completo com status dos modelos.

    CORRIGIDO: Usa asyncio.to_thread() para nao bloquear event loop.
    """

    def _do_health_checks():
        """Executa health checks (sync - roda em thread)."""
        embedder = get_embedder()
        reranker = get_reranker()

        embedder_health = embedder.health_check()
        reranker_health = reranker.health_check()

        all_online = (
            embedder_health.get("status") == "online"
            and reranker_health.get("status") == "online"
        )

        return {
            "status": "healthy" if all_online else "degraded",
            "embedder": embedder_health,
            "reranker": reranker_health,
        }

    # Executa em thread para nao bloquear
    result = await asyncio.to_thread(_do_health_checks)

    return HealthResponse(
        status=result["status"],
        embedder=result["embedder"],
        reranker=result["reranker"],
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
        embedder = get_embedder()
        reranker = get_reranker()

        if embedder._model is None or reranker._model is None:
            raise HTTPException(status_code=503, detail="Models not ready")

        return {"status": "ready"}

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/stats")
async def stats():
    """Estatisticas de concorrencia, batching, rate limiting e uso."""
    # GPU hardware metrics (non-blocking)
    gpu_metrics = await asyncio.to_thread(get_gpu_hardware_metrics)

    return {
        "uptime_seconds": round(time.time() - _start_time, 2),
        "gpu": gpu_metrics,  # Métricas de hardware da GPU
        "gpu_executor": {
            "max_workers": GPU_EXECUTOR._max_workers,
            "active_threads": len(GPU_EXECUTOR._threads),
        },
        "batch_collectors": {
            "embed": EMBED_COLLECTOR.stats() if EMBED_COLLECTOR else None,
            "rerank": RERANK_COLLECTOR.stats() if RERANK_COLLECTOR else None,
        },
        "rate_limiter": RATE_LIMITER.get_stats(),
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
