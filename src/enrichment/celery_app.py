"""
Celery App - Arquitetura 2 Filas

FILAS:
- llm_enrich: Tasks que chamam o Qwen (vLLM) - 6 workers
- embed_store: Tasks de embedding (BGE-M3) + Milvus - 2 workers
- celery: Fila default (batch tasks)

COMO SUBIR OS WORKERS:

# Pool LLM (6 workers)
celery -A src.enrichment.celery_app worker -Q llm_enrich -c 6 -n llm@%h

# Pool Embed/Store (2 workers)
celery -A src.enrichment.celery_app worker -Q embed_store -c 2 -n embed@%h
"""

from celery import Celery
import os

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

app = Celery(
    "enrichment",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    include=["src.enrichment.tasks"],
)

# =============================================================================
# ROTEAMENTO DE TASKS - Evita erros de fila
# =============================================================================

app.conf.task_routes = {
    "src.enrichment.tasks.enrich_chunk_llm": {"queue": "llm_enrich"},
    "src.enrichment.tasks.embed_and_store": {"queue": "embed_store"},
    # Tasks batch/legado vão para default 'celery'
    "src.enrichment.tasks.enrich_batch_task": {"queue": "celery"},
    "src.enrichment.tasks.enrich_chunk_task": {"queue": "celery"},
}

# =============================================================================
# CONFIGURACOES GERAIS
# =============================================================================

app.conf.update(
    # Timeouts
    task_time_limit=300,        # 5 min hard limit
    task_soft_time_limit=240,   # 4 min soft limit

    # Rate limiting (por worker)
    task_default_rate_limit="30/m",

    # Prefetch - 1 task por vez (evita "sequestrar" tasks)
    worker_prefetch_multiplier=1,

    # Acks late - retry se worker cair
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Serialização
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Resultados expiram em 1 hora
    result_expires=3600,
)
