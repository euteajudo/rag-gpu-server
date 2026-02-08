"""
Router FastAPI para enriquecimento de chunks.

Endpoints:
    POST /enrich/start           - Inicia enriquecimento de um documento
    GET  /enrich/status/{task_id} - Verifica status do enriquecimento
    POST /enrich/cancel/{task_id} - Cancela enriquecimento em andamento
    GET  /enrich/health          - Health check do modulo

O enriquecimento usa Celery para processamento distribuido:
- Fila 'llm_enrich': Workers que chamam o LLM (Qwen)
- Fila 'embed_store': Workers que geram embeddings e atualizam Milvus
"""

import json
import logging
import time
import hashlib
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enrich", tags=["Enrichment"])


# =============================================================================
# MODELS
# =============================================================================

class EnrichStartRequest(BaseModel):
    """Request para iniciar enriquecimento."""
    document_id: str = Field(..., description="ID do documento (ex: LEI-14133-2021)")
    collection_name: str = Field("leis_v4", description="Nome da collection Milvus")


class EnrichStartResponse(BaseModel):
    """Response do inicio de enriquecimento."""
    enrich_task_id: str
    document_id: str
    total_chunks: int
    message: str


class EnrichStatusResponse(BaseModel):
    """Response do status de enriquecimento."""
    enrich_task_id: str
    document_id: str
    status: str  # pending, in_progress, completed, failed, cancelled
    total_chunks: int
    chunks_completed: int
    chunks_failed: int
    progress_percent: float
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    errors: List[str] = []


class EnrichCancelResponse(BaseModel):
    """Response do cancelamento."""
    enrich_task_id: str
    status: str
    message: str


# =============================================================================
# HELPERS
# =============================================================================

def _get_redis():
    """Retorna conexao Redis."""
    import redis
    import os

    host = os.environ.get("REDIS_HOST", "127.0.0.1")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    return redis.Redis(host=host, port=port, db=2, decode_responses=True)


def _generate_enrich_task_id(document_id: str) -> str:
    """Gera ID unico para task de enriquecimento."""
    hash_input = f"enrich-{document_id}-{time.time()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def _get_milvus_connection():
    """Retorna conexao Milvus."""
    import os
    from pymilvus import connections

    host = os.environ.get("MILVUS_HOST", "127.0.0.1")
    port = os.environ.get("MILVUS_PORT", "19530")

    # Usa alias unico para evitar conflitos
    alias = f"enrich_router_{time.time_ns()}"
    connections.connect(alias=alias, host=host, port=port)
    return alias


def _disconnect_milvus(alias: str):
    """Desconecta do Milvus."""
    from pymilvus import connections
    try:
        connections.disconnect(alias)
    except Exception:
        pass


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/start", response_model=EnrichStartResponse)
async def start_enrichment(request: EnrichStartRequest):
    """
    Inicia enriquecimento de todos os chunks de um documento.

    O enriquecimento roda em background via Celery:
    1. Busca todos os chunks do documento no Milvus
    2. Dispara tasks de enriquecimento LLM para cada chunk
    3. Cada task LLM dispara task de embedding/store ao terminar

    Use GET /enrich/status/{enrich_task_id} para acompanhar progresso.
    """
    from pymilvus import Collection

    alias = None
    try:
        # Conecta ao Milvus
        alias = _get_milvus_connection()
        collection = Collection(request.collection_name, using=alias)
        collection.load()

        # Busca chunks do documento
        results = collection.query(
            expr=f'document_id == "{request.document_id}"',
            output_fields=[
                "chunk_id", "text", "device_type", "article_number",
                "tipo_documento", "numero", "ano", "enriched_text"
            ],
            limit=10000,  # Maximo de chunks por documento
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Documento {request.document_id} nao encontrado na collection {request.collection_name}"
            )

        # Filtra chunks que ainda nao foram enriquecidos (enriched_text vazio ou igual ao text)
        chunks_to_enrich = []
        for chunk in results:
            enriched = chunk.get("enriched_text", "")
            original = chunk.get("text", "")
            # Se enriched_text esta vazio ou e igual ao original, precisa enriquecer
            if not enriched or enriched == original or not enriched.startswith("[CONTEXTO:"):
                chunks_to_enrich.append(chunk)

        if not chunks_to_enrich:
            return EnrichStartResponse(
                enrich_task_id="",
                document_id=request.document_id,
                total_chunks=0,
                message=f"Todos os {len(results)} chunks ja estao enriquecidos"
            )

        # Gera task ID
        enrich_task_id = _generate_enrich_task_id(request.document_id)

        # Registra task no Redis
        r = _get_redis()
        task_status = {
            "document_id": request.document_id,
            "collection_name": request.collection_name,
            "status": "in_progress",
            "total_chunks": len(chunks_to_enrich),
            "chunks_completed": 0,
            "chunks_failed": 0,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "errors": [],
        }
        r.setex(f"enrich:task:{enrich_task_id}", 86400, json.dumps(task_status))

        # Dispara tasks Celery
        from .tasks import enrich_chunk_llm

        dispatched = 0
        for chunk in chunks_to_enrich:
            try:
                enrich_chunk_llm.apply_async(
                    kwargs={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "device_type": chunk.get("device_type", "article"),
                        "article_number": chunk.get("article_number", ""),
                        "document_id": request.document_id,
                        "document_type": chunk.get("tipo_documento", "LEI"),
                        "number": chunk.get("numero", ""),
                        "year": chunk.get("ano", 0),
                        "enrich_task_id": enrich_task_id,
                    },
                    queue="llm_enrich",
                )
                dispatched += 1
            except Exception as e:
                logger.error(f"Erro ao disparar task para {chunk['chunk_id']}: {e}")

        logger.info(
            f"[Enrich] Iniciado {enrich_task_id} para {request.document_id}: "
            f"{dispatched}/{len(chunks_to_enrich)} tasks disparadas"
        )

        return EnrichStartResponse(
            enrich_task_id=enrich_task_id,
            document_id=request.document_id,
            total_chunks=len(chunks_to_enrich),
            message=f"Enriquecimento iniciado: {dispatched} chunks na fila"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao iniciar enriquecimento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if alias:
            _disconnect_milvus(alias)


@router.get("/status/{enrich_task_id}", response_model=EnrichStatusResponse)
async def get_enrichment_status(enrich_task_id: str):
    """
    Verifica o status de uma task de enriquecimento.

    Status possiveis:
    - pending: Aguardando inicio
    - in_progress: Processando chunks
    - completed: Todos os chunks processados
    - failed: Falhou (mais de 50% de erros)
    - cancelled: Cancelado pelo usuario
    """
    try:
        r = _get_redis()
        key = f"enrich:task:{enrich_task_id}"
        data = r.get(key)

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Task {enrich_task_id} nao encontrada"
            )

        status = json.loads(data)

        total = status.get("total_chunks", 0)
        completed = status.get("chunks_completed", 0)
        failed = status.get("chunks_failed", 0)
        processed = completed + failed

        # Calcula progresso
        progress = (processed / total * 100) if total > 0 else 0

        # Determina status final
        current_status = status.get("status", "in_progress")
        if current_status == "in_progress" and processed >= total:
            # Todos processados - verifica se completou ou falhou
            if failed > total * 0.5:
                current_status = "failed"
            else:
                current_status = "completed"

            # Atualiza status no Redis
            status["status"] = current_status
            status["completed_at"] = datetime.now().isoformat()
            r.setex(key, 86400, json.dumps(status))

        return EnrichStatusResponse(
            enrich_task_id=enrich_task_id,
            document_id=status.get("document_id", ""),
            status=current_status,
            total_chunks=total,
            chunks_completed=completed,
            chunks_failed=failed,
            progress_percent=round(progress, 1),
            started_at=status.get("started_at"),
            completed_at=status.get("completed_at"),
            errors=status.get("errors", [])[-10:],  # Ultimos 10 erros
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao verificar status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel/{enrich_task_id}", response_model=EnrichCancelResponse)
async def cancel_enrichment(enrich_task_id: str):
    """
    Cancela uma task de enriquecimento em andamento.

    NOTA: Tasks ja disparadas no Celery continuarao rodando,
    mas novas tasks nao serao processadas e o status sera marcado como cancelled.
    """
    try:
        r = _get_redis()
        key = f"enrich:task:{enrich_task_id}"
        data = r.get(key)

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Task {enrich_task_id} nao encontrada"
            )

        status = json.loads(data)

        if status.get("status") in ["completed", "failed", "cancelled"]:
            return EnrichCancelResponse(
                enrich_task_id=enrich_task_id,
                status=status.get("status"),
                message=f"Task ja finalizada com status: {status.get('status')}"
            )

        # Marca como cancelada
        status["status"] = "cancelled"
        status["completed_at"] = datetime.now().isoformat()
        r.setex(key, 86400, json.dumps(status))

        logger.info(f"[Enrich] Task {enrich_task_id} cancelada")

        return EnrichCancelResponse(
            enrich_task_id=enrich_task_id,
            status="cancelled",
            message="Task marcada como cancelada. Tasks ja em execucao continuarao."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao cancelar: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def enrich_health():
    """
    Health check do modulo de enriquecimento.

    Verifica:
    - Conexao Redis
    - Conexao Celery (broker)
    """
    health = {
        "status": "healthy",
        "redis": {"status": "unknown"},
        "celery": {"status": "unknown"},
    }

    # Verifica Redis
    try:
        r = _get_redis()
        r.ping()
        health["redis"] = {"status": "connected"}
    except Exception as e:
        health["redis"] = {"status": "error", "error": str(e)}
        health["status"] = "degraded"

    # Verifica Celery
    try:
        from .celery_app import app as celery_app
        inspector = celery_app.control.inspect()

        # Tenta pegar workers ativos (timeout curto)
        active = inspector.active()
        if active:
            workers = list(active.keys())
            health["celery"] = {
                "status": "connected",
                "workers": workers,
                "worker_count": len(workers),
            }
        else:
            health["celery"] = {
                "status": "no_workers",
                "message": "Nenhum worker Celery ativo"
            }
            health["status"] = "degraded"
    except Exception as e:
        health["celery"] = {"status": "error", "error": str(e)}
        health["status"] = "degraded"

    return health


@router.get("/document/{document_id}")
async def get_document_enrichment_status(
    document_id: str,
    collection_name: str = Query("leis_v4", description="Nome da collection Milvus")
):
    """
    Verifica status de enriquecimento de um documento especifico.

    Retorna quantos chunks estao enriquecidos vs total.
    """
    from pymilvus import Collection

    alias = None
    try:
        alias = _get_milvus_connection()
        collection = Collection(collection_name, using=alias)
        collection.load()

        # Busca todos os chunks do documento
        results = collection.query(
            expr=f'document_id == "{document_id}"',
            output_fields=["chunk_id", "text", "enriched_text", "context_header"],
            limit=10000,
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Documento {document_id} nao encontrado"
            )

        total = len(results)
        enriched = 0
        not_enriched = []

        for chunk in results:
            enriched_text = chunk.get("enriched_text", "")
            context_header = chunk.get("context_header", "")

            # Considera enriquecido se tem context_header ou enriched_text com [CONTEXTO:]
            if context_header or (enriched_text and enriched_text.startswith("[CONTEXTO:")):
                enriched += 1
            else:
                not_enriched.append(chunk["chunk_id"])

        return {
            "document_id": document_id,
            "collection_name": collection_name,
            "total_chunks": total,
            "enriched_chunks": enriched,
            "not_enriched_chunks": total - enriched,
            "enrichment_percent": round(enriched / total * 100, 1) if total > 0 else 0,
            "is_fully_enriched": enriched == total,
            "not_enriched_chunk_ids": not_enriched[:20],  # Primeiros 20
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao verificar documento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if alias:
            _disconnect_milvus(alias)
