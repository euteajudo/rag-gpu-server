"""
Celery tasks para enriquecimento de chunks - Arquitetura 2 Filas.

FILAS:
- llm_enrich: Tasks que chamam o Qwen (vLLM) - 6 workers
- embed_store: Tasks de embedding (BGE-M3) + Milvus - 2 workers

IMPORTANTE: As variaveis GPU_SERVER_URL e VLLM_BASE_URL devem estar
configuradas em /etc/rag-api.env.
"""

import os
import sys
import time
import json
import logging
import re
from pathlib import Path
from typing import Optional, List

import redis

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .celery_app import app

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACOES
# =============================================================================

GPU_SERVER_URL = os.environ.get("GPU_SERVER_URL")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
MILVUS_HOST = os.environ.get("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

# Validacao
if not GPU_SERVER_URL:
    raise ValueError("GPU_SERVER_URL nao configurado. Defina em /etc/rag-api.env")
if not VLLM_BASE_URL:
    raise ValueError("VLLM_BASE_URL nao configurado. Defina em /etc/rag-api.env")


def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=2, decode_responses=True)


def update_enrich_progress(task_id: str, success: bool, chunk_id: str, error: str = None):
    """Atualiza progresso no Redis."""
    try:
        r = get_redis()
        key = f"enrich:task:{task_id}"
        data = r.get(key)
        if data:
            status = json.loads(data)
            if success:
                status["chunks_completed"] = status.get("chunks_completed", 0) + 1
            else:
                status["chunks_failed"] = status.get("chunks_failed", 0) + 1
                if error:
                    errors = status.get("errors", [])
                    errors.append(f"{chunk_id}: {error[:100]}")
                    status["errors"] = errors[-50:]
            r.setex(key, 86400, json.dumps(status))
    except Exception as e:
        logger.warning(f"Erro ao atualizar progresso: {e}")


# =============================================================================
# TASK 1: LLM ENRICHMENT (Fila: llm_enrich)
# =============================================================================

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def enrich_chunk_llm(
    self,
    chunk_id: str,
    text: str,
    device_type: str,
    article_number: str,
    document_id: str,
    document_type: str,
    number: str,
    year: int,
    enrich_task_id: str = None,
) -> dict:
    """
    Fase 1: Chama Qwen (vLLM) para gerar enriquecimento.

    Roda na fila 'llm_enrich' com 6 workers.
    Ao terminar, dispara embed_and_store na fila 'embed_store'.
    """
    from remote import RemoteLLM
    from remote.llm import RemoteLLMConfig

    start_time = time.time()
    logger.info(f"[LLM] Iniciando chunk: {chunk_id}")

    try:
        # Inicializa cliente LLM
        llm_config = RemoteLLMConfig(
            vllm_base_url=VLLM_BASE_URL,
            model=VLLM_MODEL,
            temperature=0.1,
            max_tokens=256,  # Otimizado: JSON curto
        )
        llm = RemoteLLM(llm_config)

        # Prepara contexto
        doc_context = f"{document_type} {number}/{year}"
        if document_type == "LEI":
            issuing_body = "Presidencia da Republica"
        elif document_type == "IN":
            issuing_body = "SEGES/ME"
        else:
            issuing_body = "Orgao Publico"

        # Prompt
        system_prompt = """Voce e um especialista em legislacao brasileira.
Analise o texto legal fornecido e gere:
1. context_header: Uma frase curta (max 150 caracteres) contextualizando o dispositivo
2. thesis_text: Resumo objetivo do que o dispositivo estabelece (max 300 caracteres)
3. thesis_type: Classificacao (definicao, procedimento, prazo, obrigacao, proibicao, excecao, disposicao)
4. synthetic_questions: 3 perguntas que este dispositivo responde

Responda APENAS em JSON valido."""

        user_prompt = f"""/no_think
Documento: {doc_context} ({issuing_body})
Tipo de dispositivo: {device_type}
Artigo: {article_number}

Texto:
{text}

Responda em JSON:
{{"context_header": "...", "thesis_text": "...", "thesis_type": "...", "synthetic_questions": ["...", "...", "..."]}}"""

        # Chama LLM
        llm_response = llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        response = llm_response.content

        # Parse JSON
        try:
            response_text = response.strip()
            if response_text.startswith("```"):
                parts = response_text.split("```")
                if len(parts) > 1:
                    response_text = parts[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
            enrichment = json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                enrichment = json.loads(match.group())
            else:
                raise ValueError(f"Resposta nao e JSON valido: {response[:200]}")

        context_header = enrichment.get("context_header", "")[:500]
        thesis_text = enrichment.get("thesis_text", "")[:1000]
        thesis_type = enrichment.get("thesis_type", "disposicao")
        synthetic_questions = enrichment.get("synthetic_questions", [])
        if isinstance(synthetic_questions, str):
            synthetic_questions = [synthetic_questions]

        llm_elapsed = time.time() - start_time
        logger.info(f"[LLM] Completo em {llm_elapsed:.1f}s: {chunk_id}")

        # Dispara proxima fase na fila embed_store
        embed_and_store.apply_async(
            kwargs={
                "chunk_id": chunk_id,
                "text": text,
                "context_header": context_header,
                "thesis_text": thesis_text,
                "thesis_type": thesis_type,
                "synthetic_questions": synthetic_questions,
                "enrich_task_id": enrich_task_id,
            },
            queue="embed_store",
        )

        return {
            "success": True,
            "chunk_id": chunk_id,
            "phase": "llm",
            "elapsed": llm_elapsed,
        }

    except Exception as e:
        logger.error(f"[LLM] Erro em {chunk_id}: {e}")
        if enrich_task_id:
            update_enrich_progress(enrich_task_id, False, chunk_id, str(e))
        try:
            self.retry(exc=e)
        except self.MaxRetriesExceededError:
            return {"success": False, "chunk_id": chunk_id, "error": str(e)}


# =============================================================================
# TASK 2: EMBED AND STORE (Fila: embed_store)
# =============================================================================

@app.task(bind=True, max_retries=3, default_retry_delay=30)
def embed_and_store(
    self,
    chunk_id: str,
    text: str,
    context_header: str,
    thesis_text: str,
    thesis_type: str,
    synthetic_questions: List[str],
    enrich_task_id: str = None,
) -> dict:
    """
    Fase 2: Gera embeddings (BGE-M3) e atualiza Milvus.

    Roda na fila 'embed_store' com 2 workers.
    """
    from pymilvus import connections, Collection
    from remote import RemoteEmbedder
    from remote.embedder import RemoteEmbedderConfig

    start_time = time.time()
    logger.info(f"[EMBED] Iniciando chunk: {chunk_id}")

    try:
        # Inicializa embedder
        embedder_config = RemoteEmbedderConfig(gpu_server_url=GPU_SERVER_URL, gpu_api_key=os.getenv("GPU_API_KEY", ""))
        embedder = RemoteEmbedder(embedder_config)

        # Monta enriched_text
        enriched_text = f"[CONTEXTO: {context_header}]\n\n{text}"
        if synthetic_questions:
            questions_str = "\n".join(f"- {q}" for q in synthetic_questions[:5])
            enriched_text += f"\n\n[PERGUNTAS RELACIONADAS:\n{questions_str}]"

        # Gera embeddings
        embed_result = embedder.encode([enriched_text])
        dense_vector = embed_result.dense_embeddings[0]
        sparse_vector = embed_result.sparse_embeddings[0]

        # Thesis vector
        if thesis_text:
            thesis_result = embedder.encode([thesis_text])
            thesis_vector = thesis_result.dense_embeddings[0]
        else:
            thesis_vector = dense_vector

        # Conecta ao Milvus
        connections.connect(alias="embed_task", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection("leis_v3", using="embed_task")
        collection.load()

        # Busca chunk existente
        results = collection.query(
            expr=f'chunk_id == "{chunk_id}"',
            output_fields=["*"],
            limit=1,
        )

        if not results:
            connections.disconnect("embed_task")
            error_msg = "Chunk nao encontrado no Milvus"
            if enrich_task_id:
                update_enrich_progress(enrich_task_id, False, chunk_id, error_msg)
            return {"success": False, "chunk_id": chunk_id, "error": error_msg}

        chunk = results[0]

        # Prepara dados para upsert
        row = {
            "chunk_id": chunk["chunk_id"],
            "parent_chunk_id": chunk.get("parent_chunk_id", ""),
            "span_id": chunk.get("span_id", ""),
            "device_type": chunk.get("device_type", ""),
            "chunk_level": chunk.get("chunk_level", ""),
            "text": chunk["text"],
            "enriched_text": enriched_text,
            "dense_vector": dense_vector,
            "thesis_vector": thesis_vector,
            "sparse_vector": sparse_vector,
            "context_header": context_header,
            "thesis_text": thesis_text,
            "thesis_type": thesis_type,
            "synthetic_questions": "\n".join(synthetic_questions),
            "aliases": chunk.get("aliases", ""),
            "sparse_source": chunk.get("sparse_source", ""),
            "citations": chunk.get("citations", ""),
            "document_id": chunk["document_id"],
            "tipo_documento": chunk.get("tipo_documento", ""),
            "numero": chunk.get("numero", ""),
            "ano": chunk.get("ano", 0),
            "article_number": chunk.get("article_number", ""),
            "schema_version": chunk.get("schema_version", "1.0.0"),
            "extractor_version": chunk.get("extractor_version", "1.0.0"),
            "ingestion_timestamp": chunk.get("ingestion_timestamp", ""),
            "document_hash": chunk.get("document_hash", ""),
        }

        # Delete e insert (sem flush - deixa Milvus gerenciar)
        collection.delete(expr=f'chunk_id == "{chunk_id}"')
        collection.insert([row])
        # Sem flush() - otimização de I/O

        connections.disconnect("embed_task")

        elapsed = time.time() - start_time
        logger.info(f"[EMBED] Completo em {elapsed:.1f}s: {chunk_id}")

        # Atualiza progresso
        if enrich_task_id:
            update_enrich_progress(enrich_task_id, True, chunk_id)

        return {
            "success": True,
            "chunk_id": chunk_id,
            "phase": "embed_store",
            "context_header": context_header[:100],
            "thesis_type": thesis_type,
            "elapsed": elapsed,
        }

    except Exception as e:
        logger.error(f"[EMBED] Erro em {chunk_id}: {e}")
        if enrich_task_id:
            update_enrich_progress(enrich_task_id, False, chunk_id, str(e))
        try:
            self.retry(exc=e)
        except self.MaxRetriesExceededError:
            return {"success": False, "chunk_id": chunk_id, "error": str(e)}


# =============================================================================
# TASK BATCH (dispara na fila llm_enrich)
# =============================================================================

@app.task
def enrich_batch_task(chunk_ids: List[str], enrich_task_id: str = None) -> dict:
    """
    Enriquece um batch de chunks.
    Dispara tasks na fila llm_enrich.
    """
    from pymilvus import connections, Collection

    connections.connect(alias="batch", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection("leis_v3", using="batch")
    collection.load()

    dispatched = 0
    for chunk_id in chunk_ids:
        results = collection.query(
            expr=f'chunk_id == "{chunk_id}"',
            output_fields=["text", "device_type", "article_number", "document_id", "tipo_documento", "numero", "ano"],
            limit=1,
        )

        if results:
            chunk = results[0]
            enrich_chunk_llm.apply_async(
                kwargs={
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "device_type": chunk["device_type"],
                    "article_number": chunk["article_number"],
                    "document_id": chunk["document_id"],
                    "document_type": chunk["tipo_documento"],
                    "number": chunk["numero"],
                    "year": chunk["ano"],
                    "enrich_task_id": enrich_task_id,
                },
                queue="llm_enrich",
            )
            dispatched += 1

    connections.disconnect("batch")

    return {
        "dispatched": dispatched,
        "total_requested": len(chunk_ids),
    }


# =============================================================================
# TASK LEGADO (mantido para compatibilidade)
# =============================================================================

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def enrich_chunk_task(
    self,
    chunk_id: str,
    text: str,
    device_type: str,
    article_number: str,
    document_id: str,
    document_type: str,
    number: str,
    year: int,
    enrich_task_id: str = None,
) -> dict:
    """
    Task legado - redireciona para nova arquitetura.
    """
    enrich_chunk_llm.apply_async(
        kwargs={
            "chunk_id": chunk_id,
            "text": text,
            "device_type": device_type,
            "article_number": article_number,
            "document_id": document_id,
            "document_type": document_type,
            "number": number,
            "year": year,
            "enrich_task_id": enrich_task_id,
        },
        queue="llm_enrich",
    )
    return {"redirected": True, "chunk_id": chunk_id}


# =============================================================================
# TASK: MONITOR + AUTO-VALIDACAO
# =============================================================================

@app.task(bind=True, max_retries=60, default_retry_delay=30)
def monitor_and_validate(
    self,
    document_id: str,
    total_chunks: int,
    enrich_task_id: str,
    collection_name: str = "leis_v3",
) -> dict:
    """
    Monitora progresso do enriquecimento e dispara validação quando completo.
    
    Fluxo:
    1. Verifica progresso via Redis (enrich:task:{enrich_task_id})
    2. Se completo (completed >= total), dispara validação
    3. Se não completo, faz retry (max 60 tentativas = 30 min)
    """
    import requests
    
    try:
        r = get_redis()
        key = f"enrich:task:{enrich_task_id}"
        data = r.get(key)
        
        if not data:
            logger.warning(f"[MONITOR] Task {enrich_task_id} nao encontrada no Redis")
            return {"success": False, "error": "Task nao encontrada"}
        
        status = json.loads(data)
        completed = status.get("chunks_completed", 0)
        failed = status.get("chunks_failed", 0)
        processed = completed + failed
        
        logger.info(f"[MONITOR] {document_id}: {processed}/{total_chunks} processados ({completed} ok, {failed} erros)")
        
        # Ainda em andamento?
        if processed < total_chunks:
            # Retry - vai tentar novamente em 30s
            raise self.retry(
                exc=Exception(f"Enriquecimento em andamento: {processed}/{total_chunks}"),
                countdown=30,
            )
        
        # Enriquecimento completo! Dispara validação
        logger.info(f"[MONITOR] Enriquecimento completo para {document_id}. Disparando validação...")
        
        # Atualiza status no Redis
        status["validation_triggered"] = True
        status["validation_triggered_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        r.setex(key, 86400, json.dumps(status))
        
        # Dispara validação via API interna
        try:
            response = requests.post(
                f"http://127.0.0.1:8000/api/v1/validation/collections/{collection_name}/validate",
                params={"document_id": document_id, "force": False},
                headers={"X-Internal-Request": "true"},
                timeout=30,
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[MONITOR] Validação disparada: {result.get('task_id')}")
                return {
                    "success": True,
                    "document_id": document_id,
                    "enrichment_completed": completed,
                    "enrichment_failed": failed,
                    "validation_task_id": result.get("task_id"),
                }
            else:
                logger.error(f"[MONITOR] Erro ao disparar validação: {response.status_code} - {response.text}")
                return {
                    "success": True,
                    "document_id": document_id,
                    "enrichment_completed": completed,
                    "enrichment_failed": failed,
                    "validation_error": f"HTTP {response.status_code}",
                }
        except requests.RequestException as e:
            logger.error(f"[MONITOR] Erro de conexão ao disparar validação: {e}")
            return {
                "success": True,
                "document_id": document_id,
                "enrichment_completed": completed,
                "enrichment_failed": failed,
                "validation_error": str(e),
            }
    
    except self.MaxRetriesExceededError:
        logger.error(f"[MONITOR] Timeout aguardando enriquecimento de {document_id}")
        return {
            "success": False,
            "document_id": document_id,
            "error": "Timeout aguardando enriquecimento (30 min)",
        }
