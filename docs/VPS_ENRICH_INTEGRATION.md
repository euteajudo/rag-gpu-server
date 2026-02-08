# VPS - Implementacao dos Endpoints de Enriquecimento

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (Next.js) - vectorgov.io/admin/ingestao                       │
│  Chama: /api/v1/ingest/enrich/* (VPS)                                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  VPS (77.37.43.160) - FastAPI + Celery                                  │
│                                                                         │
│  Endpoints a implementar:                                               │
│  ├── POST /api/v1/ingest/enrich/start                                   │
│  ├── GET  /api/v1/ingest/enrich/status/{task_id}                        │
│  ├── POST /api/v1/ingest/enrich/cancel/{task_id}                        │
│  └── GET  /api/v1/ingest/enrich/document/{document_id}                  │
│                                                                         │
│  Servicos locais:                                                       │
│  ├── Milvus (localhost:19530)                                           │
│  ├── Redis (localhost:6379)                                             │
│  └── Celery Workers (tasks.py)                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS (Cloudflare Tunnel)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  RUNPOD (GPU) - Apenas servicos de ML                                   │
│                                                                         │
│  gpu.vectorgov.io                                                       │
│  ├── POST /embed      → BGE-M3 embeddings                               │
│  ├── POST /rerank     → BGE Reranker                                    │
│  └── POST /ingest     → Docling (PDF → chunks)                          │
│                                                                         │
│  llm.vectorgov.io                                                       │
│  └── POST /v1/chat/completions → vLLM (Qwen3-8B-AWQ)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## O que mudou no RunPod

O enriquecimento LLM foi **removido** do pipeline de ingestao.

**Antes:**
```
PDF → Docling → Parsing → Extração → Materialização → [Enriquecimento LLM] → Embeddings → Response
```

**Depois:**
```
PDF → Docling → Parsing → Extração → Materialização → Embeddings → Response
(Enriquecimento é disparado separadamente pela VPS)
```

## Endpoints a Implementar na VPS

### 1. POST /api/v1/ingest/enrich/start

Inicia enriquecimento de um documento.

**Request:**
```json
{
  "document_id": "LEI-14133-2021",
  "collection_name": "leis_v4"
}
```

**Response (200):**
```json
{
  "enrich_task_id": "a1b2c3d4e5f6g7h8",
  "document_id": "LEI-14133-2021",
  "total_chunks": 47,
  "message": "Enriquecimento iniciado: 47 chunks na fila"
}
```

**Logica:**
1. Buscar chunks do documento no Milvus (localhost:19530)
2. Filtrar chunks que ainda nao tem enriched_text
3. Registrar task no Redis
4. Disparar Celery tasks (enrich_chunk_llm) para cada chunk
5. Retornar enrich_task_id

---

### 2. GET /api/v1/ingest/enrich/status/{enrich_task_id}

Verifica progresso do enriquecimento.

**Response (200):**
```json
{
  "enrich_task_id": "a1b2c3d4e5f6g7h8",
  "document_id": "LEI-14133-2021",
  "status": "in_progress",
  "total_chunks": 47,
  "chunks_completed": 23,
  "chunks_failed": 1,
  "progress_percent": 51.1,
  "started_at": "2026-01-26T15:30:00",
  "completed_at": null,
  "errors": ["LEI-14133-2021#ART-015: Timeout LLM"]
}
```

**Status possiveis:**
- `pending` - Aguardando inicio
- `in_progress` - Processando chunks
- `completed` - Todos processados
- `failed` - Mais de 50% de erros
- `cancelled` - Cancelado pelo usuario

---

### 3. POST /api/v1/ingest/enrich/cancel/{enrich_task_id}

Cancela enriquecimento em andamento.

**Response (200):**
```json
{
  "enrich_task_id": "a1b2c3d4e5f6g7h8",
  "status": "cancelled",
  "message": "Task marcada como cancelada"
}
```

---

### 4. GET /api/v1/ingest/enrich/document/{document_id}

Verifica status de enriquecimento de um documento.

**Query params:**
- `collection_name`: Nome da collection (default: `leis_v4`)

**Response (200):**
```json
{
  "document_id": "LEI-14133-2021",
  "total_chunks": 47,
  "enriched_chunks": 45,
  "not_enriched_chunks": 2,
  "enrichment_percent": 95.7,
  "is_fully_enriched": false,
  "not_enriched_chunk_ids": ["LEI-14133-2021#ART-015", "LEI-14133-2021#ART-016"]
}
```

---

## Codigo de Referencia

O arquivo `docs/vps_enrich_router.py` contem uma implementacao de referencia
dos endpoints. Adapte para a estrutura da VPS.

---

## Fluxo de Enriquecimento (Celery)

O codigo Celery ja existe em `src/enrichment/tasks.py` e funciona assim:

```
1. POST /enrich/start (VPS endpoint)
   │
   ├─▶ Busca chunks no Milvus
   ├─▶ Registra task no Redis
   └─▶ Dispara enrich_chunk_llm.apply_async() para cada chunk

2. enrich_chunk_llm (Celery task - fila llm_enrich)
   │
   ├─▶ Chama llm.vectorgov.io/v1/chat/completions
   │   (gera context_header, thesis_text, thesis_type, synthetic_questions)
   │
   └─▶ Dispara embed_and_store.apply_async()

3. embed_and_store (Celery task - fila embed_store)
   │
   ├─▶ Chama gpu.vectorgov.io/embed
   │   (gera dense_vector, sparse_vector, thesis_vector)
   │
   └─▶ Atualiza chunk no Milvus
```

---

## Configuracao Necessaria na VPS

### Variaveis de Ambiente (/etc/rag-api.env)

```bash
# RunPod endpoints
GPU_SERVER_URL=https://gpu.vectorgov.io
VLLM_BASE_URL=https://llm.vectorgov.io/v1
GPU_API_KEY=<api_key_do_runpod>

# Servicos locais
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
```

### Celery Workers

```bash
# Worker para enriquecimento LLM (6 workers)
celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=6 -Q llm_enrich

# Worker para embeddings (2 workers)
celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=2 -Q embed_store
```

---

## Mudancas no Frontend

### 1. Formulario de Upload

Adicionar checkbox:
```html
<input type="checkbox" id="enrich_after_ingest" name="enrich_after_ingest">
<label>Enriquecer automaticamente apos indexacao</label>
```

Se marcado: Apos ingestao completar, chamar POST /enrich/start

### 2. Lista de Documentos

Adicionar coluna "Enriquecimento":
- `pending` → Botao "Enriquecer" (azul)
- `in_progress` → Spinner + percentual
- `completed` → Badge verde
- `failed` → Badge vermelho + Botao "Retry"

### 3. Banco de Dados

```sql
ALTER TABLE documentos ADD COLUMN enrichment_status VARCHAR(20) DEFAULT 'pending';
ALTER TABLE documentos ADD COLUMN enrichment_task_id VARCHAR(32);
ALTER TABLE documentos ADD COLUMN enrichment_progress FLOAT DEFAULT 0;
```

---

## Checklist de Implementacao

- [ ] Criar router FastAPI com os 4 endpoints
- [ ] Integrar com Celery tasks existentes (src/enrichment/tasks.py)
- [ ] Adicionar tracking no Redis
- [ ] Atualizar frontend com checkbox e status
- [ ] Adicionar colunas no banco de dados
- [ ] Testar fluxo completo: upload → ingest → enrich
