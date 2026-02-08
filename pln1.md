# PLN1: Separar Enriquecimento da Ingestao

**Status: RUNPOD CONCLUIDO - AGUARDANDO VPS**

**Data:** 2026-01-26
**Servidor:** Testado e funcionando

---

## Objetivo
Mover a fase de enriquecimento LLM para **DEPOIS** da ingestao, permitindo:
1. Disparo manual pelo usuario
2. Disparo automatico via checkbox no frontend

---

## Pipeline Antes vs Depois

### ANTES (enriquecimento inline):
```
PDF -> Docling -> Parsing -> Extracao -> Materializacao -> [ENRIQUECIMENTO LLM] -> Embeddings -> Response
                                                              (2-3s por chunk)
```

### DEPOIS (enriquecimento separado):
```
INGESTAO (rapido):
PDF -> Docling -> Parsing -> Extracao -> Materializacao -> Embeddings -> Response

ENRIQUECIMENTO (separado, via VPS):
Trigger (manual/auto) -> Celery Tasks -> LLM -> Atualiza Milvus
```

---

## Tarefas RunPod

### Tarefa 1: Remover enriquecimento do pipeline.py [CONCLUIDA]
**Arquivo:** `src/ingestion/pipeline.py`

**Mudancas realizadas:**
- Comentado chamada `_phase_enrichment_acordao()` do fluxo principal
- Metodo original preservado para uso futuro
- Parametro `skip_enrichment` mantido (sera usado pela VPS)

**Verificacao:** Servidor iniciado sem erros, /health retorna "healthy"

---

### Tarefa 2: Documentacao para VPS [CONCLUIDA]
**Arquivos criados:**
- `docs/VPS_ENRICH_INTEGRATION.md` - Documentacao completa da arquitetura
- `docs/vps_enrich_router.py` - Codigo de referencia para endpoints

**Arquitetura Final:**
```
RUNPOD (GPU) - gpu.vectorgov.io:
  POST /embed      -> BGE-M3 embeddings
  POST /rerank     -> BGE Reranker
  POST /ingest     -> Docling (PDF -> chunks)
  (SEM endpoints /enrich/*)

VPS (77.37.43.160):
  POST /api/v1/ingest/enrich/start
  GET  /api/v1/ingest/enrich/status/{task_id}
  POST /api/v1/ingest/enrich/cancel/{task_id}
  GET  /api/v1/ingest/enrich/document/{document_id}
  + Milvus (localhost:19530)
  + Redis (localhost:6379)
  + Celery workers
```

---

### Tarefa 3: Atualizar IngestResponse [OPCIONAL - AGUARDAR VPS]
**Arquivo:** `src/ingestion/models.py`

**Mudancas propostas:**
- Adicionar campo `enrichment_status: str = "pending"`
- Possiveis valores: "pending", "in_progress", "completed", "skipped"

**Nota:** Implementar somente se VPS precisar dessa informacao no response.

---

## Tarefas VPS (NAO EXECUTAR AQUI)

### 1. Implementar endpoints /enrich/*
Usar codigo de referencia em `docs/vps_enrich_router.py`

### 2. Configurar Celery workers
```bash
# Worker para enriquecimento LLM (6 workers)
celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=6 -Q llm_enrich

# Worker para embeddings (2 workers)
celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=2 -Q embed_store
```

### 3. Atualizar frontend
- Checkbox "Enriquecer automaticamente apos indexacao"
- Coluna de status na lista de documentos
- Botao "Enriquecer" para documentos pendentes

### 4. Banco de dados
```sql
ALTER TABLE documentos ADD COLUMN enrichment_status VARCHAR(20) DEFAULT 'pending';
ALTER TABLE documentos ADD COLUMN enrichment_task_id VARCHAR(32);
ALTER TABLE documentos ADD COLUMN enrichment_progress FLOAT DEFAULT 0;
```

---

## Rollback (se necessario)

Para reverter no RunPod:
1. Descomentar linhas em `src/ingestion/pipeline.py` (buscar "PLN1")
2. Reiniciar servidor

---

## Historico

| Data       | Acao                                          | Status |
|------------|-----------------------------------------------|--------|
| 2026-01-26 | Tarefa 1: Remover enriquecimento do pipeline  | OK     |
| 2026-01-26 | Tarefa 2: Criar documentacao para VPS         | OK     |
| 2026-01-26 | Servidor testado e funcionando                | OK     |
| -          | VPS: Implementar endpoints                    | PENDENTE |
| -          | VPS: Atualizar frontend                       | PENDENTE |
