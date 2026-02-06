# Plano: Cache de Inspeção Aprovada para Re-Ingestão

> **Data**: 2026-02-06
> **Status**: PLANEJADO (não implementado)
> **De**: Claude Code (RunPod)
> **Para**: Claude Code (VPS) + futuro Claude Code (RunPod)
> **Prioridade**: Fase futura (após estabilização do pipeline VLM)

---

## 1. Problema

Quando o usuário quer reingerir uma norma que já passou pelo pipeline de inspeção e foi aprovada, o sistema hoje roda **todo o pipeline VLM do zero**: PyMuPDF → Qwen3-VL → offset resolution → etc.

Isso é desnecessário porque:
- A inspeção já validou a extração do VLM
- O usuário já aprovou a classificação dos dispositivos
- Os artefatos aprovados já estão persistidos no MinIO da VPS

## 2. Solução Proposta

Usar os artefatos de inspeção aprovada como **cache**. Na re-ingestão, se existir uma inspeção aprovada para o mesmo documento (mesmo PDF), pular toda a extração VLM e usar os chunks aprovados diretamente.

### Fluxo Atual (sem cache)

```
PDF → PyMuPDF → Qwen3-VL → offset resolution → embeddings → Milvus
                  ↑
           ~30-60s por documento (GPU intensivo)
```

### Fluxo Proposto (com cache)

```
PDF → Checar cache na VPS → Cache existe E PDF é o mesmo?
        │                         │
        │ NÃO                     │ SIM
        ↓                         ↓
    Pipeline VLM normal       Baixar chunks_preview.json
    (PyMuPDF → Qwen3-VL      + canonical.md + offsets.json
     → offset resolution)         ↓
        │                    Converter ChunkPreview → ProcessedChunk
        ↓                         │
    embeddings → Milvus      embeddings → Milvus
```

**Economia estimada**: Pula ~80% do tempo de processamento (toda a extração VLM).

---

## 3. Arquitetura: Dois Lados

### 3.1 Lado VPS (a implementar)

A VPS precisa de **2 novos endpoints** no router de inspeção:

#### Endpoint A: Consultar se existe cache

```
GET /api/v1/inspect/cache/{document_id}
```

**Request**:
```
GET /api/v1/inspect/cache/LEI-14133-2021
Headers:
  X-Ingest-Key: <api_key>
```

**Response (cache existe)**:
```json
{
  "has_cache": true,
  "document_id": "LEI-14133-2021",
  "approved_at": "2026-02-06T14:30:00Z",
  "approved_by": "admin",
  "sha256_source": "abc123def456...",
  "canonical_hash": "789fed...",
  "total_chunks": 47,
  "pipeline_version": "1.0.0"
}
```

**Response (sem cache)**:
```json
{
  "has_cache": false,
  "document_id": "LEI-14133-2021"
}
```

**Implementação na VPS**:
1. Verificar se existe `inspections/{document_id}/metadata.json` no MinIO
2. Se existir, ler o metadata e verificar `status == "APPROVED"`
3. Retornar os campos relevantes do metadata

#### Endpoint B: Baixar artefatos do cache

```
GET /api/v1/inspect/cache/{document_id}/artifacts
```

**Request**:
```
GET /api/v1/inspect/cache/LEI-14133-2021/artifacts
Headers:
  X-Ingest-Key: <api_key>
```

**Response** (JSON com os artefatos embutidos):
```json
{
  "document_id": "LEI-14133-2021",
  "canonical_md": "Art. 1º Esta Lei estabelece...",
  "canonical_hash": "789fed...",
  "offsets": {
    "ART-001": {"start": 0, "end": 245, "device_type": "article"},
    "ART-002": {"start": 246, "end": 512, "device_type": "article"},
    "PAR-002-1": {"start": 300, "end": 400, "device_type": "paragraph"}
  },
  "chunks": [
    {
      "span_id": "ART-001",
      "node_id": "leis:LEI-14133-2021#ART-001",
      "device_type": "article",
      "text": "Art. 1º Esta Lei estabelece...",
      "parent_span_id": "",
      "canonical_start": 0,
      "canonical_end": 245,
      "page_number": 1,
      "bbox": [72.0, 100.0, 540.0, 150.0],
      "confidence": 0.95
    }
  ]
}
```

**Implementação na VPS**:
1. Ler `inspections/{document_id}/chunks_preview.json` do MinIO
2. Ler `inspections/{document_id}/canonical.md` do MinIO
3. Ler `inspections/{document_id}/offsets.json` do MinIO
4. Montar a resposta JSON combinando os 3 artefatos
5. Se qualquer artefato faltar, retornar 404

**Nota sobre tamanho**: O canonical.md pode ser grande (100KB+ para leis longas). Se isso for um problema, considerar:
- Compressão gzip na resposta (Accept-Encoding)
- Ou dividir em 2 chamadas: uma para chunks+offsets, outra para canonical.md

---

### 3.2 Lado RunPod (a implementar)

O RunPod precisa de modificações no pipeline de ingestão (`src/ingestion/pipeline.py`):

#### Novo método: `_check_inspection_cache()`

```python
def _check_inspection_cache(
    self,
    document_id: str,
    pdf_sha256: str,
) -> Optional[dict]:
    """
    Consulta a VPS se existe inspeção aprovada para este documento.

    Args:
        document_id: ID do documento (ex: LEI-14133-2021)
        pdf_sha256: SHA256 do PDF sendo ingerido (para validar que é o mesmo)

    Returns:
        dict com artefatos do cache se existir E PDF for o mesmo.
        None se não existir cache ou PDF mudou.
    """
```

**Lógica**:
1. `GET /api/v1/inspect/cache/{document_id}` — verifica existência
2. Se `has_cache == false` → retorna None
3. Se `sha256_source != pdf_sha256` → retorna None (PDF mudou, cache inválido)
4. `GET /api/v1/inspect/cache/{document_id}/artifacts` — baixa artefatos
5. Retorna dict com chunks, canonical_md, offsets

#### Novo método: `_cache_to_processed_chunks()`

```python
def _cache_to_processed_chunks(
    self,
    cache_data: dict,
    request: IngestRequest,
) -> tuple[list[ProcessedChunk], str, str]:
    """
    Converte artefatos do cache em ProcessedChunks.

    Returns:
        (chunks, canonical_text, canonical_hash)
    """
```

**Lógica**:
1. Extrai `canonical_md` e `canonical_hash` do cache
2. Para cada chunk no cache, cria um `ProcessedChunk` com:
   - Todos os campos de identificação (node_id, span_id, device_type, etc.)
   - Metadados do documento vindos do `IngestRequest`
   - `canonical_start`/`canonical_end`/`canonical_hash` do cache
   - `text` do chunk (texto PyMuPDF reconciliado)
   - `bbox`, `page_number`, `confidence` do cache
3. Os campos que o cache NÃO tem ficam com defaults:
   - `dense_vector`, `sparse_vector`, `thesis_vector` → None (preenchidos depois pelo embedding)
   - `enriched_text`, `thesis_text`, `synthetic_questions` → vazio (preenchidos depois pelo enrichment, se habilitado)

#### Modificação em `_phase_vlm_extraction()`

No início de `_phase_vlm_extraction()`, antes de chamar o VLM:

```python
# Tenta usar cache de inspeção aprovada
pdf_sha256 = compute_sha256(pdf_content)
cache_data = self._check_inspection_cache(request.document_id, pdf_sha256)

if cache_data is not None:
    logger.info(f"Cache de inspeção encontrado para {request.document_id}, pulando VLM")
    report_progress("cache_hit", 0.80)
    chunks, canonical_text, canonical_hash = self._cache_to_processed_chunks(
        cache_data, request,
    )
    # Pula direto para embeddings
    ...
else:
    logger.info(f"Sem cache para {request.document_id}, executando pipeline VLM completo")
    # Pipeline VLM normal: PyMuPDF → Qwen3-VL → offset resolution
    ...
```

---

## 4. Validação de Integridade

A peça mais importante é garantir que o cache é válido para o PDF sendo ingerido.

### Regra: SHA256 do PDF fonte

O `metadata.json` da inspeção aprovada contém `sha256_source` (hash do PDF original que foi inspecionado). Na re-ingestão, computamos o SHA256 do PDF sendo enviado e comparamos:

| Cenário | sha256 match? | Ação |
|---------|--------------|------|
| Mesmo PDF, mesma norma | Sim | Usar cache |
| PDF diferente (nova versão da norma) | Não | Pipeline VLM completo |
| Mesmo PDF, document_id diferente | N/A | Cache não encontrado (busca por document_id) |

### Por que não comparar só o document_id?

Se o usuário enviar um PDF corrigido/atualizado com o mesmo document_id (ex: LEI-14133-2021 com erratas), o cache antigo seria inválido. O hash garante que é exatamente o mesmo documento.

---

## 5. Diagrama Completo do Fluxo

```
RunPod (GPU Server)                        VPS (Hostinger)
┌─────────────────────────────────┐       ┌──────────────────────────────────┐
│ pipeline.py                     │       │ inspect.py (router)              │
│                                 │       │                                  │
│ process(pdf, request)           │       │ MinIO (127.0.0.1:9100)           │
│   │                             │       │   vectorgov-evidence/            │
│   ▼                             │       │     inspections/{doc_id}/        │
│ _check_inspection_cache()       │       │       metadata.json              │
│   │                             │       │       canonical.md               │
│   │ GET /inspect/cache/{id}     │       │       offsets.json               │
│   │────────────────────────────→│       │       chunks_preview.json        │
│   │                             │       │       pymupdf_result.json        │
│   │◄────────────────────────────│       │       vlm_result.json            │
│   │  {has_cache, sha256_source} │       │       reconciliation_result.json │
│   │                             │       │       integrity_result.json      │
│   ▼                             │       │       pages/                     │
│ sha256 match?                   │       │                                  │
│   │                             │       │                                  │
│   ├─ SIM ──────────────────┐    │       │                                  │
│   │  GET /inspect/cache/   │    │       │                                  │
│   │    {id}/artifacts      │    │       │                                  │
│   │────────────────────────│───→│       │                                  │
│   │                        │    │       │                                  │
│   │◄───────────────────────│────│       │                                  │
│   │  {chunks, canonical,   │    │       │                                  │
│   │   offsets}             │    │       │                                  │
│   │                        │    │       │                                  │
│   │  _cache_to_processed   │    │       │                                  │
│   │    _chunks()           │    │       │                                  │
│   │         │              │    │       │                                  │
│   │         ▼              │    │       │                                  │
│   │    ProcessedChunks     │    │       │                                  │
│   │    (com offsets reais) │    │       │                                  │
│   │         │              │    │       │                                  │
│   │         ▼              │    │       │                                  │
│   │    embeddings          │    │       │                                  │
│   │         │              │    │       │                                  │
│   │         ▼              │    │       │                                  │
│   │    Milvus              │    │       │                                  │
│   │                        │    │       │                                  │
│   ├─ NÃO ─────────────────┘    │       │                                  │
│   │                             │       │                                  │
│   ▼                             │       │                                  │
│ Pipeline VLM completo           │       │                                  │
│ (PyMuPDF → Qwen3-VL → etc.)    │       │                                  │
│   │                             │       │                                  │
│   ▼                             │       │                                  │
│ embeddings → Milvus             │       │                                  │
└─────────────────────────────────┘       └──────────────────────────────────┘
```

---

## 6. O que cada lado precisa implementar

### 6.1 VPS — 2 endpoints novos

| Endpoint | Método | O que faz |
|----------|--------|-----------|
| `/api/v1/inspect/cache/{document_id}` | GET | Checa se existe inspeção aprovada. Lê `metadata.json` do MinIO, verifica `status == APPROVED`, retorna `has_cache` + `sha256_source` |
| `/api/v1/inspect/cache/{document_id}/artifacts` | GET | Retorna `chunks_preview.json` + `canonical.md` + `offsets.json` combinados num JSON único |

**Auth**: Mesma `X-Ingest-Key` dos endpoints existentes.

**Dependência**: O endpoint `POST /api/v1/inspect/artifacts` (descrito em `VPS_INSTRUCTIONS_INSPECT_ARTIFACTS.md`) precisa estar funcionando primeiro, pois é ele que persiste os artefatos que o cache vai servir.

**Nota sobre o `sha256_source`**: O `InspectionMetadata` já tem este campo? Se não, o RunPod precisa incluí-lo nos metadados da inspeção. Verificar o modelo `InspectionMetadata` em `src/inspection/models.py` do RunPod.

### 6.2 RunPod — 3 métodos novos + 1 modificação

| Arquivo | Mudança |
|---------|---------|
| `src/ingestion/pipeline.py` | Novo: `_check_inspection_cache(document_id, pdf_sha256)` |
| `src/ingestion/pipeline.py` | Novo: `_cache_to_processed_chunks(cache_data, request)` |
| `src/ingestion/pipeline.py` | Modificar: `_phase_vlm_extraction()` — adicionar check no início |
| `src/sinks/` (novo ou existente) | Opcional: `CacheClient` para encapsular os GETs à VPS |

---

## 7. Campos do ChunkPreview que viram ProcessedChunk

O `chunks_preview.json` (gerado pelo pipeline de inspeção, fase 5) contém uma lista de `ChunkPreview`. Na conversão para `ProcessedChunk`:

| Campo ChunkPreview | → Campo ProcessedChunk | Nota |
|--------------------|----------------------|------|
| `span_id` | `span_id` | Direto |
| `node_id` | `node_id` | Direto |
| `device_type` | `device_type` | Direto |
| `text` | `text` | Texto PyMuPDF reconciliado |
| `parent_span_id` | `parent_node_id` | Precisa converter para formato `leis:DOC#SPAN_ID` |
| `canonical_start` | `canonical_start` | Direto |
| `canonical_end` | `canonical_end` | Direto |
| `page_number` | `page_number` | Direto |
| `bbox` | `bbox` | Em PDF points |
| `confidence` | `confidence` | Direto |
| — | `canonical_hash` | Do `offsets.json` ou `metadata.json` |
| — | `document_id` | Do `IngestRequest` |
| — | `tipo_documento` | Do `IngestRequest` |
| — | `numero`, `ano` | Do `IngestRequest` |
| — | `retrieval_text` | Construir: `text` (ou `parent_text + text` para filhos) |
| — | `chunk_level` | `"article"` se article, `"device"` caso contrário |
| — | `chunk_id` | `"{document_id}#{span_id}"` |
| — | `parent_text` | Buscar no `canonical_md` via offsets do pai |

Campos que ficam vazios (preenchidos depois):
- `dense_vector`, `sparse_vector`, `thesis_vector` → embeddings
- `thesis_text`, `thesis_type`, `synthetic_questions` → enrichment LLM
- `enriched_text`, `context_header` → deprecated

---

## 8. Cenários de Teste

| # | Cenário | Resultado Esperado |
|---|---------|-------------------|
| 1 | Ingestão sem inspeção prévia | `_check_inspection_cache` retorna None, pipeline VLM roda normalmente |
| 2 | Ingestão com inspeção aprovada, mesmo PDF | Cache hit, pula VLM, usa chunks do cache |
| 3 | Ingestão com inspeção aprovada, PDF diferente | SHA256 mismatch, pipeline VLM roda normalmente |
| 4 | Ingestão com inspeção NÃO aprovada (COMPLETED mas não APPROVED) | `has_cache: false`, pipeline VLM roda normalmente |
| 5 | VPS offline/timeout | `_check_inspection_cache` retorna None (fallback gracioso), pipeline VLM roda normalmente |
| 6 | Artefatos corrompidos no MinIO | VPS retorna 500, RunPod trata como "sem cache" |

---

## 9. Pré-requisitos

Antes de implementar esta feature:

1. **Pipeline VLM funcionando** — A extração PyMuPDF → Qwen3-VL → offset resolution precisa estar estável
2. **POST /api/v1/inspect/artifacts funcionando na VPS** — Os artefatos de inspeção precisam estar sendo persistidos corretamente no MinIO
3. **Pipeline de inspeção testado** — Pelo menos uma norma inspecionada e aprovada com sucesso
4. **`sha256_source` no InspectionMetadata** — Verificar que o hash do PDF original está sendo salvo nos metadados da inspeção

---

## 10. Notas de Design

1. **Fallback gracioso**: Se qualquer coisa falhar no cache (VPS offline, artefatos corrompidos, sha256 mismatch), o pipeline simplesmente roda o VLM normalmente. O cache é uma **otimização**, nunca um bloqueador.

2. **Não há invalidação explícita do cache**: Se o usuário quer forçar re-inspeção, basta rodar o pipeline de inspeção novamente. A nova aprovação sobrescreve os artefatos no MinIO (idempotência do `put_object`).

3. **Flag para desabilitar cache**: Considerar um campo `skip_cache: bool` no `IngestRequest` para permitir que o usuário force o pipeline VLM completo mesmo com cache disponível.

4. **Logging**: O cache hit/miss deve ser claramente logado para debugging. O `PhaseResult` pode incluir uma fase `"cache_check"` com `message` indicando hit/miss/motivo.
