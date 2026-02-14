# Fix: inspection_snapshot não aparecia na interface da VPS

**Data:** 2026-02-14
**Commit do fix:** `8a1c237`
**Arquivos afetados:** `src/ingestion/router.py`

---

## Sintoma

Após ingestão de um documento, o snapshot de inspeção aparecia corretamente na interface local do RunPod (VectorGov Pipeline Inspector, alimentada pelo Redis DB 2), mas **não aparecia** na interface da VPS (`/admin/inspecao-snapshot`), que é alimentada pela tabela `inspection_snapshots` do PostgreSQL.

## Dois fluxos de dados de inspeção

Existem dois fluxos independentes que enviam dados de inspeção para a VPS:

### Fluxo 1 — VpsInspectionForwarder (fire-and-forget)

```
Pipeline → _emit_regex_inspection_snapshot()
         → VpsInspectionForwarder.forward_full_snapshot()
         → POST /api/v1/inspection/stages        (para cada stage)
         → POST /api/v1/inspection/runs/{id}/complete
         → Tabelas: inspection_runs + inspection_stages
```

Este fluxo **funcionava normalmente**. Os dados chegavam na VPS e eram persistidos nas tabelas `inspection_runs` e `inspection_stages`.

### Fluxo 2 — inspection_snapshot via task.result (polling)

```
Pipeline → result.inspection_snapshot = {...}
         → _set_task_result() salva em task.result
         → VPS faz polling: GET /ingest/result/{task_id}
         → VPS extrai result["inspection_snapshot"]
         → VPS salva na tabela: inspection_snapshots
```

Este fluxo **estava quebrado**. A interface nova da VPS (`/admin/inspecao-snapshot`) lê da tabela `inspection_snapshots`, que estava com 0 rows.

## Causa raiz

No arquivo `src/ingestion/router.py`, a classe `IngestResponse` (Pydantic model usada como `response_model` do endpoint `GET /ingest/result/{task_id}`) **não declarava o campo `inspection_snapshot`**.

O Pydantic v2 por padrão usa `extra = "ignore"`, então ao construir `IngestResponse(**task.result)`, o campo `inspection_snapshot` era **silenciosamente descartado** da resposta JSON.

A VPS recebia o JSON sem o campo → `result.get("inspection_snapshot")` retornava `None` → `save_snapshot()` nunca executava.

### Código antes do fix

```python
class IngestResponse(BaseModel):
    success: bool
    document_id: str
    status: str
    total_chunks: int = 0
    phases: List[dict] = []
    errors: List[dict] = []
    total_time_seconds: float = 0.0
    chunks: List[dict] = []
    document_hash: str = ""
    manifest: dict = {}
    # inspection_snapshot NÃO ESTAVA AQUI
```

### Código após o fix

```python
class IngestResponse(BaseModel):
    success: bool
    document_id: str
    status: str
    total_chunks: int = 0
    phases: List[dict] = []
    errors: List[dict] = []
    total_time_seconds: float = 0.0
    chunks: List[dict] = []
    document_hash: str = ""
    manifest: dict = {}
    inspection_snapshot: Optional[dict] = None  # ← ADICIONADO
```

## Como verificar

1. Fazer uma ingestão de teste
2. Pegar o `task_id` retornado
3. Consultar `GET /ingest/result/{task_id}` e verificar se o campo `inspection_snapshot` está presente no JSON
4. Verificar na VPS se a tabela `inspection_snapshots` tem o registro

## Lição aprendida

Sempre que um novo campo é adicionado ao `task.result` dentro de `_set_task_result()` no `router.py`, o modelo Pydantic `IngestResponse` **também precisa ser atualizado** com o novo campo. Caso contrário, o Pydantic v2 descarta silenciosamente o campo e ele não aparece na resposta da API.

**Regra:** `task.result` e `IngestResponse` devem ter os mesmos campos.
