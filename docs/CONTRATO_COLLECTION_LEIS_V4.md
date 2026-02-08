# Contrato da Collection `leis_v4` — RunPod ↔ VPS

> **Versão**: 4.1.0
> **Data**: 06/02/2026
> **Schema**: `extracao/src/milvus/schema_leis_v4.py`
> **Campos**: 36 (dynamic=OFF)

---

## Visão Geral

Este documento define o **contrato** entre o RunPod (GPU Server) e a VPS (API) para a collection `leis_v4` do Milvus. O RunPod envia chunks via `ProcessedChunk.model_dump()` e a VPS mapeia para os 36 campos do schema.

```
RunPod (GPU)                          VPS (API)                      Milvus
┌──────────────┐   POST /ingest   ┌──────────────────┐  insert   ┌──────────┐
│ProcessedChunk│ ───────────────→ │ ingest_service.py │ ────────→ │ leis_v4  │
│ .model_dump()│   JSON chunks    │ _index_chunks()   │  row dict │ 36 fields│
└──────────────┘                  └──────────────────┘           └──────────┘
```

---

## Mapeamento Completo: ProcessedChunk → Milvus leis_v4

### A) Identidade e Split/Hierarquia

| # | Milvus Field | Tipo Milvus | ProcessedChunk Field | Tipo Python | Default | Notas |
|---|-------------|-------------|---------------------|-------------|---------|-------|
| 1 | `node_id` | VARCHAR(300) PK | `node_id` | str | — | `leis:DOC#SPAN@Pnn` |
| 2 | `logical_node_id` | VARCHAR(300) | `logical_node_id` | str | `""` | `leis:DOC#SPAN` (sem @P) |
| 3 | `span_id` | VARCHAR(100) | `span_id` | str | — | `ART-006`, `PAR-006-1` |
| 4 | `parent_node_id` | VARCHAR(300) | `parent_node_id` | str | `""` | Vazio para artigos |
| 5 | `device_type` | VARCHAR(32) | `device_type` | str | — | `article`, `paragraph`, `inciso`, `alinea` |
| 6 | `chunk_level` | VARCHAR(32) | `chunk_level` | str | `""` | `article`, `device` |
| 7 | `part_index` | INT64 | `part_index` | int | `1` | 1..N para splits |
| 8 | `part_total` | INT64 | `part_total` | int | `1` | 1 se não-split |
| 9 | `chunk_id` | VARCHAR(200) | `chunk_id` | str | `""` | Legado: `DOC#SPAN` |
| 10 | `ingest_run_id` | VARCHAR(100) | `ingest_run_id` | str | `""` | ID da execução |

### B) Texto

| # | Milvus Field | Tipo Milvus | ProcessedChunk Field | Tipo Python | Default | Notas |
|---|-------------|-------------|---------------------|-------------|---------|-------|
| 11 | `text` | VARCHAR(65535) | `text` | str | `""` | Ground truth (original) |
| 12 | `retrieval_text` | VARCHAR(65535) | `retrieval_text` | str | `""` | Para embedding (contexto + perguntas) |

### C) Metadados do Documento

| # | Milvus Field | Tipo Milvus | ProcessedChunk Field | Tipo Python | Default | Notas |
|---|-------------|-------------|---------------------|-------------|---------|-------|
| 13 | `document_id` | VARCHAR(200) | `document_id` | str | — | `LEI-14133-2021` |
| 14 | `tipo_documento` | VARCHAR(64) | `tipo_documento` | str | — | `LEI`, `DECRETO`, `IN`, `PORTARIA` |
| 15 | `numero` | VARCHAR(32) | `numero` | str | — | `14133`, `10.947` |
| 16 | `ano` | INT64 | `ano` | int | — | `2021` |
| 17 | `article_number` | VARCHAR(32) | `article_number` | str | `""` | `6`, `6-A` |
| 18 | `aliases` | VARCHAR(5000) | `aliases` | str | `""` | JSON: `["ETP", "Estudo Técnico"]` |

### D) Offsets Canônicos (PR13 — Evidence Storage)

| # | Milvus Field | Tipo Milvus | ProcessedChunk Field | Tipo Python | Default | Notas |
|---|-------------|-------------|---------------------|-------------|---------|-------|
| 19 | `canonical_start` | INT64 | `canonical_start` | int | `-1` | Offset início (-1 = desconhecido) |
| 20 | `canonical_end` | INT64 | `canonical_end` | int | `-1` | Offset fim (-1 = desconhecido) |
| 21 | `canonical_hash` | VARCHAR(64) | `canonical_hash` | str | `""` | SHA-256 do canonical_text |

### E) Vetores (Hybrid Search — BGE-M3)

| # | Milvus Field | Tipo Milvus | ProcessedChunk Field | Tipo Python | Default | Notas |
|---|-------------|-------------|---------------------|-------------|---------|-------|
| 22 | `dense_vector` | FLOAT_VECTOR(1024) | `dense_vector` | list[float] | — | BGE-M3 dense do `retrieval_text` |
| 23 | `sparse_vector` | SPARSE_FLOAT_VECTOR | `sparse_vector` | dict[int,float] | — | BGE-M3 learned sparse |

### F) Telemetria

| # | Milvus Field | Tipo Milvus | ProcessedChunk Field | Tipo Python | Default | Notas |
|---|-------------|-------------|---------------------|-------------|---------|-------|
| 24 | `has_citations` | BOOL | `has_citations` | bool | `False` | Se tem citações normativas |
| 25 | `citations_count` | INT64 | `citations_count` | int | `0` | Quantidade de citações |

### G) Origem do Material (OriginClassifier)

| # | Milvus Field | Tipo Milvus | ProcessedChunk Field | Tipo Python | Default | Notas |
|---|-------------|-------------|---------------------|-------------|---------|-------|
| 26 | `origin_type` | VARCHAR(16) | `origin_type` | str | `"self"` | `self` ou `external` |
| 27 | `origin_reference` | VARCHAR(128) | `origin_reference` | str | `""` | `DL-2848-1940` |
| 28 | `origin_reference_name` | VARCHAR(128) | `origin_reference_name` | str | `""` | `Código Penal` |
| 29 | `is_external_material` | BOOL | `is_external_material` | bool | `False` | Flag booleana |
| 30 | `origin_confidence` | VARCHAR(8) | `origin_confidence` | str | `""` | `high`, `medium`, `low` |
| 31 | `origin_reason` | VARCHAR(256) | `origin_reason` | str | `""` | `rule:codigo_penal_art337` |

### H) Localização Física no PDF (NOVO v4.1.0)

| # | Milvus Field | Tipo Milvus | ProcessedChunk Field | Tipo Python | Default RunPod | Default Milvus | Conversão VPS |
|---|-------------|-------------|---------------------|-------------|---------------|---------------|--------------|
| 32 | `page_number` | INT64 | `page_number` | int | `-1` | `0` | `max(value, 0)` |
| 33 | `bbox_x0` | FLOAT | `bbox[0]` | list[float] | `[]` | `0.0` | `bbox[0] if bbox else 0.0` |
| 34 | `bbox_y0` | FLOAT | `bbox[1]` | list[float] | `[]` | `0.0` | `bbox[1] if len(bbox)>1 else 0.0` |
| 35 | `bbox_x1` | FLOAT | `bbox[2]` | list[float] | `[]` | `0.0` | `bbox[2] if len(bbox)>2 else 0.0` |
| 36 | `bbox_y1` | FLOAT | `bbox[3]` | list[float] | `[]` | `0.0` | `bbox[3] if len(bbox)>3 else 0.0` |

---

## Conversões Importantes

### page_number: RunPod `-1` → Milvus `0`

O RunPod usa `-1` para "página desconhecida" mas o Milvus schema convencionou `0`.

```python
# VPS ingest_service.py
"page_number": max(chunk.get("page_number", 0), 0)
```

### bbox: RunPod `list[float]` → Milvus 4 campos FLOAT

O RunPod envia um **array único** `[x0, y0, x1, y1]` em coordenadas PDF (72 DPI).
A VPS decompõe em 4 campos escalares para o Milvus.

```python
# RunPod ProcessedChunk
bbox: list[float] = Field(default_factory=list, description="[x0, y0, x1, y1] (72 DPI)")

# VPS ingest_service.py
"bbox_x0": float(chunk.get("bbox", [0.0])[0]) if chunk.get("bbox") else 0.0,
"bbox_y0": float(chunk.get("bbox", [0.0, 0.0])[1]) if len(chunk.get("bbox", [])) > 1 else 0.0,
"bbox_x1": float(chunk.get("bbox", [0.0, 0.0, 0.0])[2]) if len(chunk.get("bbox", [])) > 2 else 0.0,
"bbox_y1": float(chunk.get("bbox", [0.0, 0.0, 0.0, 0.0])[3]) if len(chunk.get("bbox", [])) > 3 else 0.0,
```

### citations: RunPod `list[dict|str]` → VPS Neo4j (NÃO no Milvus v4)

O campo `citations` **não existe mais no schema v4 do Milvus**. Citações são persistidas no Neo4j via sync.
A VPS usa `has_citations` (bool) e `citations_count` (int) para telemetria no Milvus.

---

## Campos Enviados pelo RunPod que NÃO entram no Milvus

Estes campos existem no `ProcessedChunk` mas **não possuem coluna correspondente** no `leis_v4`:

| ProcessedChunk Field | Tipo | Motivo |
|---------------------|------|--------|
| `parent_text` | str | Usado apenas pelo GPU internamente |
| `enriched_text` | str | **DEPRECATED** — substituído por `retrieval_text` |
| `context_header` | str | **DEPRECATED** — removido do schema v4 |
| `thesis_text` | str | Removido do schema v4 |
| `thesis_type` | str | Removido do schema v4 |
| `synthetic_questions` | str | Removido do schema v4 |
| `sparse_source` | str | Removido do schema v4 |
| `citations` | list | Vai para **Neo4j** via sync, não Milvus |
| `schema_version` | str | Removido do schema v4 |
| `bbox_img` | list[float] | Debug only (coordenadas normalizadas 0-1) |
| `img_width` | int | Debug only (resolução pixmap) |
| `img_height` | int | Debug only (resolução pixmap) |
| `confidence` | float | Confiança VLM (não persistida no Milvus) |
| `colegiado` | str | TCU only — vai para `acordaos_v1` |
| `processo` | str | TCU only — vai para `acordaos_v1` |
| `relator` | str | TCU only — vai para `acordaos_v1` |
| `data_sessao` | str | TCU only — vai para `acordaos_v1` |
| `unidade_tecnica` | str | TCU only — vai para `acordaos_v1` |

---

## Índices (9 total)

### Vetoriais (2)

| Campo | Tipo | Métrica | Parâmetros |
|-------|------|---------|------------|
| `dense_vector` | HNSW | COSINE | M=16, efConstruction=256 |
| `sparse_vector` | SPARSE_INVERTED | IP | drop_ratio_build=0.2 |

### Escalares (7)

| Campo | Tipo | Uso |
|-------|------|-----|
| `document_id` | INVERTED | Filtro por documento |
| `tipo_documento` | INVERTED | Filtro por tipo |
| `ano` | INVERTED | Filtro por ano |
| `device_type` | INVERTED | Filtro por tipo de dispositivo |
| `article_number` | INVERTED | Filtro por artigo |
| `logical_node_id` | INVERTED | Join Neo4j ↔ Milvus |
| `is_external_material` | INVERTED | Filtro origem |

---

## Validação (Checklist RunPod)

Antes de enviar chunks, o RunPod deve garantir:

- [ ] `node_id` não vazio e no formato `leis:DOC#SPAN@Pnn`
- [ ] `dense_vector` tem exatamente 1024 dimensões
- [ ] `sparse_vector` é dict com chaves int e valores float
- [ ] `text` não vazio (ground truth)
- [ ] `retrieval_text` não vazio (embedding source)
- [ ] `document_id` no formato `DOC-NUMERO-ANO`
- [ ] `page_number >= -1` (-1 = desconhecido, convertido para 0 na VPS)
- [ ] `bbox` é lista de 4 floats `[x0, y0, x1, y1]` ou lista vazia
- [ ] Se `bbox` não vazio: coordenadas em PDF points (72 DPI)
- [ ] `part_index >= 1` e `part_total >= 1`
- [ ] `part_index <= part_total`

---

## Exemplo de Chunk Completo

```json
{
  "node_id": "leis:LEI-14133-2021#ART-006@P01",
  "logical_node_id": "leis:LEI-14133-2021#ART-006",
  "span_id": "ART-006",
  "parent_node_id": "",
  "device_type": "article",
  "chunk_level": "article",
  "part_index": 1,
  "part_total": 1,
  "chunk_id": "LEI-14133-2021#ART-006",
  "ingest_run_id": "run-20260206-abc123",

  "text": "Art. 6º Para os fins desta Lei, consideram-se...",
  "retrieval_text": "[CONTEXTO: Este artigo da Lei 14.133/2021 define...] Art. 6º...",

  "document_id": "LEI-14133-2021",
  "tipo_documento": "LEI",
  "numero": "14133",
  "ano": 2021,
  "article_number": "6",
  "aliases": "[\"ETP\", \"Estudo Técnico Preliminar\"]",

  "canonical_start": 1542,
  "canonical_end": 2890,
  "canonical_hash": "a1b2c3d4...",

  "dense_vector": [0.023, -0.015, ...],
  "sparse_vector": {"1234": 0.5, "5678": 0.3, ...},

  "has_citations": true,
  "citations_count": 3,

  "origin_type": "self",
  "origin_reference": "",
  "origin_reference_name": "",
  "is_external_material": false,
  "origin_confidence": "",
  "origin_reason": "",

  "page_number": 2,
  "bbox": [72.0, 150.5, 540.0, 210.3]
}
```

**Resultado no Milvus** (campos H):

| Campo | Valor |
|-------|-------|
| `page_number` | `2` |
| `bbox_x0` | `72.0` |
| `bbox_y0` | `150.5` |
| `bbox_x1` | `540.0` |
| `bbox_y1` | `210.3` |

---

## Changelog

| Versão | Data | Mudança |
|--------|------|---------|
| 4.0.0 | 22/01/2025 | Schema v4 definitivo (31 campos) |
| 4.1.0 | 06/02/2026 | +5 campos: page_number + bbox (36 campos) |
