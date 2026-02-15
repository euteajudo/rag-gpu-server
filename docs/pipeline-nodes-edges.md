# Pipeline de Nodes e Edges — Ingestão de Leis

**Última atualização:** 2026-02-15

Este documento descreve como os nodes (nós) e edges (arestas/citações) são criados durante o pipeline de ingestão de leis.

---

## 1. Nodes — Dispositivos legais

Cada dispositivo classificado pelo regex (artigo, parágrafo, inciso, alínea) vira um **node**. O `node_id` é construído na fase `_regex_to_processed_chunks()` em `src/ingestion/pipeline.py`:

```
chunk_id = "{document_id}#{span_id}"     → LEI-14.133-2021#ART-005
node_id  = "leis:{chunk_id}"             → leis:LEI-14.133-2021#ART-005
```

### Hierarquia pai-filho

Definida pelo campo `parent_node_id`:

| Tipo | parent_node_id | Exemplo |
|------|----------------|---------|
| Artigo | `""` (raiz) | — |
| Parágrafo | node_id do artigo | `leis:LEI-14.133-2021#ART-005` |
| Inciso | node_id do artigo ou parágrafo | `leis:LEI-14.133-2021#ART-005` |
| Alínea | node_id do inciso | `leis:LEI-14.133-2021#INC-005-III` |

### Formato do span_id (após fix commit `8dec004`)

- Artigos: `ART-005`, `ART-075`
- Parágrafos: `PAR-005-1`, `PAR-005-UNICO`
- Incisos: `INC-005-III` (numeral romano, não arábico)
- Alíneas: `ALI-005-III-a`

### node_id físico vs lógico

No Milvus, o chunk pode ser dividido em partes quando o texto é muito longo:

```
node_id (físico):  leis:LEI-14.133-2021#ART-005@P00   (com sufixo de parte)
logical_node_id:   leis:LEI-14.133-2021#ART-005        (sem sufixo, para joins)
```

---

## 2. Edges — Citações entre dispositivos

### Extração de citações

Na fase `_regex_to_processed_chunks()`, para cada dispositivo, o `CitationExtractor` (`src/chunking/citation_extractor.py`) analisa o texto e extrai referências normativas:

```python
citations = extract_citations_from_chunk(
    text=device.text,
    document_id=request.document_id,
    chunk_node_id=node_id,
    parent_chunk_id=parent_node_id,
    document_type=request.tipo_documento,
)
```

### O que o CitationExtractor faz

1. **Referências externas** — regex detecta menções a outras normas:
   - "Lei nº 14.133/2021" → `leis:LEI-14.133-2021`
   - "art. 9º da IN 58/2022" → `leis:IN-58-2022#ART-009`
   - "Acórdão 2450/2025" → `acordaos:AC-2450-2025`

2. **Referências internas** — regex detecta dispositivos do mesmo documento:
   - "art. 75, inciso II" → `leis:LEI-14.133-2021#INC-075-II`
   - "art. 5º, alínea 'a'" → `leis:LEI-14.133-2021#ALI-005-a`

3. **Normalização** — remove:
   - Self-loops (dispositivo citando a si mesmo)
   - Parent-loops (filho citando o pai direto)
   - Duplicatas

4. **Classificação de rel_type** (PR5) — classifica o tipo de relação contextual via `rel_type_classifier.py`:
   - `CITA` (referência genérica)
   - `ALTERA_EXPRESSAMENTE` (alteração explícita)
   - etc.

### Formato da citação

Cada citação é armazenada como dict no campo `citations` do `ProcessedChunk`:

```json
{
    "target_node_id": "leis:LEI-14.133-2021#ART-075",
    "rel_type": "CITA",
    "rel_type_confidence": 0.85
}
```

### Resolução de doc_id

O CitationExtractor usa três recursos para resolver referências:

- **Tabela `CANONICAL_NORMS`** — mapeamento `(tipo, número) → ano` para ~40 normas conhecidas (Lei 14.133→2021, Decreto 10.947→2022, etc.)
- **Catálogo `KNOWN_DOCUMENTS`** — nomes por extenso → doc_id ("lei de licitações" → `LEI-14133-2021`)
- **`normalize_document_id()`** — aplica ponto de milhar em números >= 1000 (14133 → 14.133)

### Cálculo de confiança

| Situação | Confiança | Ambíguo? |
|----------|-----------|----------|
| Número + ano válido | 0.95 | Não |
| Número sem ano | 0.60 | Sim |
| Sem número | 0.30 | Sim |
| Constituição Federal | 0.95 | Não |
| Referência interna com document_id | 0.90 | Não |
| Referência interna sem document_id | 0.50 | Sim |

---

## 3. Persistência

### Milvus (vetor store) — ativo

Os chunks são inseridos no Milvus na fase `_phase_milvus_sink()` com os campos:

| Campo | Descrição | Exemplo |
|-------|-----------|---------|
| `node_id` | PK física (com @Pxx) | `leis:LEI-14.133-2021#ART-005@P00` |
| `logical_node_id` | ID lógico (sem @Pxx) | `leis:LEI-14.133-2021#ART-005` |
| `span_id` | ID do span | `ART-005` |
| `parent_node_id` | ID do pai | `leis:LEI-14.133-2021#ART-005` |
| `device_type` | Tipo do dispositivo | `article`, `paragraph`, `inciso`, `alinea` |
| `has_citations` | Tem citações? | `true` / `false` |
| `citations_count` | Quantidade de citações | `3` |

As citações completas (com `target_node_id`, `rel_type`, `rel_type_confidence`) ficam no campo `citations` do `ProcessedChunk` e são retornadas no `task.result`.

### Neo4j (grafo) — implementado, não integrado

O `Neo4jEdgeWriter` (`src/sinks/neo4j_writer.py`) está implementado mas **não é chamado pelo pipeline atual**. Ele suporta:

- **`upsert_node()`** — `MERGE (:LegalNode {node_id})` com propriedades (document_id, span_id, device_type, text_preview)
- **`create_edge()`** — `MERGE (source)-[:CITA]->(target)` com confidence, confidence_tier, extraction_method, citation_text
- **`create_edges_batch()`** — `UNWIND` para inserção em lote
- Previne self-loops automaticamente
- Usa `MERGE` para idempotência (re-ingestão não duplica)

O modelo `EdgeCandidate` inclui:
- `confidence_tier`: HIGH (>=0.8), MEDIUM (0.5-0.8), LOW (<0.5)
- `extraction_method`: REGEX, HEURISTIC, NLI, MANUAL
- Campos de custódia: `document_id`, `document_version`, `pipeline_version`, `ingest_run_id`

---

## 4. Gap de persistência das citações

### O problema

O pipeline do RunPod faz todo o trabalho pesado de extrair citações (target_node_id, rel_type, confiança), mas na hora de persistir no Milvus (`_phase_milvus_sink()`), **apenas metadados superficiais são gravados**:

```python
"has_citations": has_citations,    # True/False
"citations_count": citations_count, # número inteiro
```

Os **target_node_ids reais** — para onde cada citação aponta, o `rel_type` e a confiança — **não são gravados no Milvus**.

### Fluxo real confirmado (RunPod + VPS)

#### Durante a ingestão (funciona)

```
RunPod (GPU Server)                          VPS
─────────────────                            ───
CitationExtractor extrai citations
        │
        ▼
ProcessedChunk.citations = [                 ingest_service.py recebe chunks
  {target_node_id, rel_type, confidence}     com citations EM MEMÓRIA
]                                                    │
        │                                            ▼
        └──── POST chunks ────────────────►  sync_chunks() cria edges no Neo4j
                                             usando citations da memória ✓
```

As citações viajam **em memória** do RunPod → VPS dentro dos chunks do `task.result`. O `ingest_service.py` da VPS chama `sync_chunks()` que cria os edges `:CITA` no Neo4j diretamente a partir dessas citações.

#### Durante reconcile (NÃO funciona)

```
VPS
───
reconcile() → query Milvus → campo "citations" NÃO EXISTE no schema
           → row.get("citations", "") → "" → 0 edges ✗
```

Quando se usa `reconcile()` para reconstruir o Neo4j a partir do Milvus (ex: após limpeza do grafo), **todas as relações `:CITA` se perdem** porque o Milvus não armazena os target_node_ids — apenas `has_citations` (bool) e `citations_count` (int).

### Consequências

| Cenário | Citações | Status |
|---------|----------|--------|
| Ingestão nova (fluxo quente) | Chegam via memória no `task.result` → `sync_chunks()` cria edges | **Funciona** |
| Reconcile/rebuild do Neo4j | Query Milvus → campo citations não existe → 0 edges | **Não funciona** |
| Restart do GPU server | `task.result` é dict in-memory, perde-se | **Perde dados** |
| Re-ingestão do mesmo documento | CitationExtractor re-extrai → edges recriados | **Funciona** |

### Neo4jEdgeWriter do RunPod — código morto

O `Neo4jEdgeWriter` (`src/sinks/neo4j_writer.py`) está implementado mas **nunca é chamado** pelo pipeline. A criação de nodes e edges no Neo4j é responsabilidade exclusiva do **sync_service da VPS** (`extracao/src/graph/sync_service.py`).

O writer suporta:
- **`upsert_node()`** — `MERGE (:LegalNode {node_id})` com propriedades
- **`create_edge()`** — `MERGE (source)-[:CITA]->(target)` com confidence, confidence_tier, extraction_method
- **`create_edges_batch()`** — `UNWIND` para inserção em lote
- Prevenção de self-loops, idempotência via `MERGE`

O modelo `EdgeCandidate` inclui:
- `confidence_tier`: HIGH (>=0.8), MEDIUM (0.5-0.8), LOW (<0.5)
- `extraction_method`: REGEX, HEURISTIC, NLI, MANUAL
- Campos de custódia: `document_id`, `document_version`, `pipeline_version`, `ingest_run_id`

### Possíveis soluções para o gap

**Opção A — Persistir citações no Milvus:** Adicionar campo VarChar JSON com os target_node_ids completos ao schema do Milvus (`leis_v4`). Permitiria `reconcile()` reconstruir edges sem re-extração.

**Opção B — Re-extrair no reconcile:** Modificar `reconcile()` na VPS para rodar o `CitationExtractor` sobre o texto de cada chunk lido do Milvus, replicando o que o GPU server faz durante a ingestão.

**Opção C — Integrar Neo4jEdgeWriter no pipeline:** Adicionar fase `_phase_neo4j_sink()` no pipeline do RunPod que escreva diretamente no Neo4j. Eliminaria a dependência do fluxo em memória, mas requer que o RunPod tenha acesso de rede ao Neo4j da VPS.

---

## 5. Fluxo completo

```
PDF
 │
 ▼
PyMuPDF (extração de texto nativo)
 │
 ▼
Regex Classifier (classifica dispositivos: ART, PAR, INC, ALI)
 │
 ▼
_regex_to_processed_chunks()
 │  Para cada dispositivo:
 │   ├─ Constrói node_id:        "leis:{doc_id}#{span_id}"
 │   ├─ Constrói parent_node_id: "leis:{doc_id}#{parent_span_id}"
 │   ├─ CitationExtractor.extract(texto)
 │   │    ├─ Regex: referências externas (Lei X, IN Y, Decreto Z)
 │   │    ├─ Regex: referências internas (art. X deste documento)
 │   │    ├─ Normaliza: remove self-loops, parent-loops, duplicatas
 │   │    └─ Classifica rel_type (CITA, ALTERA, etc.)
 │   └─ Monta ProcessedChunk com node_id + parent_node_id + citations
 │
 ▼
OriginClassifier (classifica self vs external)
 │
 ▼
Embeddings (BGE-M3: dense 1024d + sparse)
 │
 ▼
Milvus Sink (persiste node_id, logical_node_id, parent_node_id, has_citations)
 │
 ▼
Artifacts Upload (PDF + canonical_text + offsets → VPS/MinIO)
 │
 ▼
Inspection Snapshot (Redis local + VPS PostgreSQL)
 │
 ▼
Manifest (contabiliza node_count, edge_count)
```

---

## 6. Arquivos-chave

| Arquivo | Responsabilidade |
|---------|------------------|
| `src/extraction/regex_classifier.py` | Classifica dispositivos, gera span_id e hierarquia |
| `src/ingestion/pipeline.py` | Orquestra fases, constrói node_id/parent_node_id |
| `src/chunking/citation_extractor.py` | Extrai citações (edges) do texto de cada dispositivo |
| `src/chunking/rel_type_classifier.py` | Classifica tipo de relação (CITA, ALTERA, etc.) |
| `src/ingestion/models.py` | Define ProcessedChunk com campos node_id, citations |
| `src/sinks/neo4j_writer.py` | Writer para Neo4j (implementado, não integrado) |
| `src/sinks/milvus_writer.py` | Writer para Milvus (sink ativo) |
| `src/manifest/manifest_builder.py` | Contabiliza node_count, edge_count no manifest |
| `src/utils/normalization.py` | Normaliza document_id (ponto de milhar, etc.) |
