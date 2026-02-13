# Entrada 1 — PyMuPDF + Regex (Leis, Decretos, INs)

Pipeline de ingestão **sem GPU** para documentos legislativos com texto nativo extraível.
Utilizado como modo padrão (`extraction_mode = "pymupdf_regex"`).

---

## Visão Geral

```
PDF (bytes)
  │
  ▼
PyMuPDFExtractor.extract_pages()          ← texto nativo do PDF
  │  pages_data: List[PageData]
  │  canonical_text: str
  ▼
Idempotency Check                         ← canonical_text == normalize_canonical_text()
  ▼
DriftDetector.check()                     ← detecta mudança de hash para mesmo PDF+versão
  ▼
RegexClassifier.classify_document()       ← blocos → ClassifiedDevice[]
  ▼
_regex_to_processed_chunks()              ← ClassifiedDevice → ProcessedChunk
  ▼
OriginClassifier.classify_document()      ← detecta material externo (Código Penal, etc.)
  ▼
BGE-M3 Embeddings                         ← dense (1024d) + sparse
  ▼
Artifacts Upload (HTTP POST → VPS)        ← PDF + canonical.md + offsets.json
  ▼
validate_chunk_invariants()               ← abort se violação
  ▼
IngestResponse (chunks + manifest)
```

---

## Fases do Pipeline

### Fase 1 — Extração PyMuPDF

**Arquivo:** `src/extraction/pymupdf_extractor.py`
**Classe:** `PyMuPDFExtractor`
**Método:** `extract_pages(pdf_content: bytes) -> Tuple[List[PageData], str]`

1. Abre o PDF com PyMuPDF (fitz)
2. Para cada página:
   - Renderiza imagem PNG a 300 DPI (para inspeção visual)
   - Extrai blocos via `page.get_text("dict")`
   - Normalização inline: NFC + rstrip por linha + trailing `\n`
   - Computa `char_start` / `char_end` durante concatenação (offsets nativos)
3. Retorna `(pages_data, canonical_text)`

**Output — PageData:**
```python
PageData:
    page_number: int          # 1-indexed
    width: float              # largura do PDF em points
    height: float             # altura do PDF em points
    img_width: int            # largura do pixmap (pixels)
    img_height: int           # altura do pixmap (pixels)
    image_base64: str         # PNG em base64
    blocks: List[BlockData]   # blocos com offsets nativos
```

**Output — BlockData:**
```python
BlockData:
    block_index: int
    text: str
    bbox_pdf: list[float]     # [x0, y0, x1, y1] em PDF points (72 DPI)
    char_start: int           # offset global no canonical_text
    char_end: int             # offset global no canonical_text
    lines: list               # linhas com spans (font, size, color)
```

**Invariante crítica:**
```python
canonical_text[block.char_start:block.char_end] == block.text
```

### Fase 2 — Idempotency Check

Verifica que o texto retornado pelo extrator já está normalizado:
```python
assert canonical_text == normalize_canonical_text(canonical_text)
```
Se divergir, **aborta** com `RuntimeError` — offsets seriam inválidos.

### Fase 3 — Drift Detection

**Arquivo:** `src/utils/drift_detector.py`

Compara `canonical_hash` atual com execuções anteriores para o mesmo `(document_id, pdf_hash, pipeline_version)`. Se o hash mudou, registra warning — indica não-determinismo na extração.

### Fase 4 — Classificação Regex

**Arquivo:** `src/extraction/regex_classifier.py`
**Função:** `classify_document(pages) -> ClassificationResult`
**Função:** `classify_to_devices(pages) -> List[ClassifiedDevice]`

Classificação em 3 passes:

| Pass | Descrição |
|------|-----------|
| **Pass 1** | Classifica cada bloco: `article`, `paragraph`, `inciso`, `alinea`, `metadata`, `filtro`, `unclassified` |
| **Pass 2** | Constrói hierarquia via máquina de estados (`current_article`, `current_paragraph`, `current_inciso`) |
| **Pass 3** | Vincula filhos via `parent_span_id` e popula `children_span_ids` |

**Regex patterns:**
| Tipo | Padrão | Exemplo | span_id |
|------|--------|---------|---------|
| Artigo | `Art. \d+[º°]?(-[A-Z]+)?` | Art. 5º, Art. 337-E | `ART-005`, `ART-337-E` |
| Parágrafo | `§ \d+[º°]` ou `Parágrafo único` | § 1º | `PAR-005-1` |
| Inciso | `[IVXL]+ [-–—]` | IV — | `INC-005-4` |
| Alínea | `[a-z]) ` | a) | `ALI-005-1-2-a` |

**Output — ClassifiedDevice:**
```python
ClassifiedDevice:
    device_type: str           # "article", "paragraph", "inciso", "alinea"
    span_id: str               # "ART-005", "PAR-005-1", etc.
    parent_span_id: str        # "" para artigos, "ART-005" para filhos
    children_span_ids: list    # ["PAR-005-1", "PAR-005-2"]
    text: str                  # texto completo do bloco
    identifier: str            # "Art. 5º", "§ 1º", "I", "a"
    article_number: int        # 5 (numérico)
    hierarchy_depth: int       # 0=artigo, 1=§/inciso, 2=inciso sob §, 3=alínea
    char_start: int            # offset no canonical_text
    char_end: int              # offset no canonical_text
    page_number: int           # página (1-indexed)
    bbox: list                 # [x0, y0, x1, y1] PDF points
```

### Fase 5 — Conversão para ProcessedChunk

**Função:** `_regex_to_processed_chunks(devices, canonical_text, canonical_hash, request)`

Para cada `ClassifiedDevice`:
- `node_id` = `"leis:{document_id}#{span_id}"` (ex: `leis:LEI-14133-2021#ART-005`)
- `chunk_id` = `"{document_id}#{span_id}"`
- `parent_node_id` = `"leis:{document_id}#{parent_span_id}"` (vazio se artigo)
- `retrieval_text` = caput do artigo + texto de todos os filhos concatenados
- `canonical_start` / `canonical_end` = offsets nativos do classificador
- `canonical_hash` = SHA256 do canonical_text
- `citations` = referências extraídas via `extract_citations_from_chunk()`

### Fase 6 — OriginClassifier

**Arquivo:** `src/classification/origin_classifier.py`
**Função:** `classify_document(chunks, canonical_text, document_id) -> List[ProcessedChunk]`

Detecta material proveniente de outras normas (ex: artigos do Código Penal inseridos pela Lei 14.133/2021).

**Máquina de estados:**
- **Sinais de entrada (E1-E7):** frases-gatilho ("passa a vigorar com a seguinte redação"), aspas abertas, sequência interrompida, referência a documento externo
- **Sinais de saída (S1-S4):** marcador NR, aspas fechadas, retomada de sequência, nova frase-gatilho

**Campos enriquecidos:**
```python
origin_type: str              # "self" ou "external"
origin_reference: str         # ex: "DL-2848-1940"
origin_reference_name: str    # ex: "Código Penal"
is_external_material: bool    # True se material de outra lei
origin_reason: str            # regra que determinou (ex: "rule:codigo_penal_art337")
```

Quando `is_external_material = True`, o `retrieval_text` é prefixado:
```
Código Penal (DL-2848-1940)
Art. 337-E. ...
```

### Fase 7 — Embeddings (BGE-M3)

Para cada chunk:
- Texto de entrada: `retrieval_text` (ou `text` se vazio)
- `dense_vector`: 1024 dimensões (embedding denso)
- `sparse_vector`: `dict[int, float]` (embedding esparso para keyword matching)

Pulada se `skip_embeddings = True`.

### Fase 8 — Artifacts Upload

**Arquivo:** `src/sinks/artifacts_uploader.py`

Envia para o VPS via **HTTP POST** (RunPod nunca acessa MinIO diretamente):
- PDF original
- `canonical.md` (texto normalizado)
- `offsets.json` (mapa de offsets dos chunks)

### Fase 9 — Validação de Invariantes

**Função:** `validate_chunk_invariants(chunks, document_id)`

| # | Invariante | Falha → |
|---|-----------|---------|
| 1 | `node_id` começa com `"leis:"` ou `"acordaos:"`, sem `"@P"` | `ContractViolationError` |
| 2 | `parent_node_id` (quando não vazio) tem prefixo válido | `ContractViolationError` |
| 3 | Filhos (`paragraph`, `inciso`, `alinea`) devem ter `parent_node_id` | `ContractViolationError` |
| 4 | PR13 trio coerente: sentinela `(-1,-1,"")` **ou** válido `(≥0, >start, hash≠"")` | `ContractViolationError` |
| 5 | Evidence chunks (`article`, `paragraph`, `inciso`, `alinea`) **proibidos** de ter sentinela | `ContractViolationError` |

---

## Endpoint HTTP

```
POST /ingest
Content-Type: multipart/form-data

Campos:
  file: PDF (binary)
  document_id: "LEI-14133-2021"
  tipo_documento: "LEI"          # LEI, DECRETO, IN
  numero: "14133"
  ano: 2021
  extraction_mode: "pymupdf_regex"   # default
  skip_embeddings: false
```

**Response:**
```json
{
  "success": true,
  "document_id": "LEI-14133-2021",
  "status": "completed",
  "chunks": [...],
  "total_chunks": 115,
  "manifest": {
    "total_spans": 115,
    "by_type": {"article": 30, "paragraph": 47, "inciso": 23, "alinea": 15},
    "external_material": {"count": 8, "target_documents": ["DL-2848-1940"]}
  },
  "phases": [
    {"name": "pymupdf_regex_extraction", "duration_seconds": 12.5},
    {"name": "embedding", "duration_seconds": 28.0},
    {"name": "artifacts_upload", "duration_seconds": 3.2}
  ]
}
```

---

## Arquivos-Chave

| Arquivo | Papel |
|---------|-------|
| `src/ingestion/pipeline.py` | Orquestração (`_phase_pymupdf_regex_extraction`) |
| `src/ingestion/models.py` | `IngestRequest`, `ProcessedChunk`, `IngestResponse` |
| `src/ingestion/router.py` | Endpoint FastAPI `POST /ingest` |
| `src/extraction/pymupdf_extractor.py` | Extração de texto + imagens via PyMuPDF |
| `src/extraction/regex_classifier.py` | Classificação regex (3 passes) |
| `src/classification/origin_classifier.py` | Detecção de material externo |
| `src/chunking/canonical_offsets.py` | `normalize_canonical_text()`, `compute_canonical_hash()` |
| `src/chunking/citation_extractor.py` | Extração de citações inter-documentais |
| `src/sinks/artifacts_uploader.py` | Upload de artefatos para VPS |
| `src/utils/drift_detector.py` | Detecção de não-determinismo |
