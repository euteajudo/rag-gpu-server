# Entrada Única — Acórdãos do TCU

Pipeline de ingestão para acórdãos do Tribunal de Contas da União (TCU).
Ativado quando `tipo_documento = "ACORDAO"`. Suporta ambos os modos de extração de texto (`pymupdf_regex` e `vlm`).

---

## Visão Geral

Diferente das leis (que possuem duas entradas distintas), os acórdãos seguem uma **entrada única** com dois modos de extração de texto intercambiáveis. A lógica de parsing, chunking e indexação é sempre a mesma — apenas a fonte do texto muda.

```
PDF (bytes)
  │
  ├─ extraction_mode = "pymupdf_regex"
  │    └── PyMuPDFExtractor.extract_pages()        ← texto nativo
  │
  └─ extraction_mode = "vlm"
       └── VLMService.ocr_document()               ← OCR via Qwen3-VL
  │
  ▼
_process_acordao()                    ← lógica compartilhada (ambos os modos)
  │
  ├── Idempotency Check
  ├── DriftDetector.check()
  ├── AcordaoHeaderParser.parse_header()     ← metadados do cabeçalho
  ├── AcordaoParser.parse()                  ← seções, parágrafos, itens
  ├── build_sections()                       ← AcordaoDevice → ParsedSection
  ├── AcordaoChunker.chunk()                 ← seções → chunks com overlap
  ├── _acordao_to_processed_chunks()         ← AcordaoChunk → ProcessedChunk
  ├── BGE-M3 Embeddings
  ├── Artifacts Upload (HTTP POST → VPS)
  └── validate_chunk_invariants()
  │
  ▼
IngestResponse (chunks + manifest)
```

---

## Diferenças Fundamentais: Acórdão vs. Lei

| Aspecto | Leis (Entradas 1 e 2) | Acórdãos |
|---------|----------------------|----------|
| **Classificador** | `RegexClassifier` (dispositivos legais) | `AcordaoParser` (seções textuais) |
| **Unidade de chunk** | Dispositivo individual (artigo, §, inciso, alínea) | Seção com overlap (EMENTA, RELATÓRIO, VOTO, ACÓRDÃO) |
| **Hierarquia** | Artigo → § → Inciso → Alínea | Seção primária → Parágrafos/Itens |
| **Prefixo node_id** | `"leis:"` | `"acordaos:"` |
| **Tipos de dispositivo** | `article`, `paragraph`, `inciso`, `alinea` | `section`, `paragraph`, `item_dispositivo` |
| **OriginClassifier** | Sim (detecta material externo) | Não (todo material é "self") |
| **Metadados extras** | `article_number`, `citations` | `colegiado`, `processo`, `relator`, `data_sessao`, `section_type`, `authority_level` |
| **Chunks por documento** | 50-150 (1:1 com hierarquia) | 15-25 (seções com overlap de 20%) |
| **Estratégia de chunking** | Determinístico (1 chunk = 1 dispositivo) | Overlap por parágrafos (max ~4000 chars/chunk) |

---

## Fases do Pipeline

### Fase 1 — Extração de Texto

Dois caminhos possíveis, controlados por `extraction_mode`:

**Modo PyMuPDF** (`_phase_acordao_extraction`):
```python
extractor = PyMuPDFExtractor(dpi=300)
pages_data, raw_canonical = extractor.extract_pages(pdf_content)
```

**Modo VLM** (`_phase_acordao_vlm_extraction`):
```python
pages_data, raw_canonical = asyncio.run(
    self.vlm_service.ocr_document(pdf_bytes=pdf_content, ...)
)
```

Ambos os modos retornam `(List[PageData], str)` — a mesma interface. A partir daqui, tudo converge em `_process_acordao()`.

### Fase 2 — Idempotency Check + Drift Detection

Idêntico às Entradas 1 e 2 de leis. Verifica normalização e detecta mudanças de hash.

### Fase 3 — Extração de Metadados do Cabeçalho

**Arquivo:** `src/extraction/acordao_header_parser.py`
**Classe:** `AcordaoHeaderParser`
**Método:** `parse_header(text: str) -> dict`

Extrai metadados estruturados do cabeçalho do acórdão usando regex:

| Campo | Regex | Exemplo |
|-------|-------|---------|
| `numero` | `ACÓRDÃO Nº (\d+)/(\d{4})` | "1234" |
| `ano` | (mesmo regex) | "2023" |
| `colegiado` | `Plenário\|1ª Câmara\|2ª Câmara` | "Plenario" |
| `processo` | `TC (\d{3}\.\d{3}/\d{4}-\d)` | "TC 045.123/2022-5" |
| `natureza` | `Natureza: (.+)` | "Representação" |
| `relator` | `Relator: (Ministro\s+)?(.+)` | "Bruno Dantas" |
| `data_sessao` | `Data da Sessão: (\d{1,2}/\d{1,2}/\d{4})` | "15/03/2023" |
| `unidade_tecnica` | `Unidade Técnica: (.+)` | "SecexDefesa" |
| `sumario` | `SUMÁRIO: (.+?)(?=RELATÓRIO)` | "Representação sobre..." |
| `resultado` | `considerar ... (procedente\|improcedente)` | "procedente" |

**Normalização do colegiado:**
- "Plenário" → `"Plenario"`
- "1ª Câmara" / "Primeira Câmara" → `"1a_Camara"`
- "2ª Câmara" / "Segunda Câmara" → `"2a_Camara"`

### Fase 4 — Parsing Estrutural (AcordaoParser)

**Arquivo:** `src/extraction/acordao_parser.py`
**Classe:** `AcordaoParser`
**Método:** `parse(canonical_text: str, page_boundaries: list) -> List[AcordaoDevice]`

Identifica a estrutura hierárquica do acórdão:

#### Seções Primárias (hierarquia nível 0)
| Seção | Regex | span_id |
|-------|-------|---------|
| RELATÓRIO | `^\s*RELAT[OÓ]RIO\s*$` | `SEC-RELATORIO` |
| VOTO | `^\s*VOTO\s*$` | `SEC-VOTO` |
| ACÓRDÃO | `^\s*AC[OÓ]RD[AÃ]O\s+N[°º]?\s*\d+` | `SEC-ACORDAO` |

#### Subseções do RELATÓRIO (hierarquia nível 1+)
| Tipo | Regex | Exemplo | span_id |
|------|-------|---------|---------|
| Numeral romano | `^([IVX]+)\.\s+(.+)` | "I. EXAME TÉCNICO" | Dinâmico |
| Sub-numeração | `^([IVX]+\.\d+(?:\.\d+)*)\.\s+(.+)` | "I.3.1. Análise" | Dinâmico |
| Título uppercase | `^([A-Z]{4,})$` | "EXAME TÉCNICO" | Dinâmico |

#### Parágrafos Numerados
| Contexto | Regex | Exemplo | span_id |
|----------|-------|---------|---------|
| RELATÓRIO | `^(\d{1,3})\.\s+` | "25. O relatório indica..." | `PAR-RELATORIO-25` |
| VOTO | `^(\d{1,3})\.\s+` | "13. Concordo com..." | `PAR-VOTO-13` |

#### Itens do Dispositivo (parte decisória do ACÓRDÃO)
| Regex | Exemplo | span_id |
|-------|---------|---------|
| `^(\d+\.\d+(?:\.\d+)*)\.\s+` | "9.1. Dar ciência..." | `ITEM-9.1` |
| | "9.4.1. Aplicar multa..." | `ITEM-9.4.1` |

**Output — AcordaoDevice:**
```python
AcordaoDevice:
    device_type: str           # "section", "paragraph", "item_dispositivo"
    span_id: str               # "SEC-RELATORIO", "PAR-VOTO-7", "ITEM-9.4.1"
    parent_span_id: str        # "" para seções primárias
    children_span_ids: list
    text: str                  # texto completo
    identifier: str            # "RELATÓRIO", "7", "9.4.1"
    section_type: str          # "relatorio", "voto", "acordao"
    authority_level: str       # "opinativo", "fundamentacao", "vinculante"
    section_path: str          # "RELATÓRIO > EXAME TÉCNICO > I.3.1"
    hierarchy_depth: int       # 0=primária, 1=subseção, 2=sub-subseção
    char_start: int            # offset no canonical_text
    char_end: int              # offset no canonical_text
    page_number: int           # 1-indexed
```

Tipicamente um acórdão de 30-50 páginas gera ~100 dispositivos.

### Fase 5 — Construção de Seções (build_sections)

**Arquivo:** `src/extraction/acordao_chunker.py`
**Função:** `build_sections(devices, canonical_text, header_metadata) -> List[ParsedSection]`

Converte a lista detalhada de `AcordaoDevice` em 4 seções consolidadas:

| # | Seção | Fonte | Parágrafos filhos |
|---|-------|-------|-------------------|
| 1 | **EMENTA** | SUMÁRIO no header (regex no canonical_text) | Nenhum |
| 2 | **RELATÓRIO** | `SEC-RELATORIO` device | `PAR-RELATORIO-*` |
| 3 | **VOTO** | `SEC-VOTO` device | `PAR-VOTO-*` |
| 4 | **ACÓRDÃO** | `SEC-ACORDAO` device | `ITEM-*` |

**Output — ParsedSection:**
```python
ParsedSection:
    section_type: str          # "ementa", "relatorio", "voto", "acordao"
    text: str                  # canonical_text[canonical_start:canonical_end]
    canonical_start: int
    canonical_end: int
    paragraphs: list           # [{"num": str, "text": str, "start": int, "end": int}]
```

### Fase 6 — Chunking com Overlap (AcordaoChunker)

**Arquivo:** `src/extraction/acordao_chunker.py`
**Classe:** `AcordaoChunker`
**Método:** `chunk(sections, document_id, canonical_hash, metadata) -> List[AcordaoChunk]`

Divide cada seção em chunks respeitando fronteiras de parágrafo, com 20% de overlap entre partes consecutivas.

**Parâmetros:**
| Parâmetro | Valor padrão | Descrição |
|-----------|-------------|-----------|
| `max_chunk_chars` | 4000 | Tamanho máximo de cada chunk |
| `overlap_ratio` | 0.20 | 20% do chunk anterior repetido no próximo |
| `min_overlap_chars` | 200 | Overlap mínimo em caracteres |
| `max_overlap_chars` | 1200 | Overlap máximo em caracteres |

**Lógica de split:**
1. Se a seção cabe em `max_chunk_chars` → 1 chunk único
2. Senão, divide por fronteiras de parágrafos
3. Cada novo chunk inicia com ~20% do chunk anterior (overlap)

**Nomenclatura de span_id:**
| Caso | span_id |
|------|---------|
| Seção inteira (1 chunk) | `SEC-EMENTA`, `SEC-VOTO`, `SEC-RELATORIO`, `SEC-ACORDAO` |
| Seção particionada | `SEC-VOTO-P01`, `SEC-VOTO-P02`, `SEC-RELATORIO-P03` |

**Níveis de autoridade:**
| Seção | `authority_level` | Significado |
|-------|------------------|-------------|
| EMENTA | `metadado` | Resumo informativo |
| RELATÓRIO | `opinativo` | Descrição factual, sem força vinculante |
| VOTO | `fundamentacao` | Fundamentação jurídica do relator |
| ACÓRDÃO | `vinculante` | Decisão final com força de precedente |

**retrieval_text:**
Cada chunk recebe um prefixo de contexto para melhorar a qualidade do embedding:
```
[CONTEXTO: VOTO do Acórdão 1234/2023 - Plenário, Rel. Min. Bruno Dantas, Parte 2/3]
13. Concordo com o parecer da unidade técnica...
```

**Output — AcordaoChunk:**
```python
AcordaoChunk:
    span_id: str               # "SEC-VOTO-P01"
    section_type: str          # "voto"
    authority_level: str       # "fundamentacao"
    text: str                  # texto do chunk
    retrieval_text: str        # [CONTEXTO: ...] + text
    canonical_start: int       # offset absoluto
    canonical_end: int         # offset absoluto
    page_number: int           # estimativa (~3500 chars/página)
    part_number: int           # 1-indexed
    total_parts: int           # total de partes da seção
    section_path: str          # "VOTO"
```

Tipicamente um acórdão gera **15-25 chunks**.

### Fase 7 — Conversão para ProcessedChunk

**Função:** `_acordao_to_processed_chunks(acordao_chunks, canonical_text, canonical_hash, request, header_metadata)`

Converte cada `AcordaoChunk` em `ProcessedChunk` com:

- **node_id:** `"acordaos:{document_id}#{span_id}"` (ex: `acordaos:ACORDAO-1234-2023#SEC-VOTO-P02`)
- **device_type:** sempre `"section"`
- **chunk_level:** sempre `"section"`
- **parent_node_id:** sempre vazio (seções não possuem hierarquia pai-filho no Milvus)
- **Metadados do cabeçalho:** `colegiado`, `processo`, `relator`, `data_sessao`
- **Metadados de seção:** `section_type`, `authority_level`, `section_path`
- **Offsets:** `canonical_start`, `canonical_end`, `canonical_hash`
- **Campos de lei vazios:** `article_number = ""`, `citations = []`, todos os campos `origin_*` mantêm defaults

### Fase 8 — Embeddings (BGE-M3)

Idêntico às leis. Para cada chunk:
- `dense_vector`: 1024 dimensões
- `sparse_vector`: embedding esparso

O texto usado para embedding é o `retrieval_text` (com prefixo de contexto), o que melhora a busca semântica ao incluir informação sobre o tipo de seção e metadados do acórdão.

### Fase 9 — Artifacts Upload + Validação

**Artifacts:** Mesmo padrão — PDF + canonical.md + offsets.json via HTTP POST ao VPS.

**Validação de invariantes:** Mesma função `validate_chunk_invariants()`. Para acórdãos:
- `node_id` deve começar com `"acordaos:"`
- Evidence types = `{"section", "paragraph", "item_dispositivo"}`
- Offsets válidos obrigatórios (sentinela proibida)

---

## OriginClassifier — Não se Aplica

Acórdãos **não passam** pelo OriginClassifier. O campo `origin_type` mantém o valor default `"self"` para todos os chunks.

**Motivo:** Acórdãos citam outros acórdãos e normas extensivamente no VOTO e RELATÓRIO, mas essa citação é parte natural do texto — não é "material externo inserido" como acontece nas leis (que transcrevem artigos de outros diplomas). A detecção de citações em acórdãos será tratada por outro mecanismo no futuro.

---

## Endpoint HTTP

```
POST /ingest
Content-Type: multipart/form-data

Campos:
  file: PDF (binary)
  document_id: "ACORDAO-1234-2023"
  tipo_documento: "ACORDAO"          # ← ativa pipeline de acórdãos
  numero: "1234"
  ano: 2023
  extraction_mode: "pymupdf_regex"   # ou "vlm"
  skip_embeddings: false

  # Campos opcionais (preenchidos automaticamente pelo header parser se ausentes):
  colegiado: "P"                     # P (Plenário), 1C, 2C
  processo: "TC 045.123/2022-5"
  relator: "Bruno Dantas"
  data_sessao: "15/03/2023"
```

**Response:**
```json
{
  "success": true,
  "document_id": "ACORDAO-1234-2023",
  "status": "completed",
  "chunks": [...],
  "total_chunks": 18,
  "manifest": {
    "total_spans": 18,
    "by_type": {"section": 18},
    "acordao_metadata": {
      "numero": "1234",
      "ano": "2023",
      "colegiado": "Plenario",
      "processo": "TC 045.123/2022-5",
      "relator": "Bruno Dantas"
    }
  },
  "phases": [
    {"name": "acordao_extraction", "method": "pymupdf_native", ...},
    {"name": "embedding", ...},
    {"name": "artifacts_upload", ...}
  ]
}
```

---

## Exemplo Concreto de Estrutura Parseada

Para um acórdão típico do TCU Plenário (~40 páginas):

```
AcordaoParser: ~103 dispositivos
  ├── SEC-RELATORIO (section, hierarquia 0)
  │   ├── PAR-RELATORIO-1 (paragraph)
  │   ├── PAR-RELATORIO-2 (paragraph)
  │   ├── ...
  │   └── PAR-RELATORIO-35 (paragraph)
  │
  ├── SEC-VOTO (section, hierarquia 0)
  │   ├── PAR-VOTO-1 (paragraph)
  │   ├── PAR-VOTO-2 (paragraph)
  │   ├── ...
  │   └── PAR-VOTO-28 (paragraph)
  │
  └── SEC-ACORDAO (section, hierarquia 0)
      ├── ITEM-9.1 (item_dispositivo)
      ├── ITEM-9.2 (item_dispositivo)
      ├── ITEM-9.3 (item_dispositivo)
      ├── ITEM-9.4 (item_dispositivo)
      └── ITEM-9.4.1 (item_dispositivo)

build_sections: 4 seções (ementa, relatorio, voto, acordao)

AcordaoChunker: ~18 chunks
  ├── SEC-EMENTA          (metadado, 1 parte)
  ├── SEC-RELATORIO-P01   (opinativo, parte 1/4)
  ├── SEC-RELATORIO-P02   (opinativo, parte 2/4)
  ├── SEC-RELATORIO-P03   (opinativo, parte 3/4)
  ├── SEC-RELATORIO-P04   (opinativo, parte 4/4)
  ├── SEC-VOTO-P01        (fundamentacao, parte 1/3)
  ├── SEC-VOTO-P02        (fundamentacao, parte 2/3)
  ├── SEC-VOTO-P03        (fundamentacao, parte 3/3)
  └── SEC-ACORDAO         (vinculante, 1 parte)
```

---

## Arquivos-Chave

| Arquivo | Papel |
|---------|-------|
| `src/ingestion/pipeline.py` | Orquestração (`_phase_acordao_extraction`, `_phase_acordao_vlm_extraction`, `_process_acordao`) |
| `src/extraction/acordao_header_parser.py` | Extração de metadados do cabeçalho |
| `src/extraction/acordao_parser.py` | Parsing estrutural (seções, parágrafos, itens) |
| `src/extraction/acordao_chunker.py` | `build_sections()`, `AcordaoChunker`, overlap chunking |
| `src/ingestion/models.py` | `ProcessedChunk` (campos de acórdão: `section_type`, `authority_level`, etc.) |
| `src/extraction/pymupdf_extractor.py` | Extração de texto nativo (modo PyMuPDF) |
| `src/extraction/vlm_service.py` | OCR por documento (modo VLM) |
