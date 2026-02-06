# TASK: Pipeline de Extração VLM — Contexto no Pipeline Existente

## ⚠️ LEIA PRIMEIRO

Esta tarefa NÃO é construir um pipeline do zero. Estamos **substituindo apenas uma parte** de um pipeline que já existe e funciona. A maior parte do pipeline permanece intacta. Este documento explica exatamente O QUE muda, o que NÃO muda, e onde os novos componentes se conectam aos existentes.

---

## 1. O Pipeline Completo Atual (ANTES)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE INGESTÃO ATUAL                       │
│                                                                     │
│  PDF (do MinIO)                                                     │
│   │                                                                 │
│   ▼                                                                 │
│  ┌──────────────┐                                                   │
│  │   Docling     │  ← Extrai texto do PDF (gera markdown)          │
│  └──────┬───────┘                                                   │
│         │ texto markdown (não-determinístico, line breaks errados)   │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │  SpanParser   │  ← Regex identifica Art., §, incisos, alíneas   │
│  └──────┬───────┘                                                   │
│         │ dispositivos legais com span_id (falsos positivos!)       │
│         ▼                                                           │
│  ┌───────────────────┐                                              │
│  │ canonical_builder  │  ← Constrói texto canônico + offsets        │
│  │     (PR12)         │     normalize_canonical_text()               │
│  │                    │     compute_canonical_hash()                 │
│  └──────┬────────────┘                                              │
│         │ canonical_text + offsets + canonical_hash                  │
│         ▼                                                           │
│  ┌───────────────────┐                                              │
│  │ canonical_offsets  │  ← Resolve offsets dos filhos dentro do pai  │
│  └──────┬────────────┘                                              │
│         │ chunks com (text, canonical_start, canonical_end,         │
│         │             canonical_hash, node_id, parent_id)           │
│         ▼                                                           │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ PONTO DE CONEXÃO ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│         │                                                           │
│         │  Daqui para baixo, os módulos são AGNÓSTICOS ao           │
│         │  parser — recebem chunks com campos padronizados          │
│         │  e não sabem quem os produziu.                            │
│         │                                                           │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ canonical_validation   │  ← Valida formato node_id               │
│  │                        │     prefixo leis:/acordaos:              │
│  └──────┬────────────────┘                                          │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ pr13_validator (PR13)  │  ← GATE CRÍTICO pré-Milvus             │
│  │                        │     canonical_start >= 0                │
│  │                        │     canonical_end >= canonical_start    │
│  │                        │     canonical_hash != "" e != None      │
│  │                        │     FALHOU? → aborta doc inteiro        │
│  └──────┬────────────────┘                                          │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ alarm_service          │  ← Persiste alarmes no PostgreSQL       │
│  └──────┬────────────────┘                                          │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ BGE-M3 (RunPod GPU)   │  ← Gera dense + sparse embeddings       │
│  └──────┬────────────────┘                                          │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ Milvus + Neo4j         │  ← Armazena chunks + hierarquia         │
│  └──────┬────────────────┘                                          │
│         │                                                           │
│    (query-time)                                                     │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ snippet_extractor      │  ← Slicing puro com hash anti-mismatch  │
│  │       (PR10)           │     canonical_text[start:end]            │
│  └──────┬────────────────┘                                          │
│         ▼                                                           │
│  Evidence Link ao PDF no MinIO 🎯                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. O Que MUDA vs O Que NÃO MUDA

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ❌ MORRE (será substituído)         ✅ INTACTO (não mexer)        │
│   ─────────────────────────           ──────────────────────        │
│   • Docling                           • canonical_validation.py     │
│   • SpanParser (regex)                • pr13_validator.py (PR13)    │
│   • canonical_builder.py              • alarm_service.py            │
│     (construção do canonical)         • snippet_extractor.py (PR10) │
│                                       • BGE-M3 embeddings           │
│   🔄 TRANSFORMA                       • Milvus (schema evolui)     │
│   ──────────────────                  • Neo4j                       │
│   • canonical_builder.py              • MinIO storage               │
│     → canonical_utils.py              • PostgreSQL                  │
│     (só normalize + hash)             • FastAPI endpoints           │
│   • canonical_offsets.py                                            │
│     → fallback/validação                                            │
│                                                                     │
│   🆕 NOVO (será criado)                                            │
│   ──────────────────────                                            │
│   • pymupdf_extractor.py                                           │
│   • vlm_service.py                                                  │
│   • reconciliator.py                                                │
│   • integrity_validator.py                                          │
│   • coord_utils.py                                                  │
│   • text_normalizer.py                                              │
│   • artifacts.py                                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. O Pipeline DEPOIS (com a área de mudança destacada)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE INGESTÃO NOVO                        │
│                                                                     │
│  PDF (do MinIO)                                                     │
│   │                                                                 │
│   │                                                                 │
│ ╔═══════════════════════════════════════════════════════════════╗    │
│ ║              ZONA DE MUDANÇA (só isso muda)                  ║    │
│ ║                                                              ║    │
│ ║  │                                                           ║    │
│ ║  ├───► PyMuPDF Extractor (ETAPA 1)                          ║    │
│ ║  │      │                                                    ║    │
│ ║  │      ├── get_text("dict") → blocos com bbox (PDF space)   ║    │
│ ║  │      ├── concatena blocos → canonical_text                ║    │
│ ║  │      ├── calcula char_start/char_end por bloco            ║    │
│ ║  │      ├── normalize + SHA-256 → canonical_hash             ║    │
│ ║  │      └── get_pixmap(dpi=300) → imagens PNG das páginas    ║    │
│ ║  │             │                                             ║    │
│ ║  │             ▼                                             ║    │
│ ║  │      Qwen3-VL via vLLM (ETAPA 2)                         ║    │
│ ║  │      │                                                    ║    │
│ ║  │      └── recebe imagens → identifica dispositivos legais  ║    │
│ ║  │          tipo + identificador + bbox (image space)        ║    │
│ ║  │          hierarquia pai/filho + confidence                ║    │
│ ║  │             │                                             ║    │
│ ║  │             ▼                                             ║    │
│ ║  │      Reconciliator (ETAPA 3)                              ║    │
│ ║  │      │                                                    ║    │
│ ║  │      ├── coord_utils: image space → PDF space             ║    │
│ ║  │      ├── matching: bbox VLM ↔ blocos PyMuPDF             ║    │
│ ║  │      ├── text_normalizer: valida match por similaridade   ║    │
│ ║  │      ├── texto final = SEMPRE do PyMuPDF                  ║    │
│ ║  │      ├── classificação = do VLM (tipo + hierarquia)       ║    │
│ ║  │      └── monta chunks com offsets no canonical_text       ║    │
│ ║  │             │                                             ║    │
│ ║  │             ▼                                             ║    │
│ ║  │      IntegrityValidator                                   ║    │
│ ║  │      │                                                    ║    │
│ ║  │      └── invariantes T1-T3, H1-H4, G1-G4, C1-C2         ║    │
│ ║  │                                                           ║    │
│ ╚══╪═══════════════════════════════════════════════════════════╝    │
│    │                                                                │
│    │  OUTPUT: chunks com os mesmos campos que o pipeline            │
│    │  anterior produzia + campos novos (page_number, bbox,         │
│    │  confidence, extraction_method)                                │
│    │                                                                │
│    │  Campos que o pipeline antigo já produzia e que                │
│    │  continuam existindo no output:                                │
│    │    • text (agora do PyMuPDF, antes do canonical_builder)      │
│    │    • canonical_start, canonical_end, canonical_hash            │
│    │    • node_id, chunk_id, span_id, parent_id                    │
│    │    • device_type, chunk_level                                  │
│    │    • document_id, tipo_documento, numero, ano                  │
│    │                                                                │
│ ─ ─ ─ ─ ─ ─ ─ ─ PONTO DE CONEXÃO (igual ao anterior) ─ ─ ─ ─ ─ ─ │
│    │                                                                │
│    │  A partir daqui, NADA MUDA. Os módulos abaixo recebem         │
│    │  chunks com campos padronizados — não sabem e não se           │
│    │  importam se vieram do SpanParser ou do Reconciliator.         │
│    │                                                                │
│    ▼                                                                │
│  ┌───────────────────────┐                                          │
│  │ canonical_validation   │  ← INTACTO — valida formato node_id    │
│  └──────┬────────────────┘                                          │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ pr13_validator (PR13)  │  ← INTACTO — gate pré-Milvus          │
│  └──────┬────────────────┘                                          │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ alarm_service          │  ← INTACTO — persiste alarmes          │
│  └──────┬────────────────┘                                          │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ BGE-M3 (RunPod GPU)   │  ← INTACTO — embeddings                │
│  └──────┬────────────────┘                                          │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ Milvus + Neo4j         │  ← INTACTO (schema ganha campos novos) │
│  └──────┬────────────────┘                                          │
│         │                                                           │
│    (query-time)                                                     │
│         ▼                                                           │
│  ┌───────────────────────┐                                          │
│  │ snippet_extractor      │  ← INTACTO — slicing + hash check      │
│  │       (PR10)           │                                         │
│  └──────┬────────────────┘                                          │
│         ▼                                                           │
│  Evidence Link ao PDF no MinIO 🎯                                   │
│  (agora com page_number + bbox para highlight visual)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Visão Lado a Lado (ANTES → DEPOIS)

```
            ANTES                              DEPOIS
    ┌─────────────────┐                ┌─────────────────────────┐
    │     Docling      │    ═══►       │   PyMuPDF Extractor     │
    │  (texto sujo)    │  SUBSTITUÍDO  │   (texto determinístico) │
    └────────┬────────┘    POR         └────────┬────────────────┘
             │                                  │
             ▼                                  ├── canonical_text
    ┌─────────────────┐                         ├── blocos com bbox
    │   SpanParser     │    ═══►                ├── imagens PNG
    │    (regex)       │  SUBSTITUÍDO                    │
    └────────┬────────┘    POR                          ▼
             │                         ┌─────────────────────────┐
             │                         │     Qwen3-VL (vLLM)     │
             │                         │  (estrutura visual)      │
             │                         └────────┬────────────────┘
             │                                  │
             ▼                                  ▼
    ┌─────────────────┐                ┌─────────────────────────┐
    │canonical_builder │    ═══►       │    Reconciliator         │
    │     (PR12)       │  SUBSTITUÍDO  │ (merge PyMuPDF + VLM)   │
    │                  │    POR        │                          │
    │canonical_offsets │               │  IntegrityValidator      │
    └────────┬────────┘                └────────┬────────────────┘
             │                                  │
    ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─
             │          INTERFACE IDÊNTICA       │
             │     (mesmos campos de output)     │
             ▼                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │            PIPELINE DOWNSTREAM (NÃO MUDA)                   │
    │                                                             │
    │  canonical_validation → pr13_validator → alarm_service      │
    │         → BGE-M3 → Milvus + Neo4j                          │
    │         → snippet_extractor → Evidence Link                 │
    └─────────────────────────────────────────────────────────────┘
```

---

## 5. As 3 Etapas Novas em Detalhe

### ETAPA 1: PyMuPDF Extractor (`pymupdf_extractor.py`)

**Substitui**: Docling + parte do canonical_builder (construção do texto)

**O que faz**:
1. `page.get_text("dict")` → blocos de texto com bbox em PDF space (pontos, 72 dpi)
2. `page.get_pixmap(dpi=300)` → imagem PNG da página para o VLM
3. Concatena blocos em reading order → `canonical_text`
4. Calcula `char_start`/`char_end` por bloco DURANTE concatenação
5. `normalize_canonical_text()` + `compute_canonical_hash()` (funções existentes, extraídas para `canonical_utils.py`)

**Output**:
```python
canonical_text: str           # texto completo, determinístico
canonical_hash: str           # SHA-256

# por página:
page_image: bytes             # PNG 300 DPI

# por bloco:
block = {
    "page": int,
    "bbox": (x0, y0, x1, y1), # PDF space (pontos)
    "text": str,
    "char_start": int,         # offset no canonical_text
    "char_end": int,
    "block_index": int
}
```

**Regras**:
- O `canonical_text` é IMUTÁVEL — nenhuma etapa posterior o altera
- Coordenadas em PDF space (pontos, 72 dpi), NÃO pixels
- Offsets calculados DURANTE concatenação, não mapeados depois
- Mesmo PDF + mesma versão PyMuPDF = mesmo canonical_text sempre (idempotência)

---

### ETAPA 2: Qwen3-VL via vLLM (`vlm_service.py`)

**Substitui**: SpanParser (regex)

**O que faz**:
1. Recebe imagens PNG geradas pelo PyMuPDF na Etapa 1
2. Envia cada imagem ao Qwen3-VL via API OpenAI-compatible do vLLM
3. Prompt instrui o modelo a identificar dispositivos legais brasileiros:
   - Artigo, Parágrafo, Inciso, Alínea, Caput, Item
4. Retorna para cada dispositivo: tipo, identificador, texto OCR, bbox, parent, confidence

**Output**:
```python
device = {
    "type": str,               # "artigo" | "paragrafo" | "inciso" | "alinea" | "caput"
    "identifier": str,         # "Art. 75" | "§ 2º" | "III" | "a)"
    "text_ocr": str,           # texto lido pelo VLM (usado APENAS para matching)
    "bbox_image": (x0, y0, x1, y1),  # IMAGE space (pixels, 300 dpi)
    "page": int,
    "confidence": float,
    "parent": str | None       # identificador do pai
}
```

**Regras**:
- VLM recebe APENAS imagens, NÃO texto
- `text_ocr` é usado APENAS para matching — NUNCA como texto final
- Coordenadas em IMAGE space (pixels, 300 dpi), NÃO PDF space
- Modelo: `Qwen/Qwen3-VL-8B-Instruct` servido via vLLM no RunPod

---

### ETAPA 3: Reconciliator (`reconciliator.py`)

**Substitui**: canonical_builder (construção de chunks) + canonical_offsets (resolução)

**O que faz**:
1. Converte bbox do VLM: image space → PDF space via `coord_utils.py`
2. Matching: para cada dispositivo do VLM, encontra bloco(s) do PyMuPDF por bbox overlap + text similarity
3. Monta chunk: texto do PyMuPDF + classificação do VLM + offsets no canonical_text
4. Constrói node_id canônico (ex: `leis:LEI-14133-2021#ART-023-PAR-002`)
5. Valida via IntegrityValidator (invariantes T1-T3, H1-H4, G1-G4, C1-C2)

**Output**: chunks com interface idêntica ao que o pipeline downstream espera:
```python
chunk = {
    # Campos que JÁ EXISTIAM no pipeline anterior (interface mantida):
    "node_id": "leis:LEI-14133-2021#ART-023",
    "chunk_id": "LEI-14133-2021#ART-023",
    "logical_node_id": "leis:LEI-14133-2021#ART-023",
    "document_id": "LEI-14133-2021",
    "span_id": "ART-023",
    "text": "Art. 23. O processo de contratação direta...",  # DO PYMUPDF
    "parent_id": "leis:LEI-14133-2021#CAP-V",
    "device_type": "article",
    "chunk_level": 2,
    "canonical_start": 1847,
    "canonical_end": 2103,
    "canonical_hash": "a1b2c3d4...",

    # Campos NOVOS (não existiam antes):
    "page_number": 12,
    "bbox_x0": 72.0,
    "bbox_y0": 340.5,
    "bbox_x1": 520.3,
    "bbox_y1": 410.8,
    "confidence": 0.97,
    "extraction_method": "pymupdf+qwen3vl",
    "matching_method": "bbox_exact",  # ou "bbox_fuzzy" ou "text_fallback"
    "bbox_spans": [],                 # para dispositivos cross-page
    "ingest_run_id": "...",
    "pipeline_version": "2.0"
}
```

**Regra de ouro**: O texto final vem SEMPRE do PyMuPDF. O VLM contribui APENAS classificação (tipo + hierarquia). O pipeline downstream não sabe a diferença.

---

## 6. Por Que São Sequenciais (Não Paralelos)

```
ETAPA 1 (PyMuPDF)
    │
    ├──── produz imagens ──────► ETAPA 2 (VLM) precisa das imagens
    │                                │
    ├──── produz blocos ─────────────┤
    │                                ▼
    └──── produz canonical_text ──► ETAPA 3 (Reconciliator) precisa de AMBOS
```

A Etapa 2 DEPENDE da Etapa 1 (precisa das imagens PNG).
A Etapa 3 DEPENDE das Etapas 1 E 2 (precisa dos blocos + dispositivos).

Otimização possível: pipeline página-a-página (processar página N no VLM enquanto PyMuPDF extrai página N+1). Mas para cada página individual, a ordem é sempre: PyMuPDF primeiro → VLM depois.

---

## 7. Resumo: Onde o Novo Se Liga ao Antigo

```
                    ┌─────────────────────────┐
                    │      NOVO PIPELINE       │
                    │                          │
  PDF ──────────►   │  PyMuPDF → VLM →        │
                    │  Reconciliator →         │
                    │  IntegrityValidator      │
                    │                          │
                    └────────────┬─────────────┘
                                 │
                    Produz chunks com os MESMOS campos
                    que o SpanParser produzia + extras
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                   PIPELINE EXISTENTE (NÃO MUDA)              │
  │                                                              │
  │  canonical_validation ──► pr13_validator ──► alarm_service   │
  │         │                                                    │
  │         ▼                                                    │
  │  BGE-M3 embeddings ──► Milvus (collection leis_v5) + Neo4j  │
  │         │                                                    │
  │    (query-time)                                              │
  │         ▼                                                    │
  │  snippet_extractor (PR10) ──► Evidence Link (MinIO)          │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

**A única exigência para que a conexão funcione**: o output do Reconciliator deve ter os mesmos campos obrigatórios que o pipeline downstream espera: `text`, `canonical_start`, `canonical_end`, `canonical_hash`, `node_id`, `parent_id`, `device_type`. Se esses campos estiverem corretos, tudo downstream funciona sem alteração.

---

## 8. Tabela de Componentes

| Componente | Roda onde | Status | Arquivo |
|---|---|---|---|
| PyMuPDF Extractor | RunPod (CPU) | 🆕 NOVO | `src/extraction/pymupdf_extractor.py` |
| coord_utils | RunPod (CPU) | 🆕 NOVO | `src/extraction/coord_utils.py` |
| text_normalizer | RunPod (CPU) | 🆕 NOVO | `src/extraction/text_normalizer.py` |
| VLM Service | RunPod (GPU via vLLM) | 🆕 NOVO | `src/extraction/vlm_service.py` |
| Reconciliator | RunPod (CPU) | 🆕 NOVO | `src/extraction/reconciliator.py` |
| IntegrityValidator | RunPod (CPU) | 🆕 NOVO | `src/extraction/integrity_validator.py` |
| artifacts | RunPod (CPU) | 🆕 NOVO | `src/extraction/artifacts.py` |
| canonical_utils | RunPod (CPU) | 🔄 EXTRAÍDO de canonical_builder | `src/utils/canonical_utils.py` |
| canonical_validation | VPS | ✅ INTACTO | existente |
| pr13_validator (PR13) | VPS | ✅ INTACTO | existente |
| alarm_service | VPS | ✅ INTACTO | existente |
| snippet_extractor (PR10) | VPS | ✅ INTACTO | existente |
| BGE-M3 | RunPod (GPU) | ✅ INTACTO | existente |
| Milvus | VPS | ✅ INTACTO (schema evolui) | existente |
| Neo4j | VPS | ✅ INTACTO | existente |
| MinIO | VPS | ✅ INTACTO | existente |
| Docling | — | ❌ MORRE | remover |
| SpanParser | — | ❌ MORRE | remover |
| canonical_builder (construção) | — | ❌ MORRE | funções de normalização extraídas |
