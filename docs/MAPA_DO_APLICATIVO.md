# 🗺️ Mapa do Aplicativo - RAG GPU Server

> **Repositório**: https://github.com/euteajudo/rag-gpu-server
> **Última Atualização**: 11/02/2026
> **Status**: Produção (RunPod A40 48GB)

Este documento serve como guia de navegação para desenvolvedores que precisam entender a estrutura do código e localizar funcionalidades específicas.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Estrutura de Diretórios](#estrutura-de-diretórios)
4. [Módulos Principais](#módulos-principais)
5. [Pipeline de Ingestão](#pipeline-de-ingestão)
6. [Endpoints da API](#endpoints-da-api)
7. [Fluxos de Dados](#fluxos-de-dados)
8. [Conexão com Outros Repositórios](#conexão-com-outros-repositórios)

---

## 🎯 Visão Geral

O **RAG GPU Server** é responsável pelo processamento intensivo em GPU do sistema VectorGov:

- **Embeddings**: Geração de vetores semânticos com BGE-M3 (1024 dimensões dense + sparse)
- **Reranking**: Reordenação de documentos por relevância com BGE-Reranker-v2-m3
- **Ingestão de PDFs**: Pipeline dual-entry (PyMuPDF/VLM OCR → Regex Classifier → Chunks → Embeddings)

O servidor roda no **RunPod** com GPU NVIDIA A40 (48GB VRAM) e se comunica com a VPS via **Cloudflare Tunnel**.

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RAG GPU SERVER (RunPod A40 48GB)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Server (:8000)                           │   │
│  │                                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │   │
│  │  │   /embed    │  │  /rerank    │  │       /ingest               │ │   │
│  │  │             │  │             │  │                             │ │   │
│  │  │ BGE-M3      │  │ BGE-Reranker│  │ PyMuPDF / VLM OCR →        │ │   │
│  │  │ (embeddings)│  │ (cross-enc) │  │ Regex Classifier →          │ │   │
│  │  │             │  │             │  │ Chunks → Embeddings         │ │   │
│  │  │ BatchCollect│  │ BatchCollect│  │                             │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   BGE-M3     │  │ BGE-Reranker │  │    Redis     │  │   PyMuPDF    │   │
│  │   (~2GB)     │  │   (~1GB)     │  │   :6379      │  │  (CPU only)  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    vLLM Server (:8002)                              │   │
│  │                                                                     │   │
│  │  Qwen/Qwen3-VL-8B-Instruct (multimodal)                            │   │
│  │  - OCR de páginas de PDF (Entrada 2)                                │   │
│  │  - max_model_len: 8192                                              │   │
│  │  - API OpenAI-compatible                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                         Cloudflare Tunnel
                    gpu.vectorgov.io / llm.vectorgov.io
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VPS HOSTINGER                                      │
│                                                                             │
│  RemoteEmbedder ──► gpu.vectorgov.io/embed                                 │
│  RemoteReranker ──► gpu.vectorgov.io/rerank                                │
│  MinIO (:9100) ◄── RunPod POST multipart (artefatos)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estrutura de Diretórios

```
rag-gpu-server/
├── src/
│   ├── main.py                 # Entrada FastAPI, endpoints principais
│   ├── config.py               # Configurações (modelos, URLs, pipeline)
│   ├── auth.py                 # Autenticação por API Key
│   ├── embedder.py             # BGE-M3 wrapper
│   ├── reranker.py             # BGE-Reranker wrapper
│   ├── batch_collector.py      # Micro-batching para performance
│   │
│   ├── extraction/             # Extração de texto e OCR
│   │   ├── pymupdf_extractor.py # PyMuPDF: páginas → blocos + canonical_text
│   │   ├── regex_classifier.py  # Regex Classifier: blocos → dispositivos legais
│   │   ├── vlm_client.py       # Cliente HTTP para Qwen3-VL (extract + OCR)
│   │   ├── vlm_service.py      # Orquestrador: PyMuPDF → VLM → DocumentExtraction
│   │   ├── vlm_ocr.py          # OCR: prompts, split_ocr_into_blocks, quality gate
│   │   ├── vlm_models.py       # PageData, BlockData, DocumentExtraction
│   │   ├── vlm_prompts.py      # Prompts para classificação VLM (legado)
│   │   └── coord_utils.py      # Conversão coordenadas (img 0-1 ↔ PDF pts)
│   │
│   ├── ingestion/              # Pipeline de ingestão de PDFs
│   │   ├── router.py           # Endpoints /ingest, /ingest/status, /ingest/result
│   │   ├── pipeline.py         # Pipeline dual-entry (PyMuPDF + VLM OCR)
│   │   └── models.py           # IngestRequest, IngestResult, ProcessedChunk
│   │
│   ├── inspection/             # Pipeline de inspeção visual (QA)
│   │   ├── router.py           # Endpoints /inspect/*
│   │   ├── pipeline.py         # Pipeline de inspeção (PyMuPDF + Regex)
│   │   ├── models.py           # RegexClassificationArtifact, PyMuPDFArtifact
│   │   ├── storage.py          # Redis storage para artefatos de inspeção
│   │   └── static/             # Frontend HTML para visualização
│   │
│   ├── classification/         # Classificação de origem
│   │   └── origin_classifier.py # OriginClassifier: identifica citations cruzadas
│   │
│   ├── chunking/               # Utilitários de chunking
│   │   ├── canonical_offsets.py # Offsets canônicos (char_start/char_end)
│   │   ├── citation_extractor.py # Extração de citações cruzadas
│   │   └── rel_type_classifier.py # Classificação de tipo de relação
│   │
│   ├── sinks/                  # Upload de artefatos
│   │   ├── artifacts_uploader.py # Upload de chunks → VPS → MinIO
│   │   └── inspection_uploader.py # Upload de inspeção → VPS → MinIO
│   │
│   └── utils/                  # Utilitários compartilhados
│       ├── canonical_utils.py  # normalize_canonical_text, compute_canonical_hash
│       ├── matching_normalization.py # NFKC, OCR table, hyphen break
│       └── normalization.py    # normalize_document_id
│
├── docs/                       # Documentação
│   ├── MAPA_DO_APLICATIVO.md   # Este arquivo
│   └── QWEN3_PIPELINE_ROLE.md  # Papel do Qwen3-VL no pipeline
│
└── tests/                      # Testes (329 testes)
    ├── test_pr13_acceptance.py  # Testes de aceitação (regex + OCR blocks)
    ├── test_c4_fallback.py     # Testes de fallback C4
    ├── test_origin_classifier.py # Testes do classificador de origem
    └── ...
```

---

## 🧩 Módulos Principais

### 1. API FastAPI (`src/main.py`)

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Servidor principal | `main.py:app` | FastAPI com lifespan para carregar modelos |
| Endpoint embeddings | `main.py:embed()` | POST /embed |
| Endpoint reranking | `main.py:rerank()` | POST /rerank |
| Health check | `main.py:health()` | GET /health |
| Estatísticas | `main.py:stats()` | GET /stats |
| Lifespan | `main.py:lifespan()` | Carrega modelos na GPU no startup |

### 2. Embedder (`src/embedder.py`)

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Wrapper BGE-M3 | `BGEM3Embedder` | Gera embeddings dense (1024d) + sparse |
| Singleton | `get_embedder()` | Retorna instância única |
| Health check | `BGEM3Embedder.health_check()` | Verifica status do modelo |
| Encode | `BGEM3Embedder.encode()` | Processa lista de textos |

### 3. Reranker (`src/reranker.py`)

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Wrapper BGE-Reranker | `BGEReranker` | Cross-encoder para relevância |
| Singleton | `get_reranker()` | Retorna instância única |
| Rerank | `BGEReranker.rerank()` | Reordena documentos por query |
| Rankings | `RerankResult.rankings` | Índices ordenados por score |

### 4. Batch Collector (`src/batch_collector.py`)

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Micro-batching | `BatchCollector` | Agrupa requests para GPU |
| Embed processor | `create_embed_batch_processor()` | Batch de embeddings |
| Rerank processor | `create_rerank_batch_processor()` | Batch de reranking |
| Configuração | `BATCH_CONFIG` | max_batch_size, max_wait_ms |

### 5. Autenticação (`src/auth.py`)

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Middleware | `APIKeyAuthMiddleware` | Valida X-GPU-API-Key |
| IP allowlist | `ALLOWED_IPS` | IPs permitidos (opcional) |
| API Keys | `VALID_API_KEYS` | Keys válidas (env: GPU_API_KEYS) |
| Endpoints públicos | `PUBLIC_ENDPOINTS` | /health, /docs, etc |

---

## 📄 Pipeline de Ingestão

### Visão Geral (`src/ingestion/pipeline.py`)

O pipeline suporta duas entradas que convergem no mesmo processamento downstream:

```
  ENTRADA 1 (PyMuPDF nativo)           ENTRADA 2 (VLM OCR)
  ─────────────────────────            ──────────────────────
  PDF                                  PDF
   │                                    │
   ▼                                    ▼
  PyMuPDF                              PyMuPDF (imagens only)
  extract_pages()                           │
   │                                        ▼
   │                                   Qwen3-VL OCR
   │                                   ocr_page() por página
   │                                        │
   │                                   split_ocr_into_blocks()
   │                                   ocr_to_pages_data()
   │                                        │
   ├── pages_data                      ├── pages_data (sintéticos)
   └── canonical_text                  └── canonical_text (OCR)
              │                                  │
              └──────────┬───────────────────────┘
                         │
                         ▼
              ┌──────────────────────────────────────┐
              │  _convert_pages_to_classifier_format()│
              │  classify_to_devices()                │
              │  (Regex Classifier — MESMO para E1/E2)│
              └──────────────────────────────────────┘
                         │
                         ▼
              _regex_to_processed_chunks()
              _build_retrieval_text()
                         │
              ┌──────────┼──────────┬──────────┐
              │          │          │          │
              ▼          ▼          ▼          ▼
         OriginClass  BGE-M3    Artifacts   Contract
         (citations)  Embeddings  Upload    Validation
                         │
                         ▼
                   IngestResponse
                   (chunks + vetores)
```

> **"A única variável é DE ONDE vem o texto."** — Design doc v3

### Fases do Pipeline

| Fase | Módulo | Descrição | Output |
|------|--------|-----------|--------|
| 1a (E1) | PyMuPDF | PDF → páginas + canonical_text | pages_data, canonical_text |
| 1b (E2) | PyMuPDF + Qwen3-VL | PDF → imagens → OCR por página | pages_data, canonical_text |
| 2 | Regex Classifier | Texto → dispositivos legais hierárquicos | ClassifiedDevice[] |
| 3 | Chunk Builder | Dispositivos → ProcessedChunks com retrieval_text | ProcessedChunk[] |
| 4 | OriginClassifier | Identifica citações cruzadas entre normas | citations[] |
| 5 | BGE-M3 | Geração de embeddings | Vetores dense (1024d) + sparse |
| 6 | Artifacts Upload | Upload de evidência (PDF, chunks, inspeção) | MinIO via VPS |

### Módulos do Pipeline

#### PyMuPDF Extractor (`src/extraction/pymupdf_extractor.py`)

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Extração | `PyMuPDFExtractor.extract_pages()` | Extrai blocos de texto + imagens PNG |
| Output | `(List[PageData], str)` | pages_data + canonical_text (NFC normalizado) |
| Offsets | Nativos | char_start/char_end computados durante concatenação |
| Blocos | `BlockData` | block_index, text, bbox_pdf, char_start, char_end |

#### Regex Classifier (`src/extraction/regex_classifier.py`)

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Classificação | `classify_to_devices()` | Identifica Art., §, incisos, alíneas |
| Hierarquia | `ClassifiedDevice` | parent_span_id, children_span_ids, hierarchy_depth |
| Span IDs | `ART-001`, `PAR-001-1`, `INC-001-1` | Formato determinístico |
| Filtros | metadata, cabeçalho, preâmbulo | Blocos não-normativos separados |

#### VLM OCR (`src/extraction/vlm_ocr.py`) — Entrada 2 only

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Prompt OCR | `OCR_SYSTEM_PROMPT` | Transcrição precisa de documentos legais |
| Split em blocos | `split_ocr_into_blocks()` | Texto OCR → blocos sintéticos com offsets |
| Montagem | `ocr_to_pages_data()` | Combina imagens PyMuPDF + blocos OCR |
| Quality Gate | `validate_ocr_quality()` | 3 checks: artigos, chars/página, dispositivos/página |

---

## 🔄 Fluxo Detalhado por Entrada

### Entrada 1 — PyMuPDF nativo (`extraction_mode != "vlm"`)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         GPU Server (pipeline.py)                             │
│                                                                              │
│  PDF ─► PyMuPDF extract_pages()                                              │
│              │                                                               │
│              ├── pages_data (blocos com bbox, offsets nativos)                │
│              └── canonical_text (NFC normalizado)                             │
│                       │                                                      │
│                       ▼                                                      │
│              Regex Classifier ─► ClassifiedDevice[]                           │
│                       │                                                      │
│                       ▼                                                      │
│              ProcessedChunks ─► OriginClassifier ─► BGE-M3 ─► Artifacts      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Entrada 2 — VLM OCR (`extraction_mode == "vlm"`)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         GPU Server (pipeline.py)                             │
│                                                                              │
│  PDF ─► PyMuPDF extract_pages() ─► imagens PNG (texto descartado)            │
│              │                                                               │
│              ▼                                                               │
│         Qwen3-VL ocr_page() (sequencial, 1 página por vez)                   │
│              │                                                               │
│              ▼                                                               │
│         split_ocr_into_blocks() ─► blocos sintéticos + canonical_text        │
│         ocr_to_pages_data()     ─► pages_data (mesmo formato de E1)          │
│              │                                                               │
│              ▼                                                               │
│         validate_ocr_quality()  ─► warnings (artigos, chars, dispositivos)   │
│              │                                                               │
│              ▼                                                               │
│         MESMO pipeline de E1:                                                │
│         Regex Classifier ─► ProcessedChunks ─► OriginClassifier ─► ...       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Onde o Trabalho GPU Acontece

| Operação | Módulo | GPU? |
|----------|--------|------|
| PyMuPDF (extração de texto/imagens) | `pymupdf_extractor.py` | Não (CPU) |
| Qwen3-VL OCR (Entrada 2) | `vlm_client.py` → vLLM :8002 | Sim |
| Regex Classifier | `regex_classifier.py` | Não (CPU) |
| BGE-M3 Embeddings | `embedder.py` | Sim |
| BGE-Reranker | `reranker.py` | Sim |

---

## 🔌 Endpoints da API

### Embeddings

```http
POST /embed
Content-Type: application/json
X-GPU-API-Key: vg_gpu_xxx

{
  "texts": ["texto 1", "texto 2"],
  "return_dense": true,
  "return_sparse": true
}

Response:
{
  "dense_embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "sparse_embeddings": [{"123": 0.5, "456": 0.3}, ...],
  "latency_ms": 45.2,
  "count": 2
}
```

### Reranking

```http
POST /rerank
Content-Type: application/json
X-GPU-API-Key: vg_gpu_xxx

{
  "query": "O que é ETP?",
  "documents": ["doc1", "doc2", "doc3"],
  "top_k": 3
}

Response:
{
  "scores": [0.95, 0.72, 0.43],
  "rankings": [0, 1, 2],
  "latency_ms": 120.5
}
```

### Ingestão

```http
POST /ingest
Content-Type: multipart/form-data
X-GPU-API-Key: vg_gpu_xxx

file: <PDF>
document_id: IN-65-2021
tipo_documento: IN
numero: 65
ano: 2021

# Parâmetros opcionais de validação de artigos:
validate_articles: true          # Habilita validação
expected_first_article: 1        # Primeiro artigo esperado
expected_last_article: 18        # Último artigo esperado

Response:
{
  "success": true,
  "document_id": "IN-65-2021",
  "status": "COMPLETED",
  "total_chunks": 47,
  "phases": [...],
  "chunks": [...],
  "document_hash": "abc123...",
  "validation_docling": {
    "status": "passed",
    "found_articles": ["1", "2", ..., "18"],
    "missing_articles": [],
    "split_articles": [],
    "coverage_percent": 100.0,
    "chunks_manifest": ["ART-001", "PAR-001-1", ...]
  }
}
```

### Health

```http
GET /health

Response:
{
  "status": "healthy",
  "embedder": {"status": "online", "model": "BAAI/bge-m3"},
  "reranker": {"status": "online", "model": "BAAI/bge-reranker-v2-m3"},
  "vlm_service": {"status": "online"},
  "uptime_seconds": 3600.5
}
```

---

## 🔄 Fluxos de Dados

### Fluxo de Embedding (VPS → GPU)

```
VPS (RemoteEmbedder)
        │
        │ POST /embed
        │ Headers: CF-Access-*, X-GPU-API-Key
        ▼
Cloudflare Access (valida Service Token)
        │
        ▼
GPU Server (auth.py valida API Key)
        │
        ▼
BatchCollector (agrupa requests)
        │
        ▼
BGEM3Embedder.encode()
        │
        ▼
GPU (FlagEmbedding)
        │
        ▼
EmbedResponse → VPS
```

### Fluxo de Ingestão Completo

```
VPS ──► POST /ingest (PDF + extraction_mode)
            │
            ▼
        ┌───────────────────────────────────────────────────────┐
        │                  GPU Server                            │
        │                                                        │
        │  extraction_mode == "vlm"?                             │
        │      │                                                 │
        │      ├── NÃO (Entrada 1):                              │
        │      │   PyMuPDF ──► pages_data + canonical_text       │
        │      │                                                 │
        │      └── SIM (Entrada 2):                              │
        │          PyMuPDF (imgs) + Qwen3-VL (OCR)               │
        │          ──► pages_data + canonical_text                │
        │                    │                                   │
        │                    ▼                                   │
        │  Regex Classifier ──► ClassifiedDevice[]               │
        │       │                                                │
        │       ▼                                                │
        │  ProcessedChunks ──► OriginClassifier (citations)      │
        │       │                                                │
        │       ▼                                                │
        │  BGE-M3 ──► Embeddings (dense 1024d + sparse)          │
        │       │                                                │
        │       ▼                                                │
        │  Artifacts Upload ──► VPS ──► MinIO                    │
        │                                                        │
        └───────────────────────────────────────────────────────┘
            │
            ▼
        IngestResponse (chunks com embeddings) ──► VPS
            │
            ▼
        VPS insere no Milvus + Neo4j
```

---

## 🔗 Conexão com Outros Repositórios

### Ecossistema VectorGov

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          ECOSSISTEMA VECTORGOV                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────────┐      ┌─────────────────────────┐             │
│  │   vector_govi_2         │      │   rag-gpu-server        │             │
│  │   (Monorepo Principal)  │      │   (GPU Processing)      │             │
│  │                         │      │                         │             │
│  │  • extracao/ (docs)     │      │  • /embed               │             │
│  │  • frontend/            │◄────►│  • /rerank              │             │
│  │  • scripts/             │ HTTP │  • /ingest              │             │
│  │  • rag-gpu-server/      │      │  • Pipeline completo    │             │
│  │    (submodule)          │      │                         │             │
│  └─────────────────────────┘      └─────────────────────────┘             │
│              │                               ▲                             │
│              │                               │                             │
│              ▼                               │                             │
│  ┌─────────────────────────┐                │                             │
│  │   vectorgov-sdk         │                │                             │
│  │   (SDK Python)          │                │                             │
│  │                         │                │                             │
│  │  • VectorGov client     │────────────────┘                             │
│  │  • LangChain/LangGraph  │   (via VPS API)                              │
│  │  • MCP Server           │                                              │
│  └─────────────────────────┘                                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Responsabilidades por Repositório

| Repositório | Responsabilidade | Componentes |
|-------------|------------------|-------------|
| **vector_govi_2** | Monorepo principal, documentação, frontend | extracao/, frontend/, scripts/ |
| **rag-gpu-server** | Processamento GPU, ingestão, embeddings | FastAPI, BGE-M3, PyMuPDF, Qwen3-VL, Pipeline |
| **vectorgov-sdk** | SDK Python para integração | VectorGov client, LangChain, MCP |

### Comunicação entre Repositórios

| De | Para | Protocolo | Endpoints |
|----|------|-----------|-----------|
| VPS (vector_govi_2) | GPU Server | HTTPS + Cloudflare | /embed, /rerank, /ingest |
| SDK | VPS API | HTTPS | /api/v1/sdk/* |
| VPS | Milvus | TCP | :19530 |
| VPS | Redis | TCP | :6379 |
| VPS | PostgreSQL | TCP | :5432 |

---

## 🔧 Configuração

### Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `HOST` | 0.0.0.0 | Host do servidor |
| `PORT` | 8000 | Porta do servidor |
| `EMBEDDING_MODEL` | BAAI/bge-m3 | Modelo de embeddings |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | Modelo de reranking |
| `VLLM_BASE_URL` | http://localhost:8002/v1 | URL do vLLM |
| `VLLM_MODEL` | Qwen/Qwen3-VL-8B-Instruct | Modelo VLM (multimodal) |
| `GPU_API_KEYS` | vg_gpu_internal_2025 | API Keys válidas |
| `ALLOWED_IPS` | * | IPs permitidos |
| `DEVICE` | cuda | Dispositivo (cuda/cpu) |
| `USE_FP16` | true | Usar FP16 |

### Arquivo de Configuração (`src/config.py`)

```python
@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8000
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    vllm_base_url: str = "http://localhost:8002/v1"
    vllm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    use_fp16: bool = True
    device: str = "cuda"
```

---

## 📊 Métricas e Monitoramento

### Endpoint /stats

```json
{
  "uptime_seconds": 3600.5,
  "gpu_executor": {
    "max_workers": 2,
    "active_threads": 1
  },
  "batch_collectors": {
    "embed": {
      "batches_processed": 150,
      "items_processed": 1200,
      "avg_batch_size": 8.0,
      "avg_latency_ms": 45.2
    },
    "rerank": {
      "batches_processed": 50,
      "items_processed": 200,
      "avg_batch_size": 4.0,
      "avg_latency_ms": 120.5
    }
  }
}
```

---

## 🚀 Deploy

### RunPod

```bash
# Iniciar servidor
cd /workspace/rag-gpu-server
/workspace/venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Com variáveis
export GPU_API_KEYS="vg_gpu_xxx,vg_gpu_yyy"
export VLLM_BASE_URL="http://localhost:8001/v1"
```

### Script de Inicialização

```bash
#!/bin/bash
# /workspace/init-after-restart.sh

export GPU_API_KEYS="vg_gpu_internal_2025"
export VLLM_BASE_URL="http://localhost:8001/v1"

nohup /workspace/venv/bin/python -m uvicorn src.main:app \
    --host 0.0.0.0 --port 8000 > /workspace/gpu-server.log 2>&1 &
```

---

## 📚 Referências

- [FlagEmbedding (BGE-M3)](https://github.com/FlagOpen/FlagEmbedding)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [vLLM](https://docs.vllm.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
