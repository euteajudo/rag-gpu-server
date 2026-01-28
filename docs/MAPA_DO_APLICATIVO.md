# 🗺️ Mapa do Aplicativo - RAG GPU Server

> **Repositório**: https://github.com/euteajudo/rag-gpu-server
> **Última Atualização**: 28/01/2026
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
- **Ingestão de PDFs**: Pipeline completo de processamento (Docling → SpanParser → LLM → Chunks → Embeddings)

O servidor roda no **RunPod** com GPU NVIDIA A40 (48GB VRAM) e se comunica com a VPS via **Cloudflare Tunnel**.

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RAG GPU SERVER (RunPod)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Server (:8000)                           │   │
│  │                                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │   │
│  │  │   /embed    │  │  /rerank    │  │       /ingest               │ │   │
│  │  │             │  │             │  │                             │ │   │
│  │  │ BGE-M3      │  │ BGE-Reranker│  │ Docling → SpanParser →      │ │   │
│  │  │ (embeddings)│  │ (cross-enc) │  │ ArticleOrchestrator →       │ │   │
│  │  │             │  │             │  │ ChunkMaterializer →         │ │   │
│  │  │ BatchCollect│  │ BatchCollect│  │ Embeddings                  │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              │ GPU                                          │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    NVIDIA A40 (48GB VRAM)                           │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │   │
│  │  │    BGE-M3       │  │  BGE-Reranker   │  │   Docling (Layout)  │ │   │
│  │  │    (~2GB)       │  │    (~1GB)       │  │      (~3GB)         │ │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    vLLM Server (:8001)                              │   │
│  │                                                                     │   │
│  │  Qwen/Qwen3-8B-AWQ                                                  │   │
│  │  - max_model_len: 8192                                              │   │
│  │  - prefix_caching: enabled                                          │   │
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
│  RemoteLLM ──────► llm.vectorgov.io/v1/chat/completions                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estrutura de Diretórios

```
rag-gpu-server/
├── src/
│   ├── main.py                 # Entrada FastAPI, endpoints principais
│   ├── config.py               # Configurações (modelos, URLs)
│   ├── auth.py                 # Autenticação por API Key
│   ├── embedder.py             # BGE-M3 wrapper
│   ├── reranker.py             # BGE-Reranker wrapper
│   ├── batch_collector.py      # Micro-batching para performance
│   │
│   ├── ingestion/              # Pipeline de ingestão de PDFs
│   │   ├── router.py           # Endpoint /ingest
│   │   ├── pipeline.py         # Pipeline completo (5 fases)
│   │   ├── models.py           # Modelos Pydantic
│   │   └── quality_validator.py # Validação de qualidade
│   │
│   ├── parsing/                # Parsing de documentos legais
│   │   ├── span_parser.py      # Regex-first parser (determinístico)
│   │   ├── article_orchestrator.py # Extração por artigo com LLM
│   │   ├── span_models.py      # Span, SpanType, ParsedDocument
│   │   ├── span_extraction_models.py # ArticleSpans (schema LLM)
│   │   └── span_extractor.py   # Extrator de spans
│   │
│   ├── chunking/               # Materialização de chunks
│   │   ├── chunk_materializer.py # Parent-child chunks
│   │   ├── chunk_models.py     # LegalChunk, ChunkLevel
│   │   ├── enrichment_prompts.py # Prompts Contextual Retrieval
│   │   └── law_chunker.py      # Chunker legado
│   │
│   ├── enrichment/             # Enriquecimento de chunks
│   │   ├── chunk_enricher.py   # Geração context/thesis/questions
│   │   ├── celery_app.py       # Configuração Celery
│   │   ├── tasks.py            # Tasks principais
│   │   ├── tasks_http.py       # Tasks via HTTP
│   │   └── tasks_pod.py        # Tasks específicas do pod
│   │
│   ├── llm/                    # Cliente LLM
│   │   └── vllm_client.py      # VLLMClient (API OpenAI-compatible)
│   │
│   ├── models/                 # Modelos de domínio
│   │   ├── legal_document.py   # LegalDocument, Chapter, Article
│   │   └── extraction_utils.py # Utilitários de extração
│   │
│   └── remote/                 # Clientes remotos (quando VPS chama GPU)
│       ├── embedder.py         # RemoteEmbedder
│       ├── reranker.py         # RemoteReranker
│       └── llm.py              # RemoteLLM
│
├── docs/                       # Documentação
│   └── MAPA_DO_APLICATIVO.md   # Este arquivo
│
└── tests/                      # Testes
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

```
PDF → Fase 1 → Fase 2 → Fase 3 → Fase 4 → Fase 5 → Chunks
       │         │         │         │         │
       ▼         ▼         ▼         ▼         ▼
    Docling   SpanParser ArticleOrch Materializer Embeddings
```

### Fases do Pipeline

| Fase | Módulo | Descrição | Output |
|------|--------|-----------|--------|
| 1 | Docling | PDF → Markdown estruturado | Texto markdown |
| 2 | SpanParser | Markdown → Spans determinísticos | ParsedDocument |
| 3 | ArticleOrchestrator | Extração LLM por artigo | ArticleChunks |
| 4 | ChunkMaterializer | Parent-child chunks | MaterializedChunks |
| 5 | BGE-M3 | Geração de embeddings | Vetores dense+sparse |

### Módulos do Pipeline

#### SpanParser (`src/parsing/span_parser.py`)

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Parser regex | `SpanParser` | Identifica estrutura hierárquica |
| Padrão Artigo | `PATTERN_ARTIGO` | `Art. 1º`, `Art. 10` |
| Padrão Parágrafo | `PATTERN_PARAGRAFO` | `§ 1º`, `Parágrafo único` |
| Padrão Inciso | `PATTERN_INCISO` | `I -`, `II -` |
| Padrão Alínea | `PATTERN_ALINEA` | `a)`, `b)` |
| Output | `ParsedDocument` | Documento com spans identificados |

#### ArticleOrchestrator (`src/parsing/article_orchestrator.py`)

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Orquestrador | `ArticleOrchestrator` | Processa artigo por artigo |
| Extração LLM | `extract_article()` | Usa Qwen para extrair hierarquia |
| Validação | `ValidationStatus` | VALID, SUSPECT, INVALID |
| Cobertura | `ArticleChunk.coverage_*` | Métricas de cobertura |

#### ChunkMaterializer (`src/chunking/chunk_materializer.py`)

| Funcionalidade | Localização | Descrição |
|----------------|-------------|-----------|
| Materialização | `ChunkMaterializer` | Transforma em chunks indexáveis |
| Parent-child | `MaterializedChunk` | chunk_id, parent_chunk_id |
| Tipos | `DeviceType` | ARTICLE, PARAGRAPH, INCISO, ALINEA |
| Metadados | `ChunkMetadata` | schema_version, document_hash |

---

## 🔄 Arquitetura de Enriquecimento

O enriquecimento de chunks adiciona contexto semântico (context_header, thesis_text, synthetic_questions) para melhorar a qualidade da busca. A arquitetura difere entre **Normas** e **Acordãos**.

### Comparativo: Normas vs Acordãos

| Aspecto | Normas (Leis/Decretos/INs) | Acordãos (TCU) |
|---------|---------------------------|----------------|
| **Orquestração** | VPS (Celery workers) | GPU Server (pipeline.py) |
| **Quando executa** | Após inserção no Milvus/Neo4j | Durante ingestão |
| **Parâmetro** | Sempre separado | `skip_enrichment` (checkbox) |
| **Trabalho GPU** | vLLM + BGE-M3 | vLLM + BGE-M3 |

### Pipeline de Normas (Enrichment Pós-Indexação)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         INGESTÃO (GPU Server)                                │
│  PDF → Docling → SpanParser → ArticleOrchestrator → Materializer → Embeddings│
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │ Chunks (sem enrichment)
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              VPS                                             │
│  1. Insere chunks no Milvus                                                  │
│  2. Cria nodes/edges no Neo4j                                                │
│  3. Dispara Celery tasks para enrichment                                     │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
┌───────────────────────────────┐    ┌────────────────────────────────────────┐
│  VPS: Celery (Orquestração)   │    │        GPU Server (Trabalho Pesado)    │
│                               │    │                                        │
│  Fila: llm_enrich (6 workers) │───►│  vLLM (Qwen3-8B-AWQ)                   │
│  • Lê chunk do Milvus         │    │  • Gera context_header                 │
│  • Chama vLLM via HTTP        │    │  • Gera thesis_text                    │
│  • Dispara embed_and_store    │    │  • Gera synthetic_questions            │
│                               │    │                                        │
│  Fila: embed_store (2 workers)│───►│  BGE-M3                                │
│  • Recebe enrichment          │    │  • Gera embeddings do enriched_text    │
│  • Chama BGE-M3 via HTTP      │    │  • Retorna dense + sparse vectors      │
│  • Atualiza chunk no Milvus   │    │                                        │
└───────────────────────────────┘    └────────────────────────────────────────┘
```

### Pipeline de Acordãos (Enrichment Durante Ingestão)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         GPU Server (pipeline.py)                             │
│                                                                              │
│  PDF → Docling → AcordaoParser → AcordaoChunker                              │
│                                       │                                      │
│                                       ▼                                      │
│                          ┌────────────────────────┐                          │
│                          │  Enrichment (se ativo) │                          │
│                          │                        │                          │
│                          │  vLLM (Qwen3-8B-AWQ)   │                          │
│                          │  • context_header      │                          │
│                          │  • thesis_text         │                          │
│                          │  • synthetic_questions │                          │
│                          └────────────────────────┘                          │
│                                       │                                      │
│                                       ▼                                      │
│                          BGE-M3 (Embeddings)                                 │
│                                       │                                      │
└───────────────────────────────────────┼──────────────────────────────────────┘
                                        │ Chunks (JÁ enriquecidos)
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              VPS                                             │
│  1. Insere chunks no Milvus (já com enriched_text)                           │
│  2. Cria nodes/edges no Neo4j                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Módulos de Enrichment

| Módulo | Localização | Descrição |
|--------|-------------|-----------|
| ChunkEnricher | `src/enrichment/chunk_enricher.py` | Classe principal de enriquecimento |
| Celery App | `src/enrichment/celery_app.py` | Configuração Celery (broker Redis) |
| Tasks | `src/enrichment/tasks.py` | Tasks `enrich_chunk_llm` e `embed_and_store` |
| Prompts | `src/chunking/enrichment_prompts.py` | Prompts para geração de contexto |

### Parâmetro `skip_enrichment`

```python
# No endpoint /ingest (router.py)
skip_enrichment: bool = Form(False, description="Pular enriquecimento LLM")

# Efeito por tipo de documento:
# - Acordãos: Se True, pula enrichment no pipeline (pode enriquecer depois via Celery)
# - Normas: Não afeta (enrichment sempre via Celery após indexação)
```

### Onde o Trabalho GPU Acontece

**Importante**: Independente de onde está a orquestração, o trabalho pesado SEMPRE acontece no GPU Server:

| Operação | Orquestrador | Executor (GPU) |
|----------|--------------|----------------|
| LLM (gerar contexto) | VPS Celery ou GPU pipeline | vLLM no RunPod |
| Embeddings | VPS Celery ou GPU pipeline | BGE-M3 no RunPod |

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

Response:
{
  "success": true,
  "document_id": "IN-65-2021",
  "status": "COMPLETED",
  "total_chunks": 47,
  "phases": [...],
  "chunks": [...],
  "document_hash": "abc123..."
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
  "docling": {"status": "online", "warmed_up": true},
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
VPS ──► POST /ingest (PDF)
            │
            ▼
        ┌───────────────────────────────────────────────────────┐
        │                  GPU Server                            │
        │                                                        │
        │  1. Docling ──► Markdown                               │
        │       │                                                │
        │       ▼                                                │
        │  2. SpanParser ──► ParsedDocument (spans)              │
        │       │                                                │
        │       ▼                                                │
        │  3. ArticleOrchestrator ──► ArticleChunks              │
        │       │       │                                        │
        │       │       └──► vLLM (Qwen 8B)                      │
        │       │              │                                 │
        │       │              ▼                                 │
        │       │           ArticleSpans JSON                    │
        │       ▼                                                │
        │  4. ChunkMaterializer ──► MaterializedChunks           │
        │       │                                                │
        │       ▼                                                │
        │  5. BGE-M3 ──► Embeddings (dense + sparse)             │
        │                                                        │
        └───────────────────────────────────────────────────────┘
            │
            ▼
        IngestResponse (chunks com embeddings) ──► VPS
            │
            ▼
        VPS insere no Milvus
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
| **rag-gpu-server** | Processamento GPU, ingestão, embeddings | FastAPI, BGE-M3, Docling, Pipeline |
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
| `VLLM_BASE_URL` | http://localhost:8001/v1 | URL do vLLM |
| `VLLM_MODEL` | Qwen/Qwen3-8B-AWQ | Modelo LLM |
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
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "Qwen/Qwen3-8B-AWQ"
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
- [Docling (IBM)](https://github.com/DS4SD/docling)
- [vLLM](https://docs.vllm.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
