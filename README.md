# RAG GPU Server

Servidor GPU para embeddings (BGE-M3), reranking (BGE-Reranker) e LLM (vLLM) do sistema RAG Legal.

**Status Atual**: Produção - VM Google Cloud operacional

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GOOGLE CLOUD VM (GPU)                                    │
│                    IP: 34.44.157.159                                        │
│                    Tipo: g2-standard-4 + NVIDIA L4 (24GB)                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                 RAG GPU Server (FastAPI)                            │   │
│   │                 Porta: 8000                                         │   │
│   │                                                                     │   │
│   │   POST /embed   →  BGE-M3 (embeddings dense 1024d + sparse)         │   │
│   │   POST /rerank  →  BGE-Reranker-v2-m3 (cross-encoder)               │   │
│   │   GET  /health  →  Status dos modelos + GPU                         │   │
│   │                                                                     │   │
│   │   Modelos carregados na GPU (~4GB VRAM)                             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                 vLLM Container                                      │   │
│   │                 Porta: 8001 (interno) → exposto como 8001           │   │
│   │                                                                     │   │
│   │   POST /v1/chat/completions  →  Qwen/Qwen3-8B-AWQ                   │   │
│   │                                                                     │   │
│   │   Modelo AWQ quantizado (~6GB VRAM)                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTPS / HTTP
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VPS HOSTINGER                                       │
│                         IP: 77.37.43.160                                    │
│                                                                             │
│   ┌─────────────────────┐  ┌─────────────┐  ┌─────────────┐                │
│   │   FastAPI (RAG API) │  │   Milvus    │  │   Redis     │                │
│   │   Porta: 8000       │  │   19530     │  │   6379      │                │
│   │                     │  │             │  │             │                │
│   │   RemoteEmbedder    │  │  leis_v3    │  │  Cache      │                │
│   │   RemoteReranker    │  │  (1312 docs)│  │  Semântico  │                │
│   │   RemoteLLM         │  │             │  │             │                │
│   └─────────────────────┘  └─────────────┘  └─────────────┘                │
│                                                                             │
│   ┌─────────────────────┐                                                  │
│   │   PostgreSQL        │                                                  │
│   │   Porta: 5432       │                                                  │
│   │   (Usuários, Logs)  │                                                  │
│   └─────────────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Endpoints do GPU Server

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/embed` | POST | Gera embeddings dense (1024d) + sparse |
| `/rerank` | POST | Reordena documentos por relevância |
| `/ingest` | POST | **NOVO** Inicia processamento async de PDF |
| `/ingest/status/{task_id}` | GET | **NOVO** Retorna progresso do processamento |
| `/ingest/result/{task_id}` | GET | **NOVO** Retorna chunks quando completo |
| `/ingest/health` | GET | **NOVO** Health check do módulo de ingestão |
| `/health` | GET | Health check completo com métricas GPU |
| `/healthz` | GET | Liveness probe (Kubernetes) |
| `/readyz` | GET | Readiness probe (Kubernetes) |
| `/docs` | GET | Swagger UI interativo |

### POST /embed

```json
// Request
{
    "texts": ["Texto 1", "Texto 2"],
    "return_dense": true,
    "return_sparse": true
}

// Response
{
    "dense_embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "sparse_embeddings": [{"123": 0.5, "456": 0.3}, ...],
    "latency_ms": 45.2,
    "count": 2
}
```

### POST /rerank

```json
// Request
{
    "query": "O que é ETP?",
    "documents": ["Doc 1", "Doc 2", "Doc 3"],
    "top_k": 3
}

// Response
{
    "scores": [0.95, 0.72, 0.45],
    "rankings": [0, 1, 2],
    "latency_ms": 120.5
}
```

### POST /ingest (Async - Novo em 20/01/2025)

Processa PDF de forma assíncrona para evitar timeout do Cloudflare (100s).

```json
// Request (multipart/form-data)
{
    "file": "<PDF binary>",
    "document_id": "LEI-14133-2021",
    "tipo_documento": "LEI",
    "numero": "14133",
    "ano": 2021,
    "skip_embeddings": false,
    "max_articles": null
}

// Response (retorna imediatamente ~1s)
{
    "task_id": "abc123def456",
    "document_id": "LEI-14133-2021",
    "message": "Processamento iniciado em background. Use GET /ingest/status/{task_id} para acompanhar."
}
```

### GET /ingest/status/{task_id}

Retorna o progresso do processamento.

```json
// Response (durante processamento)
{
    "task_id": "abc123def456",
    "document_id": "LEI-14133-2021",
    "status": "processing",
    "progress": 0.45,
    "current_phase": "extraction",
    "started_at": "2025-01-20T10:30:00Z",
    "completed_at": null,
    "error_message": null
}

// Response (quando completo)
{
    "task_id": "abc123def456",
    "document_id": "LEI-14133-2021",
    "status": "completed",
    "progress": 1.0,
    "current_phase": "completed",
    "started_at": "2025-01-20T10:30:00Z",
    "completed_at": "2025-01-20T10:38:00Z",
    "error_message": null
}
```

### GET /ingest/result/{task_id}

Retorna os chunks processados (só disponível quando status == "completed").

```json
// Response
{
    "success": true,
    "document_id": "LEI-14133-2021",
    "status": "completed",
    "total_chunks": 498,
    "phases": [
        {"phase": "docling", "duration_seconds": 15.2},
        {"phase": "parsing", "duration_seconds": 0.5},
        {"phase": "extraction", "duration_seconds": 120.3},
        {"phase": "materialization", "duration_seconds": 2.1},
        {"phase": "embedding", "duration_seconds": 45.8}
    ],
    "errors": [],
    "total_time_seconds": 183.9,
    "chunks": [
        {
            "chunk_id": "LEI-14133-2021#ART-001",
            "parent_chunk_id": "",
            "span_id": "ART-001",
            "device_type": "article",
            "text": "Art. 1º Esta Lei estabelece...",
            "dense_vector": [0.1, 0.2, ...],
            "sparse_vector": {"123": 0.5, ...}
        }
    ],
    "document_hash": "sha256:abc123..."
}
```

### Fases de Processamento

| Fase | Progress | Descrição |
|------|----------|-----------|
| `queued` | 0.0 | Aguardando início |
| `initializing` | 0.05 | Iniciando pipeline |
| `docling` | 0.1-0.3 | Convertendo PDF → Markdown |
| `parsing` | 0.3-0.4 | SpanParser extraindo spans |
| `extraction` | 0.4-0.7 | ArticleOrchestrator (LLM) |
| `materialization` | 0.7-0.8 | ChunkMaterializer |
| `embedding` | 0.8-0.95 | BGE-M3 embeddings |
| `completed` | 1.0 | Finalizado com sucesso |

### Fluxo de Polling Recomendado

```python
import time
import requests

# 1. Iniciar processamento
response = requests.post(
    "https://gpu.vectorgov.io/ingest",
    files={"file": open("lei.pdf", "rb")},
    data={"document_id": "LEI-14133-2021", "tipo_documento": "LEI", "numero": "14133", "ano": 2021}
)
task_id = response.json()["task_id"]

# 2. Polling até completar
while True:
    status = requests.get(f"https://gpu.vectorgov.io/ingest/status/{task_id}").json()

    print(f"Progress: {status['progress']*100:.1f}% - {status['current_phase']}")

    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        raise Exception(status["error_message"])

    time.sleep(3)  # Poll a cada 3 segundos

# 3. Obter resultado
result = requests.get(f"https://gpu.vectorgov.io/ingest/result/{task_id}").json()
print(f"Total chunks: {result['total_chunks']}")
```

---

## Infraestrutura Atual

### VM Google Cloud (GPU Server)

| Propriedade | Valor |
|-------------|-------|
| **Nome** | vectorgov-gpu-test |
| **IP Externo** | 34.44.157.159 |
| **Zona** | us-central1-c |
| **Tipo** | g2-standard-4 |
| **GPU** | NVIDIA L4 (24GB VRAM) |
| **CPU** | 4 vCPUs |
| **RAM** | 16 GB |
| **Disco** | 120 GB SSD |
| **OS** | Debian 11 (Deep Learning VM) |
| **Custo** | ~$0.70/hora (on-demand) |

### Serviços Rodando na VM

| Serviço | Porta | Status | Comando de Verificação |
|---------|-------|--------|------------------------|
| GPU Server (FastAPI) | 8000 | Ativo | `curl http://34.44.157.159:8000/health` |
| vLLM | 8001 | Ativo | `curl http://34.44.157.159:8001/health` |

### VPS Hostinger

| Propriedade | Valor |
|-------------|-------|
| **IP** | 77.37.43.160 |
| **RAM** | 8 GB |
| **Disco** | 200 GB NVMe |
| **OS** | Ubuntu 22.04 |

---

## Configuração do Sistema

### Variáveis de Ambiente na VPS

```bash
# /opt/rag-api/.env

# GPU Server (Google Cloud)
GPU_SERVER_URL=http://34.44.157.159:8000

# vLLM (Google Cloud)
VLLM_BASE_URL=http://34.44.157.159:8001/v1
VLLM_MODEL=Qwen/Qwen3-8B-AWQ

# Milvus (local na VPS)
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# Redis (local na VPS)
REDIS_HOST=127.0.0.1
REDIS_PORT=6379

# PostgreSQL (local na VPS)
DATABASE_URL=postgresql://rag:xxx@localhost:5432/rag_db

# JWT
JWT_SECRET_KEY=xxx
```

### Firewall do Google Cloud

Regras criadas para permitir acesso:

```bash
# Porta 8000 - GPU Server
gcloud compute firewall-rules create allow-rag-gpu \
    --allow=tcp:8000 \
    --target-tags=http-server \
    --source-ranges=0.0.0.0/0

# Porta 8001 - vLLM
gcloud compute firewall-rules create allow-vllm \
    --allow=tcp:8001 \
    --target-tags=http-server \
    --source-ranges=0.0.0.0/0
```

---

## Arquivos Modificados na VPS

### 1. model_pool.py - Modo Remoto

**Arquivo**: `/opt/rag-api/src/model_pool.py`

Adicionado suporte para alternar entre modo local (GPU na máquina) e remoto (GPU no Google Cloud):

```python
import os
from typing import Optional

GPU_SERVER_URL = os.getenv("GPU_SERVER_URL")

_embedder = None
_reranker = None


def is_remote_mode() -> bool:
    """Verifica se está usando GPU remota."""
    return GPU_SERVER_URL is not None


def get_embedder():
    """Retorna embedder (remoto ou local)."""
    global _embedder

    if is_remote_mode():
        if _embedder is None:
            from remote import RemoteEmbedder
            from remote.embedder import RemoteEmbedderConfig
            config = RemoteEmbedderConfig(gpu_server_url=GPU_SERVER_URL)
            _embedder = RemoteEmbedder(config=config)
        return _embedder

    # Fallback local (requer GPU)
    if _embedder is None:
        from embeddings import BGEM3Embedder
        _embedder = BGEM3Embedder()
    return _embedder


def get_reranker():
    """Retorna reranker (remoto ou local)."""
    global _reranker

    if is_remote_mode():
        if _reranker is None:
            from remote import RemoteReranker
            from remote.reranker import RemoteRerankerConfig
            config = RemoteRerankerConfig(gpu_server_url=GPU_SERVER_URL)
            _reranker = RemoteReranker(config=config)
        return _reranker

    # Fallback local (requer GPU)
    if _reranker is None:
        from embeddings import BGEReranker
        _reranker = BGEReranker()
    return _reranker
```

### 2. RemoteEmbedder - Cliente HTTP

**Arquivo**: `/opt/rag-api/src/remote/embedder.py`

Cliente que chama o GPU Server para gerar embeddings:

```python
@dataclass
class RemoteEmbedderConfig:
    gpu_server_url: str = "http://localhost:8000"
    timeout: int = 30
    max_batch_size: int = 32


class RemoteEmbedder:
    """Cliente para embeddings remotos via GPU Server."""

    def __init__(self, config: Optional[RemoteEmbedderConfig] = None):
        self.config = config or RemoteEmbedderConfig()
        self.embed_url = f"{self.config.gpu_server_url}/embed"

    def encode(
        self,
        texts: list[str],
        return_dense: bool = True,
        return_sparse: bool = True
    ) -> EmbeddingResult:
        """Gera embeddings via GPU Server."""
        response = requests.post(
            self.embed_url,
            json={
                "texts": texts,
                "return_dense": return_dense,
                "return_sparse": return_sparse,
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return EmbeddingResult(
            dense_embeddings=data.get("dense_embeddings", []),
            sparse_embeddings=data.get("sparse_embeddings", []),
        )

    def encode_hybrid(self, texts: list[str]) -> dict:
        """Retorna embeddings no formato esperado pelo HybridSearcher."""
        result = self.encode(texts, return_dense=True, return_sparse=True)
        return {
            "dense": result.dense_embeddings,
            "sparse": result.sparse_embeddings,
        }

    def encode_hybrid_single(self, text: str) -> dict:
        """Retorna embedding único no formato esperado pelo HybridSearcher."""
        result = self.encode([text], return_dense=True, return_sparse=True)
        return {
            "dense": result.dense_embeddings[0],
            "sparse": result.sparse_embeddings[0],
        }
```

### 3. RemoteReranker - Cliente HTTP

**Arquivo**: `/opt/rag-api/src/remote/reranker.py`

Cliente que chama o GPU Server para reranking:

```python
@dataclass
class RemoteRerankerConfig:
    gpu_server_url: str = "http://localhost:8000"
    timeout: int = 60


class RemoteReranker:
    """Cliente para reranking remoto via GPU Server."""

    def __init__(self, config: Optional[RemoteRerankerConfig] = None):
        self.config = config or RemoteRerankerConfig()
        self.rerank_url = f"{self.config.gpu_server_url}/rerank"

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        text_key: str = "text",
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Reordena documentos por relevância.

        Interface compatível com BGEReranker local.
        """
        texts = [doc.get(text_key, "") for doc in documents]

        response = requests.post(
            self.rerank_url,
            json={
                "query": query,
                "documents": texts,
                "top_k": top_k or len(documents),
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Reordena documentos conforme rankings
        rankings = data.get("rankings", list(range(len(documents))))
        scores = data.get("scores", [0.0] * len(documents))

        reranked = []
        for i, rank_idx in enumerate(rankings):
            if rank_idx < len(documents):
                doc = documents[rank_idx].copy()
                if return_scores:
                    doc["rerank_score"] = scores[i] if i < len(scores) else 0.0
                reranked.append(doc)

        return reranked[:top_k] if top_k else reranked
```

### 4. vllm_client.py - URL Configurável

**Arquivo**: `/opt/rag-api/src/llm/vllm_client.py`

Modificado para ler URL do vLLM de variável de ambiente:

```python
import os
from dataclasses import dataclass, field

@dataclass
class LLMConfig:
    model: str = field(
        default_factory=lambda: os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    temperature: float = 0.7
    max_tokens: int = 2048
```

### 5. answer_generator.py - URL Configurável

**Arquivo**: `/opt/rag-api/src/rag/answer_generator.py`

Modificado para ler configurações de variáveis de ambiente:

```python
import os
from dataclasses import dataclass, field

@dataclass
class GenerationConfig:
    model: str = field(
        default_factory=lambda: os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    # ... outros campos
```

### 6. search/config.py - Milvus Host Configurável

**Arquivo**: `/opt/rag-api/src/search/config.py`

Modificado para ler host do Milvus de variável de ambiente:

```python
import os
from dataclasses import dataclass, field

@dataclass
class SearchConfig:
    milvus_host: str = field(
        default_factory=lambda: os.getenv("MILVUS_HOST", "localhost")
    )
    milvus_port: int = 19530
    collection_name: str = "leis_v3"
```

---

## Módulo de Clientes Remotos

Estrutura do módulo `/opt/rag-api/src/remote/`:

```
src/remote/
├── __init__.py         # Exports: RemoteEmbedder, RemoteReranker, RemoteLLM
├── embedder.py         # RemoteEmbedder, RemoteEmbedderConfig, EmbeddingResult
├── reranker.py         # RemoteReranker, RemoteRerankerConfig, RerankResult
└── llm.py              # RemoteLLM, RemoteLLMConfig (wrapper para vLLM remoto)
```

### __init__.py

```python
from .embedder import RemoteEmbedder, RemoteEmbedderConfig, EmbeddingResult
from .reranker import RemoteReranker, RemoteRerankerConfig, RerankResult
from .llm import RemoteLLM, RemoteLLMConfig

__all__ = [
    "RemoteEmbedder",
    "RemoteEmbedderConfig",
    "EmbeddingResult",
    "RemoteReranker",
    "RemoteRerankerConfig",
    "RerankResult",
    "RemoteLLM",
    "RemoteLLMConfig",
]
```

---

## Dashboard de Monitoramento

### Localização

**Arquivo**: `/Users/abimaeltorcate/vector_govi_2/rag-gpu-server/monitoring/dashboard.py`

### Funcionalidades

- **CPU**: Utilização, load average (1/5/15 min)
- **Memória RAM**: Total, usado, livre, percentual
- **GPU NVIDIA**: Utilização, VRAM, temperatura, consumo de energia
- **Disco**: Espaço total, usado, livre
- **I/O**: Leituras/escritas totais, operações
- **Rede**: Bytes recebidos/enviados
- **Status dos Serviços**: GPU Server (8000), vLLM (8001)
- **Histórico**: Gráficos temporais das métricas
- **Auto-refresh**: Atualização automática configurável

### Como Executar

```bash
cd /Users/abimaeltorcate/vector_govi_2/rag-gpu-server
streamlit run monitoring/dashboard.py --server.port 8502
```

Acesse: http://localhost:8502

### Configuração

O dashboard se conecta via SSH à VM do Google Cloud:

```python
VM_IP = "34.44.157.159"
SSH_USER = "abimaeltorcate"
SSH_KEY = "~/.ssh/google_compute_engine"
```

### Requisitos

```bash
pip install streamlit pandas
```

---

## Comandos Úteis

### Conectar à VM via SSH

```bash
# Via gcloud
gcloud compute ssh vectorgov-gpu-test --zone=us-central1-c

# Via SSH direto
ssh -i ~/.ssh/google_compute_engine abimaeltorcate@34.44.157.159
```

### Verificar Status dos Serviços

```bash
# Na VM
sudo systemctl status rag-gpu
sudo systemctl status vllm

# Remotamente
curl http://34.44.157.159:8000/health
curl http://34.44.157.159:8001/health
```

### Logs dos Serviços

```bash
# GPU Server
sudo journalctl -u rag-gpu -f

# vLLM
docker logs -f vllm
```

### Verificar GPU

```bash
# Na VM
nvidia-smi

# Uso contínuo
watch -n 1 nvidia-smi
```

### Testar Endpoints

```bash
# Health check
curl http://34.44.157.159:8000/health | jq

# Embeddings
curl -X POST http://34.44.157.159:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Teste de embedding"], "return_dense": true, "return_sparse": true}'

# Reranking
curl -X POST http://34.44.157.159:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "teste", "documents": ["doc1", "doc2"], "top_k": 2}'

# vLLM Chat
curl -X POST http://34.44.157.159:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [{"role": "user", "content": "Olá"}],
    "max_tokens": 100
  }'
```

### Gerenciar VM

```bash
# Parar VM (economiza custos)
gcloud compute instances stop vectorgov-gpu-test --zone=us-central1-c

# Iniciar VM
gcloud compute instances start vectorgov-gpu-test --zone=us-central1-c

# Ver IP atual
gcloud compute instances describe vectorgov-gpu-test \
  --zone=us-central1-c \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

---

## Custos e Economia

### Custo Atual (On-Demand)

| Componente | Custo/Hora | Custo/Dia | Custo/Mês |
|------------|------------|-----------|-----------|
| g2-standard-4 + L4 | $0.70 | $16.80 | ~$504 |
| Disco 120GB SSD | $0.02 | $0.48 | ~$14 |
| Rede (estimado) | $0.05 | $1.20 | ~$36 |
| **Total** | **$0.77** | **$18.48** | **~$554** |

### Estratégias de Economia

1. **Desligar quando não usar**:
   ```bash
   gcloud compute instances stop vectorgov-gpu-test --zone=us-central1-c
   ```
   - Economia: 100% do custo de compute (ainda paga disco)

2. **Usar Spot VMs** (produção):
   - Custo: ~$0.28/hora (60% desconto)
   - Risco: VM pode ser interrompida

3. **Escalar verticalmente**:
   - T4 (16GB): ~$0.35/hora
   - L4 (24GB): ~$0.70/hora
   - Usar T4 se VRAM for suficiente

---

## Troubleshooting

### Problema: "Connection refused" na porta 8000

**Causa**: GPU Server não está rodando

**Solução**:
```bash
# Verificar status
sudo systemctl status rag-gpu

# Reiniciar
sudo systemctl restart rag-gpu

# Ver logs
sudo journalctl -u rag-gpu -f
```

### Problema: "CUDA out of memory"

**Causa**: GPU sem memória suficiente

**Solução**:
```bash
# Verificar uso de memória
nvidia-smi

# Reiniciar vLLM com menos memória
docker stop vllm
docker run -d --gpus all --name vllm \
  -p 8001:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B-AWQ \
  --max-model-len 8000 \
  --gpu-memory-utilization 0.4
```

### Problema: Latência alta nos embeddings

**Causa**: Cold start dos modelos

**Solução**:
```bash
# Warmup inicial
curl -X POST http://34.44.157.159:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["warmup"], "return_dense": true}'
```

### Problema: SSH timeout

**Causa**: VM pode estar parada ou firewall

**Solução**:
```bash
# Verificar se VM está rodando
gcloud compute instances list --filter="name=vectorgov-gpu-test"

# Iniciar se necessário
gcloud compute instances start vectorgov-gpu-test --zone=us-central1-c
```

---

## Estrutura do Código (2 Repositórios)

O projeto está dividido em **2 repositórios** que correspondem à separação da infraestrutura:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REPOSITÓRIO 1: rag-gpu-server                    │
│                    (Este repositório)                               │
│                                                                     │
│   Função: Servidor de inferência GPU                                │
│   Deploy: Google Cloud VM (34.44.157.159)                           │
│   Caminho na VM: /srv/app/                                          │
│                                                                     │
│   rag-gpu-server/                                                   │
│   ├── src/                                                          │
│   │   ├── main.py              # FastAPI app                        │
│   │   ├── embedder.py          # BGE-M3 wrapper                     │
│   │   ├── reranker.py          # BGE-Reranker wrapper               │
│   │   ├── config.py            # Configurações                      │
│   │   ├── ingestion/                                                │
│   │   │   ├── pipeline.py      # Pipeline de ingestão               │
│   │   │   ├── models.py        # Modelos Pydantic                   │
│   │   │   └── markdown_sanitizer.py  # Limpeza de markdown          │
│   │   ├── parsing/                                                  │
│   │   │   ├── span_parser.py   # Parser de estrutura legal          │
│   │   │   ├── span_models.py   # Modelos de spans                   │
│   │   │   └── article_orchestrator.py  # Extração LLM               │
│   │   ├── chunking/                                                 │
│   │   │   ├── chunk_materializer.py  # Materialização + split       │
│   │   │   └── citation_extractor.py  # Extração de citações         │
│   │   └── llm/                                                      │
│   │       └── vllm_client.py   # Cliente vLLM                       │
│   ├── monitoring/                                                   │
│   │   └── dashboard.py         # Dashboard Streamlit (roda local)   │
│   ├── scripts/                                                      │
│   │   └── warmup.py            # Pré-carrega modelos                │
│   └── requirements.txt                                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    REPOSITÓRIO 2: vector_govi_2/extracao            │
│                    (Repositório principal)                          │
│                                                                     │
│   Função: API RAG + Pipeline de Extração                            │
│   Deploy: VPS Hostinger (77.37.43.160)                              │
│   Caminho na VPS: /opt/rag-api/                                     │
│                                                                     │
│   extracao/                                                         │
│   ├── src/api/              # FastAPI endpoints (/ask, /search)     │
│   ├── src/rag/              # Answer Generator + Citações           │
│   ├── src/search/           # Busca híbrida no Milvus               │
│   ├── src/remote/           # Clientes remotos (→ GPU Server)       │
│   ├── src/cache/            # Cache semântico                       │
│   ├── src/parsing/          # SpanParser, Docling                   │
│   ├── src/chunking/         # ChunkMaterializer                     │
│   ├── src/enrichment/       # Enriquecimento LLM                    │
│   └── src/dashboard/        # Dashboard Streamlit (local)           │
└─────────────────────────────────────────────────────────────────────┘
```

**Por que 2 repositórios?**

| Aspecto | rag-gpu-server | extracao |
|---------|----------------|----------|
| **Hardware** | GPU (L4 24GB) | CPU only |
| **Dependências** | FlagEmbedding, PyTorch CUDA | Requests, Milvus client |
| **Tamanho** | ~10GB (modelos) | ~500MB |
| **Escala** | Vertical (GPU maior) | Horizontal (mais workers) |
| **Custo** | $0.70/hora | $20/mês fixo |

---

## Configuração Manual na VM (Não está no Git)

As seguintes configurações foram feitas **manualmente** na VM e não estão versionadas:

### 1. Container vLLM (Docker)

```bash
# Baixar imagem vLLM
docker pull vllm/vllm-openai:latest

# Criar container (modelo baixa automaticamente ~5.7GB)
docker run -d \
    --gpus all \
    --name vllm \
    -p 8001:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --restart unless-stopped \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-8B-AWQ \
    --max-model-len 16000 \
    --gpu-memory-utilization 0.5 \
    --dtype auto \
    --trust-remote-code
```

### 2. Serviço Systemd (rag-gpu.service)

```bash
# /etc/systemd/system/rag-gpu.service
[Unit]
Description=RAG GPU Server - Embeddings & Reranking
After=network.target

[Service]
Type=simple
User=ragapp
Group=ragapp
WorkingDirectory=/srv/app
EnvironmentFile=/etc/default/rag-gpu
Environment="PATH=/srv/app/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/srv/app/.venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 3. Usuário e Diretórios

```bash
# Criar usuário
sudo useradd -r -s /bin/bash -m -d /srv/app ragapp

# Clonar repositório
sudo -u ragapp git clone <repo> /srv/app

# Criar venv e instalar
sudo -u ragapp python3 -m venv /srv/app/.venv
sudo -u ragapp /srv/app/.venv/bin/pip install -r /srv/app/requirements.txt
```

### 4. Modelos Baixados Automaticamente

| Modelo | Tamanho | Local |
|--------|---------|-------|
| BAAI/bge-m3 | ~2GB | ~/.cache/huggingface (GPU Server) |
| BAAI/bge-reranker-v2-m3 | ~2GB | ~/.cache/huggingface (GPU Server) |
| Qwen/Qwen3-8B-AWQ | ~5.7GB | ~/.cache/huggingface (vLLM container) |

---

## Próximos Passos (TODO)

- [ ] Configurar HTTPS com Let's Encrypt
- [ ] Adicionar autenticação (API Key)
- [ ] Implementar auto-scaling
- [ ] Configurar alertas de monitoramento
- [ ] Backup automático dos modelos
- [ ] Documentar API com OpenAPI completo

---

## Histórico de Mudanças

### 2025-01-27

- **SpanParser - Prefixos Numéricos do Docling**
  - Regex atualizado para aceitar "11. Art. 56" (listas numeradas)
  - Limpeza automática dos prefixos no texto final
  - Corrige ART-056 e ART-057 que não eram detectados na Lei 14.133

- **VLLMClient - Fallback para JSON Inválido**
  - Captura `JSONDecodeError` e retorna `{}` ao invés de crash
  - Tenta extrair JSON válido da resposta truncada
  - Evita perda de artigos grandes como ART-006 (~23k chars)

- **Pipeline - Filtro `_skip_milvus_index`**
  - Adiciona verificação em `_phase_embeddings()`
  - Adiciona verificação em `_to_processed_chunks()`
  - Parents de artigos splitados não são mais indexados

- **CitationExtractor - Remoção de Parent-Loops**
  - `normalize_citations()` aceita `parent_chunk_id` e `document_type`
  - Remove citações onde filho cita o pai (ex: ART-006-P1 → ART-006)
  - Mantém citações externas válidas

- **ChunkMaterializer - Split de Artigos Grandes**
  - Threshold de 8000 chars para split
  - Gera partes ART-XXX-P1, P2, P3...
  - Parent marcado com `_skip_milvus_index=True`

- **MarkdownSanitizer - Novo Módulo**
  - Remove anomalias do Docling (`<!-- image -->`, etc)
  - Normaliza espaços e linhas em branco

- **Testado com Lei 14.133/2021**:
  - 1299 chunks gerados
  - ART-006 splitado em 6 partes (~23k chars total)
  - ART-056 e ART-057 presentes (antes faltavam)
  - ART-075 splitado em 3 partes
  - Texto limpo sem prefixos numéricos

### 2025-01-20

- **Ingestão Async**: Implementado processamento assíncrono de PDFs
  - POST `/ingest` retorna task_id imediatamente (~1s)
  - GET `/ingest/status/{task_id}` retorna progresso (0.0 a 1.0)
  - GET `/ingest/result/{task_id}` retorna chunks quando completo
  - Background processing com `threading.Thread`
  - Progress callback para reportar fases
- **Problema resolvido**: Cloudflare timeout 524 após 100s
- **Testado**: Lei 13.303/2016 (498 chunks, ~8 min processamento)

### 2025-01-01

- Criado GPU Server e deployado no Google Cloud
- Configurada VM g2-standard-4 com NVIDIA L4
- Implementados endpoints /embed e /rerank
- Configurado vLLM com Qwen3-8B-AWQ
- Criados clientes remotos (RemoteEmbedder, RemoteReranker)
- Configurada VPS para usar GPU remota
- Criado dashboard de monitoramento Streamlit
- Documentação completa

---

## Licença

Apache 2.0
