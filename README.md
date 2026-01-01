# RAG GPU Server

Servidor GPU para embeddings (BGE-M3) e reranking (BGE-Reranker) do sistema RAG Legal.

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                 Google Cloud VM (GPU)                       │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │             RAG GPU Server (FastAPI)                │   │
│   │                                                     │   │
│   │   POST /embed   →  BGE-M3 (embeddings)              │   │
│   │   POST /rerank  →  BGE-Reranker (cross-encoder)     │   │
│   │   GET  /health  →  Status dos modelos               │   │
│   │                                                     │   │
│   │   GPU: NVIDIA L4 (24GB) ou T4 (16GB)                │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                vLLM (container)                     │   │
│   │                                                     │   │
│   │   /v1/chat/completions  →  Qwen3-8B-AWQ             │   │
│   │   Porta: 8080                                       │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     VPS Hostinger                           │
│   FastAPI (RAG API) + Milvus + Redis + PostgreSQL           │
└─────────────────────────────────────────────────────────────┘
```

## Endpoints

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/embed` | POST | Gera embeddings dense (1024d) + sparse |
| `/rerank` | POST | Reordena documentos por relevância |
| `/health` | GET | Health check completo |
| `/healthz` | GET | Liveness probe (Kubernetes) |
| `/readyz` | GET | Readiness probe (Kubernetes) |
| `/docs` | GET | Swagger UI |

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

## Deploy no Google Cloud

### 1. Criar VM com GPU

```bash
# Criar VM com GPU L4
gcloud compute instances create rag-gpu-server \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --metadata-from-file=startup-script=scripts/startup.sh,shutdown-script=scripts/shutdown.sh
```

### 2. Variáveis de Ambiente

Configurar na VM ou no metadata:

```bash
export REPO_URL=https://github.com/seu-usuario/rag-gpu-server.git
export EMBEDDING_MODEL=BAAI/bge-m3
export RERANKER_MODEL=BAAI/bge-reranker-v2-m3
export PORT=8000
export DEVICE=cuda
export USE_FP16=true
```

### 3. Verificar Deploy

```bash
# SSH na VM
gcloud compute ssh rag-gpu-server --zone=us-central1-a

# Verificar serviço
sudo systemctl status rag-gpu

# Ver logs
sudo journalctl -u rag-gpu -f

# Testar API
curl http://localhost:8000/health
```

## Desenvolvimento Local

### Com GPU

```bash
# Clonar
git clone https://github.com/seu-usuario/rag-gpu-server.git
cd rag-gpu-server

# Ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Rodar
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Com Docker

```bash
# Build
docker build -t rag-gpu-server .

# Run (com GPU)
docker run --gpus all -p 8000:8000 rag-gpu-server
```

## vLLM (Separado)

O vLLM roda como container separado:

```bash
docker run -d --gpus all \
    --name vllm \
    -p 8080:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-8B-AWQ \
    --max-model-len 16000 \
    --gpu-memory-utilization 0.5
```

## Modelos

| Modelo | Uso | VRAM |
|--------|-----|------|
| BAAI/bge-m3 | Embeddings (dense + sparse) | ~2GB |
| BAAI/bge-reranker-v2-m3 | Reranking cross-encoder | ~2GB |
| Qwen/Qwen3-8B-AWQ (vLLM) | Geração de texto | ~6GB |

**Total VRAM**: ~10GB (cabe em T4 16GB ou L4 24GB)

## Integração com VPS

No arquivo `.env` da VPS:

```bash
GPU_SERVER_URL=http://<IP-DA-VM-GPU>:8000
VLLM_BASE_URL=http://<IP-DA-VM-GPU>:8080/v1
```

## Custos Estimados

| GPU | Preço/hora | Preço/mês (24/7) |
|-----|------------|------------------|
| NVIDIA T4 | ~$0.35 | ~$252 |
| NVIDIA L4 | ~$0.70 | ~$504 |
| NVIDIA A100 | ~$3.00 | ~$2,160 |

**Recomendação**: Usar VM preemptível ($0.10/h com T4) + restart automático.

## Estrutura do Repositório

```
rag-gpu-server/
├── src/
│   ├── __init__.py
│   ├── config.py       # Configurações
│   ├── embedder.py     # BGE-M3 wrapper
│   ├── reranker.py     # BGE-Reranker wrapper
│   └── main.py         # FastAPI app
├── scripts/
│   ├── startup.sh      # Script de inicialização da VM
│   ├── shutdown.sh     # Script de desligamento da VM
│   └── warmup.py       # Pré-carrega modelos
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Licença

Apache 2.0
