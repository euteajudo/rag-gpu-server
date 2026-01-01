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

### Pré-requisitos

- Google Cloud SDK instalado (`gcloud`)
- Projeto GCP com billing habilitado
- Quota de GPU aprovada (solicitar em IAM & Admin > Quotas)

### 1. Configurar Variáveis

```bash
# Configurar projeto e região
PROJECT_ID="gen-lang-client-0386547606"  # Seu project ID
REGION="us-central1"
ZONE="us-central1-a"  # ou -c onde L4 esteja disponível
SA_EMAIL="sa-vectorgov-gpu@${PROJECT_ID}.iam.gserviceaccount.com"
```

### 2. Criar Service Account (opcional, recomendado)

```bash
# Criar Service Account
gcloud iam service-accounts create sa-vectorgov-gpu \
    --project="$PROJECT_ID" \
    --display-name="VectorGov GPU Server"

# Adicionar permissões mínimas
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/monitoring.metricWriter"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectViewer"
```

### 3. Criar VM para Testes (On-Demand - Recomendado para Desenvolvimento)

```bash
# VM on-demand: fica rodando continuamente, sem risco de preempção
# Custo maior (~$0.70/h) mas ideal para testes

gcloud compute instances create vectorgov-gpu-test \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="g2-standard-4" \
    --accelerator="count=1,type=nvidia-l4" \
    --maintenance-policy="TERMINATE" \
    --image-family="pytorch-latest-gpu" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-type="pd-ssd" \
    --boot-disk-size="120GB" \
    --network-interface="network=default,network-tier=PREMIUM,stack-type=IPV4_ONLY" \
    --service-account="${SA_EMAIL}" \
    --scopes="https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/devstorage.read_only" \
    --metadata=enable-guest-attributes=true \
    --metadata-from-file=startup-script=./scripts/startup.sh,shutdown-script=./scripts/shutdown.sh \
    --tags="http-server,https-server"
```

> **Nota**: Esta VM fica rodando 24/7 até ser manualmente parada. Ideal para testes e desenvolvimento.

### 4. (Alternativa) Criar Instance Template para Produção (Spot)

```bash
# Template Spot: mais barato (~$0.28/h) mas pode ser interrompido
# Usar apenas em produção com auto-restart configurado

gcloud beta compute instance-templates create vectorgov-gpu-prod \
    --project="${PROJECT_ID}" \
    --instance-template-region="${REGION}" \
    --machine-type="g2-standard-4" \
    --accelerator="count=1,type=nvidia-l4" \
    --maintenance-policy="TERMINATE" \
    --provisioning-model="SPOT" \
    --instance-termination-action="STOP" \
    --image-family="pytorch-latest-gpu" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-type="pd-ssd" \
    --boot-disk-size="120GB" \
    --network-interface="network=default,network-tier=PREMIUM,stack-type=IPV4_ONLY" \
    --no-enable-display-device \
    --service-account="${SA_EMAIL}" \
    --scopes="https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/devstorage.read_only" \
    --metadata=enable-guest-attributes=true \
    --metadata-from-file=startup-script=./scripts/startup.sh,shutdown-script=./scripts/shutdown.sh \
    --tags="http-server,https-server"

# Criar instância a partir do template (produção)
gcloud compute instances create vectorgov-gpu1 \
    --source-instance-template="projects/${PROJECT_ID}/regions/${REGION}/instanceTemplates/vectorgov-gpu-prod" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}"
```

### 5. Verificar Deploy

```bash
# SSH na VM
gcloud compute ssh vectorgov-gpu1 --zone="$ZONE" --project="$PROJECT_ID"

# Na VM:
sudo systemctl status rag-gpu
sudo journalctl -u rag-gpu -f
curl -s localhost:8000/health | jq

# Verificar GPU
nvidia-smi
```

### 6. Configurar Firewall (se necessário)

```bash
# Permitir tráfego HTTP na porta 8000
gcloud compute firewall-rules create allow-rag-gpu \
    --project="$PROJECT_ID" \
    --allow=tcp:8000 \
    --target-tags=http-server \
    --description="Allow RAG GPU Server traffic"
```

### 7. Obter IP Externo

```bash
gcloud compute instances describe vectorgov-gpu1 \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

### Configuração Pós-Deploy

O arquivo de configuração fica em `/etc/default/rag-gpu`. Para adicionar variáveis sensíveis:

```bash
# SSH na VM
gcloud compute ssh vectorgov-gpu1 --zone="$ZONE"

# Editar configuração
sudo vim /etc/default/rag-gpu

# Adicionar no final:
# HF_TOKEN=hf_xxx
# MILVUS_URI=tcp://10.0.0.5:19530

# Reiniciar serviço
sudo systemctl restart rag-gpu
```

### Custos

| Modo | Configuração | Preço/hora | Preço/mês (24/7) | Uso |
|------|--------------|------------|------------------|-----|
| **Teste** | g2-standard-4 + L4 (On-demand) | ~$0.70 | ~$504 | Desenvolvimento, testes |
| **Produção** | g2-standard-4 + L4 (Spot) | ~$0.28 | ~$200 | Produção com auto-restart |

**Recomendação**:
- **Testes**: Use On-demand (seção 3). VM fica rodando sem interrupções.
- **Produção**: Use Spot (seção 4) + monitoramento para restart automático.

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
