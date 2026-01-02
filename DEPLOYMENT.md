# Guia de Implantação - RAG GPU Server

Este documento descreve o passo-a-passo completo da implantação realizada em 01/01/2025.

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Passo 1: Criar VM no Google Cloud](#passo-1-criar-vm-no-google-cloud)
3. [Passo 2: Configurar GPU Server na VM](#passo-2-configurar-gpu-server-na-vm)
4. [Passo 3: Configurar vLLM na VM](#passo-3-configurar-vllm-na-vm)
5. [Passo 4: Configurar Firewall](#passo-4-configurar-firewall)
6. [Passo 5: Criar Clientes Remotos na VPS](#passo-5-criar-clientes-remotos-na-vps)
7. [Passo 6: Configurar VPS para Usar GPU Remota](#passo-6-configurar-vps-para-usar-gpu-remota)
8. [Passo 7: Criar Dashboard de Monitoramento](#passo-7-criar-dashboard-de-monitoramento)
9. [Problemas Encontrados e Soluções](#problemas-encontrados-e-soluções)
10. [Validação Final](#validação-final)

---

## Visão Geral

### Arquitetura Antes

```
VPS Hostinger (sem GPU)
├── FastAPI + FlagEmbedding (ERRO: sem GPU)
├── Milvus
├── Redis
└── PostgreSQL
```

### Arquitetura Depois

```
Google Cloud VM (com GPU L4)          VPS Hostinger
├── GPU Server (FastAPI)         ←──  RemoteEmbedder
│   ├── BGE-M3                   ←──  RemoteReranker
│   └── BGE-Reranker             ←──  RemoteLLM
├── vLLM Container                    │
│   └── Qwen3-8B-AWQ                  ├── Milvus
└── NVIDIA L4 (24GB)                  ├── Redis
                                      └── PostgreSQL
```

---

## Passo 1: Criar VM no Google Cloud

### 1.1 Pré-requisitos

```bash
# Instalar Google Cloud SDK
brew install google-cloud-sdk

# Login
gcloud auth login

# Configurar projeto
gcloud config set project gen-lang-client-0386547606
```

### 1.2 Verificar Quota de GPU

```bash
# Verificar quota de GPUs
gcloud compute project-info describe \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" \
  | grep GPU
```

Se não tiver quota, solicitar em: IAM & Admin → Quotas → Pesquisar "NVIDIA L4" → Solicitar aumento

### 1.3 Verificar Disponibilidade de GPU

```bash
# Listar zonas com L4 disponível
gcloud compute accelerator-types list --filter="name=nvidia-l4"
```

Resultado: `us-central1-a`, `us-central1-c` têm L4

### 1.4 Criar a VM

```bash
gcloud compute instances create vectorgov-gpu-test \
    --project="gen-lang-client-0386547606" \
    --zone="us-central1-c" \
    --machine-type="g2-standard-4" \
    --accelerator="count=1,type=nvidia-l4" \
    --maintenance-policy="TERMINATE" \
    --image-family="pytorch-latest-gpu" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-type="pd-ssd" \
    --boot-disk-size="120GB" \
    --network-interface="network=default,network-tier=PREMIUM,stack-type=IPV4_ONLY" \
    --tags="http-server,https-server"
```

### 1.5 Obter IP Externo

```bash
gcloud compute instances describe vectorgov-gpu-test \
    --zone=us-central1-c \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

**Resultado**: `34.44.157.159`

---

## Passo 2: Configurar GPU Server na VM

### 2.1 Conectar via SSH

```bash
gcloud compute ssh vectorgov-gpu-test --zone=us-central1-c
```

### 2.2 Verificar GPU

```bash
nvidia-smi
```

Esperado: NVIDIA L4 com 24GB VRAM

### 2.3 Clonar Repositório

```bash
cd /opt
sudo git clone https://github.com/seu-usuario/rag-gpu-server.git
sudo chown -R $USER:$USER rag-gpu-server
cd rag-gpu-server
```

### 2.4 Criar Ambiente Virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.5 Testar Servidor

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 2.6 Criar Serviço Systemd

```bash
sudo tee /etc/systemd/system/rag-gpu.service << 'EOF'
[Unit]
Description=RAG GPU Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/rag-gpu-server
Environment="PATH=/opt/rag-gpu-server/.venv/bin"
ExecStart=/opt/rag-gpu-server/.venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag-gpu
sudo systemctl start rag-gpu
```

### 2.7 Verificar Status

```bash
sudo systemctl status rag-gpu
curl localhost:8000/health
```

---

## Passo 3: Configurar vLLM na VM

### 3.1 Instalar Docker (se não tiver)

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### 3.2 Instalar NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3.3 Criar Container vLLM

```bash
docker run -d \
    --gpus all \
    --name vllm \
    -p 8001:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --restart unless-stopped \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-8B-AWQ \
    --max-model-len 16000 \
    --gpu-memory-utilization 0.5
```

### 3.4 Verificar Container

```bash
docker logs -f vllm
# Aguardar "Uvicorn running on http://0.0.0.0:8000"

curl localhost:8001/health
```

---

## Passo 4: Configurar Firewall

### 4.1 Criar Regras de Firewall

```bash
# Porta 8000 - GPU Server
gcloud compute firewall-rules create allow-rag-gpu \
    --project="gen-lang-client-0386547606" \
    --allow=tcp:8000 \
    --target-tags=http-server \
    --source-ranges=0.0.0.0/0 \
    --description="Allow RAG GPU Server"

# Porta 8001 - vLLM
gcloud compute firewall-rules create allow-vllm \
    --project="gen-lang-client-0386547606" \
    --allow=tcp:8001 \
    --target-tags=http-server \
    --source-ranges=0.0.0.0/0 \
    --description="Allow vLLM"
```

### 4.2 Testar Acesso Externo

```bash
# Do seu computador local
curl http://34.44.157.159:8000/health
curl http://34.44.157.159:8001/health
```

---

## Passo 5: Criar Clientes Remotos na VPS

### 5.1 Criar Estrutura do Módulo

```bash
ssh root@77.37.43.160

mkdir -p /opt/rag-api/src/remote
```

### 5.2 Criar __init__.py

```python
# /opt/rag-api/src/remote/__init__.py

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

### 5.3 Criar embedder.py

```python
# /opt/rag-api/src/remote/embedder.py

import requests
from dataclasses import dataclass
from typing import Optional

@dataclass
class RemoteEmbedderConfig:
    gpu_server_url: str = "http://localhost:8000"
    timeout: int = 30
    max_batch_size: int = 32

@dataclass
class EmbeddingResult:
    dense_embeddings: list
    sparse_embeddings: list

class RemoteEmbedder:
    def __init__(self, config: Optional[RemoteEmbedderConfig] = None):
        self.config = config or RemoteEmbedderConfig()
        self.embed_url = f"{self.config.gpu_server_url}/embed"

    def encode(self, texts, return_dense=True, return_sparse=True):
        response = requests.post(
            self.embed_url,
            json={"texts": texts, "return_dense": return_dense, "return_sparse": return_sparse},
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return EmbeddingResult(
            dense_embeddings=data.get("dense_embeddings", []),
            sparse_embeddings=data.get("sparse_embeddings", []),
        )

    def encode_hybrid(self, texts):
        result = self.encode(texts, return_dense=True, return_sparse=True)
        return {"dense": result.dense_embeddings, "sparse": result.sparse_embeddings}

    def encode_hybrid_single(self, text):
        result = self.encode([text], return_dense=True, return_sparse=True)
        return {"dense": result.dense_embeddings[0], "sparse": result.sparse_embeddings[0]}
```

### 5.4 Criar reranker.py

```python
# /opt/rag-api/src/remote/reranker.py

import requests
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class RemoteRerankerConfig:
    gpu_server_url: str = "http://localhost:8000"
    timeout: int = 60

@dataclass
class RerankResult:
    scores: list
    rankings: list

class RemoteReranker:
    def __init__(self, config: Optional[RemoteRerankerConfig] = None):
        self.config = config or RemoteRerankerConfig()
        self.rerank_url = f"{self.config.gpu_server_url}/rerank"

    def rerank(self, query, documents, text_key="text", top_k=None, return_scores=True):
        texts = [doc.get(text_key, "") for doc in documents]
        response = requests.post(
            self.rerank_url,
            json={"query": query, "documents": texts, "top_k": top_k or len(documents)},
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()

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

---

## Passo 6: Configurar VPS para Usar GPU Remota

### 6.1 Atualizar .env

```bash
# /opt/rag-api/.env

GPU_SERVER_URL=http://34.44.157.159:8000
VLLM_BASE_URL=http://34.44.157.159:8001/v1
VLLM_MODEL=Qwen/Qwen3-8B-AWQ
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
```

### 6.2 Atualizar model_pool.py

```python
# /opt/rag-api/src/model_pool.py

import os

GPU_SERVER_URL = os.getenv("GPU_SERVER_URL")

_embedder = None
_reranker = None

def is_remote_mode():
    return GPU_SERVER_URL is not None

def get_embedder():
    global _embedder
    if is_remote_mode():
        if _embedder is None:
            from remote import RemoteEmbedder
            from remote.embedder import RemoteEmbedderConfig
            config = RemoteEmbedderConfig(gpu_server_url=GPU_SERVER_URL)
            _embedder = RemoteEmbedder(config=config)
        return _embedder
    # fallback local...

def get_reranker():
    global _reranker
    if is_remote_mode():
        if _reranker is None:
            from remote import RemoteReranker
            from remote.reranker import RemoteRerankerConfig
            config = RemoteRerankerConfig(gpu_server_url=GPU_SERVER_URL)
            _reranker = RemoteReranker(config=config)
        return _reranker
    # fallback local...
```

### 6.3 Atualizar Arquivos com URLs Hardcoded

Arquivos modificados:
- `src/llm/vllm_client.py`: Ler `VLLM_BASE_URL` do ambiente
- `src/rag/answer_generator.py`: Ler `VLLM_BASE_URL` e `VLLM_MODEL` do ambiente
- `src/search/config.py`: Ler `MILVUS_HOST` do ambiente

### 6.4 Reiniciar API

```bash
sudo systemctl restart rag-api
sudo journalctl -u rag-api -f
```

---

## Passo 7: Criar Dashboard de Monitoramento

### 7.1 Criar Estrutura

```bash
mkdir -p /Users/abimaeltorcate/vector_govi_2/rag-gpu-server/monitoring
```

### 7.2 Criar dashboard.py

O dashboard usa Streamlit e conecta via SSH à VM para coletar métricas:
- CPU, memória, load average
- GPU (nvidia-smi)
- Disco e I/O
- Status dos serviços

### 7.3 Executar Dashboard

```bash
cd /Users/abimaeltorcate/vector_govi_2/rag-gpu-server
streamlit run monitoring/dashboard.py --server.port 8502
```

---

## Problemas Encontrados e Soluções

### Problema 1: FlagEmbedding não instalado

**Erro**: `ModuleNotFoundError: No module named 'FlagEmbedding'`

**Causa**: VPS tentando usar embedder local sem GPU

**Solução**: Criar `model_pool.py` com modo remoto que retorna `RemoteEmbedder` quando `GPU_SERVER_URL` está definido

---

### Problema 2: encode_hybrid_single() não existe

**Erro**: `AttributeError: 'RemoteEmbedder' object has no attribute 'encode_hybrid_single'`

**Causa**: HybridSearcher esperava método que não existia

**Solução**: Adicionar método ao `RemoteEmbedder`:

```python
def encode_hybrid_single(self, text: str) -> dict:
    result = self.encode([text], return_dense=True, return_sparse=True)
    return {"dense": result.dense_embeddings[0], "sparse": result.sparse_embeddings[0]}
```

---

### Problema 3: encode_hybrid retornando keys erradas

**Erro**: `KeyError: 'dense'`

**Causa**: RemoteEmbedder retornava `dense_vecs` mas HyDE esperava `dense`

**Solução**: Corrigir método `encode_hybrid()`:

```python
def encode_hybrid(self, texts):
    result = self.encode(texts, return_dense=True, return_sparse=True)
    return {"dense": result.dense_embeddings, "sparse": result.sparse_embeddings}
```

---

### Problema 4: rerank() assinatura diferente

**Erro**: TypeError na chamada do reranker

**Causa**: `RemoteReranker.rerank()` esperava `texts: list[str]` mas código passava `documents: list[dict]`

**Solução**: Mudar assinatura para compatibilidade com BGEReranker:

```python
def rerank(self, query, documents, text_key="text", top_k=None, return_scores=True):
    texts = [doc.get(text_key, "") for doc in documents]
    # ...
```

---

### Problema 5: vLLM usando localhost

**Erro**: `Connection refused localhost:8000/v1/chat/completions`

**Causa**: `LLMConfig.base_url` hardcoded como `localhost:8000`

**Solução**: Ler de variável de ambiente:

```python
base_url: str = field(
    default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
)
```

---

### Problema 6: Milvus connection conflict

**Erro**: `Alias of 'default' already creating connections`

**Causa**: Diferentes partes do código usando `localhost` vs `127.0.0.1`

**Solução**: Padronizar usando variável de ambiente `MILVUS_HOST=127.0.0.1`

---

### Problema 7: SyntaxError em ask_service.py

**Erro**: `SyntaxError: unterminated f-string literal`

**Causa**: Comando sed corrompeu o arquivo

**Solução**: Reescrever arquivo completamente via Python

---

## Validação Final

### Testar GPU Server

```bash
curl http://34.44.157.159:8000/health
# {"status": "healthy", "gpu": "NVIDIA L4", ...}
```

### Testar vLLM

```bash
curl http://34.44.157.159:8001/health
# {"status": "ok"}
```

### Testar VPS API

```bash
curl -X POST http://77.37.43.160:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "O que é ETP?"}'
# Deve retornar resposta (mesmo que "no relevant documents" se Milvus vazio)
```

### Testar Dashboard

```bash
streamlit run monitoring/dashboard.py --server.port 8502
# Acessar http://localhost:8502
# Verificar métricas da VM
```

---

## Conclusão

A implantação foi concluída com sucesso:

1. **VM Google Cloud** criada com NVIDIA L4 (24GB VRAM)
2. **GPU Server** rodando na porta 8000 com BGE-M3 e BGE-Reranker
3. **vLLM** rodando na porta 8001 com Qwen3-8B-AWQ
4. **VPS** configurada para usar GPU remota via HTTP
5. **Dashboard** de monitoramento criado com Streamlit

### Status Final

| Componente | Status | Endpoint |
|------------|--------|----------|
| GPU Server | Ativo | http://34.44.157.159:8000 |
| vLLM | Ativo | http://34.44.157.159:8001 |
| VPS API | Ativo | http://77.37.43.160:8000 |
| Dashboard | Local | http://localhost:8502 |

### Próximo Passo

Popular o Milvus com documentos para que o RAG funcione completamente. Atualmente as collections estão vazias (0 entities).
