#!/bin/bash
# =============================================================================
# RunPod Startup Script - VectorGov RAG GPU Server
# =============================================================================
# Este script deve ser executado no startup do pod RunPod
# Ele configura o virtualenv, instala dependências e inicia os serviços
#
# Uso:
#   bash /workspace/rag-gpu-server/scripts/runpod_startup.sh
#
# Para adicionar como startup script no RunPod:
# 1. Vá em Pod Settings > Edit Template
# 2. Em "Start Command", adicione: bash /workspace/rag-gpu-server/scripts/runpod_startup.sh
# =============================================================================

set -e

echo "========================================"
echo "VectorGov RAG GPU Server - Startup"
echo "========================================"
echo "Data: $(date)"
echo ""

# Diretórios
WORKSPACE=/workspace
VENV=$WORKSPACE/venv
LOGS=$WORKSPACE/logs
MODELS=$WORKSPACE/models
CACHE=$WORKSPACE/cache
RAG_SERVER=$WORKSPACE/rag-gpu-server

# Cria estrutura
mkdir -p $LOGS $MODELS $CACHE

# Configura HF_HOME para cache persistente
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export TORCH_HOME=$CACHE

# ============================================
# 1. Configura virtualenv (se não existir)
# ============================================
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[1/6] Criando virtualenv em $VENV..."
    python3 -m venv $VENV
else
    echo "[1/6] Virtualenv já existe"
fi

source $VENV/bin/activate
pip install --quiet --upgrade pip

# ============================================
# 2. Instala dependências
# ============================================
echo "[2/6] Instalando dependências (pode demorar na primeira vez)..."
if [ -f "$RAG_SERVER/requirements.txt" ]; then
    pip install --quiet -r $RAG_SERVER/requirements.txt
else
    echo "  AVISO: requirements.txt não encontrado, instalando pacotes básicos..."
    pip install --quiet vllm huggingface_hub FlagEmbedding fastapi uvicorn python-dotenv
    pip install --quiet docling docling-core docling-ibm-models docling-parse
    pip install --quiet transformers accelerate safetensors sentence-transformers
    pip install --quiet httpx pydantic pydantic-settings openai loguru tqdm rich
fi

# ============================================
# 3. Clona/atualiza rag-gpu-server
# ============================================
if [ ! -d "$RAG_SERVER" ]; then
    echo "[3/6] Clonando rag-gpu-server..."
    git clone https://github.com/euteajudo/rag-gpu-server.git $RAG_SERVER
else
    echo "[3/6] rag-gpu-server já existe (use 'git pull' para atualizar)"
fi

# ============================================
# 4. Baixa modelos (se necessário)
# ============================================
echo "[4/6] Verificando modelos..."

# 4.1 Qwen3-8B-AWQ para vLLM
MODEL_PATH=$MODELS/models--Qwen--Qwen3-8B-AWQ
if [ ! -d "$MODEL_PATH" ]; then
    echo "  - Baixando Qwen3-8B-AWQ (~5.8GB)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B-AWQ', cache_dir='$MODELS')"
else
    echo "  - Qwen3-8B-AWQ: OK"
fi

# 4.2 BGE-M3 para embeddings
BGE_M3_PATH=$CACHE/models--BAAI--bge-m3
if [ ! -d "$BGE_M3_PATH" ]; then
    echo "  - Baixando BGE-M3 (~2.5GB)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', cache_dir='$CACHE')"
else
    echo "  - BGE-M3: OK"
fi

# 4.3 BGE-Reranker para reranking
RERANKER_PATH=$CACHE/models--BAAI--bge-reranker-v2-m3
if [ ! -d "$RERANKER_PATH" ]; then
    echo "  - Baixando BGE-Reranker-v2-m3 (~2.2GB)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', cache_dir='$CACHE')"
else
    echo "  - BGE-Reranker: OK"
fi

# Encontra snapshot path do Qwen
SNAPSHOT_PATH=$(ls -d $MODEL_PATH/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$SNAPSHOT_PATH" ]; then
    echo "ERRO: Modelo Qwen não encontrado!"
    exit 1
fi

# ============================================
# 5. Cria arquivo de configuração
# ============================================
echo "[5/6] Criando configuração..."

# gpu-server.env com variáveis necessárias
cat > $WORKSPACE/gpu-server.env << EOF
# VectorGov GPU Server Configuration
VLLM_MODEL=Qwen/Qwen3-8B-AWQ
VLLM_BASE_URL=http://localhost:8001/v1
HF_HOME=$CACHE
TRANSFORMERS_CACHE=$CACHE
GPU_API_KEYS=vg_gpu_internal_2025
EOF

echo "  - Configuração salva em $WORKSPACE/gpu-server.env"

# ============================================
# 6. Inicia serviços
# ============================================
echo "[6/6] Iniciando serviços..."

# Para serviços existentes
pkill -f "vllm" 2>/dev/null || true
pkill -f "uvicorn.*main:app" 2>/dev/null || true
sleep 2

# Inicia vLLM (porta 8001)
# IMPORTANTE: gpu-memory-utilization deve ser <= 0.65 para deixar espaço
# para BGE-M3 (~2.5GB), Reranker (~2.5GB) e Docling (~1GB)
# A40 48GB: 0.65 * 48 = 31.2GB para vLLM, ~12GB para outros modelos
echo "  - Iniciando vLLM (porta 8001, 8K contexto, 65% GPU)..."
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model $SNAPSHOT_PATH \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.65 \
    --enable-prefix-caching \
    > $LOGS/vllm.log 2>&1 &

# Aguarda vLLM iniciar
echo "  - Aguardando vLLM carregar modelo (pode demorar 1-2 min)..."
for i in {1..60}; do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "  - vLLM pronto!"
        break
    fi
    sleep 5
done

# Inicia GPU Server (porta 8000)
echo "  - Iniciando GPU Server (porta 8000)..."
cd $RAG_SERVER
export $(cat $WORKSPACE/gpu-server.env | xargs)
nohup python3 -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    > $LOGS/gpu_server.log 2>&1 &

# Aguarda GPU Server
echo "  - Aguardando GPU Server carregar modelos (BGE-M3, Reranker)..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  - GPU Server pronto!"
        break
    fi
    sleep 5
done

# ============================================
# Verifica status
# ============================================
echo ""
echo "========================================"
echo "Status dos serviços:"
echo "========================================"

if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "  ✓ vLLM (8001): OK"
else
    echo "  ✗ vLLM (8001): AGUARDANDO (verificar $LOGS/vllm.log)"
fi

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✓ GPU Server (8000): OK"
else
    echo "  ✗ GPU Server (8000): AGUARDANDO (verificar $LOGS/gpu_server.log)"
fi

echo ""
echo "Logs disponíveis em:"
echo "  - vLLM: tail -f $LOGS/vllm.log"
echo "  - GPU Server: tail -f $LOGS/gpu_server.log"
echo ""
echo "Para testar:"
echo "  - vLLM: curl http://localhost:8001/v1/models"
echo "  - GPU Server: curl http://localhost:8000/health"
echo ""
echo "========================================"
echo "Startup completo!"
echo "========================================"
