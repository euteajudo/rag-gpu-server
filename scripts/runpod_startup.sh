#!/bin/bash
# =============================================================================
# RunPod Startup Script - VectorGov RAG GPU Server
# =============================================================================
# Este script deve ser executado no startup do pod RunPod
# Ele configura o virtualenv, instala dependências e inicia os serviços
#
# Para adicionar como startup script no RunPod:
# 1. Vá em Pod Settings > Edit Template
# 2. Em "Start Command", adicione: bash /workspace/startup.sh
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

# ============================================
# 1. Configura virtualenv (se não existir)
# ============================================
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[1/5] Criando virtualenv em $VENV..."
    python3 -m venv $VENV
else
    echo "[1/5] Virtualenv já existe"
fi

source $VENV/bin/activate

# ============================================
# 2. Instala dependências
# ============================================
echo "[2/5] Instalando dependências..."
pip install --quiet --upgrade pip
pip install --quiet vllm huggingface_hub FlagEmbedding fastapi uvicorn python-dotenv

# ============================================
# 3. Clona/atualiza rag-gpu-server
# ============================================
if [ ! -d "$RAG_SERVER" ]; then
    echo "[3/5] Clonando rag-gpu-server..."
    git clone https://github.com/euteajudo/rag-gpu-server.git $RAG_SERVER
else
    echo "[3/5] Atualizando rag-gpu-server..."
    cd $RAG_SERVER && git pull
fi

# ============================================
# 4. Baixa modelo Qwen (se necessário)
# ============================================
MODEL_PATH=$MODELS/models--Qwen--Qwen3-8B-AWQ
if [ ! -d "$MODEL_PATH" ]; then
    echo "[4/5] Baixando modelo Qwen3-8B-AWQ..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B-AWQ', cache_dir='$MODELS')"
else
    echo "[4/5] Modelo já existe"
fi

# Encontra snapshot path
SNAPSHOT_PATH=$(ls -d $MODEL_PATH/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$SNAPSHOT_PATH" ]; then
    echo "ERRO: Modelo não encontrado!"
    exit 1
fi

# ============================================
# 5. Inicia serviços
# ============================================
echo "[5/5] Iniciando serviços..."

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
echo "  - Aguardando vLLM carregar modelo..."
sleep 30

# Inicia GPU Server (porta 8000)
echo "  - Iniciando GPU Server (porta 8000)..."
cd $RAG_SERVER
nohup python3 -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    > $LOGS/gpu_server.log 2>&1 &

# Aguarda GPU Server
sleep 15

# ============================================
# Verifica status
# ============================================
echo ""
echo "========================================"
echo "Status dos serviços:"
echo "========================================"

if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "  vLLM (8001): OK"
else
    echo "  vLLM (8001): AGUARDANDO (verificar logs)"
fi

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  GPU Server (8000): OK"
else
    echo "  GPU Server (8000): AGUARDANDO (verificar logs)"
fi

echo ""
echo "Logs disponíveis em:"
echo "  - vLLM: $LOGS/vllm.log"
echo "  - GPU Server: $LOGS/gpu_server.log"
echo ""
echo "========================================"
echo "Startup completo!"
echo "========================================"

# Mantém container rodando
tail -f /dev/null
