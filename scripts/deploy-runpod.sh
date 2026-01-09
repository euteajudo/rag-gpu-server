#!/bin/bash
# =============================================================================
# Deploy script for RAG GPU Server on RunPod
# =============================================================================
# Uso: ./deploy-runpod.sh [branch]
# Default branch: feature/security-and-dependencies

set -e

BRANCH="${1:-feature/security-and-dependencies}"
WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/rag-gpu-server"
LOG_FILE="/var/log/gpu-server.log"
PIP_CACHE="$WORKSPACE/.pip_cache"

echo "=================================================="
echo "Deploying RAG GPU Server"
echo "Branch: $BRANCH"
echo "=================================================="

# 1. Configurar pip para usar cache no workspace (disco persistente)
export PIP_CACHE_DIR="$PIP_CACHE"
mkdir -p "$PIP_CACHE"

# 2. Clone ou update repositório
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "Atualizando repositório existente..."
    cd "$PROJECT_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "Clonando repositório..."
    cd "$WORKSPACE"
    git clone -b "$BRANCH" https://github.com/euteajudo/rag-gpu-server.git
    cd "$PROJECT_DIR"
fi

# 3. Instalar dependências do PyTorch (versões específicas)
echo "Instalando PyTorch (versão compatível)..."
pip install --cache-dir "$PIP_CACHE" \
    torch==2.4.1+cu124 \
    torchvision==0.19.1+cu124 \
    torchaudio==2.4.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 \
    -q

# 4. Instalar outras dependências
echo "Instalando outras dependências..."
pip install --cache-dir "$PIP_CACHE" \
    fastapi uvicorn pydantic \
    FlagEmbedding transformers sentence-transformers \
    numpy \
    -q

# 5. Parar servidor existente (se houver)
echo "Parando servidor existente..."
pkill -f "uvicorn src.main:app" 2>/dev/null || true
sleep 2

# 6. Iniciar servidor
echo "Iniciando GPU Server..."
cd "$PROJECT_DIR"
nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "GPU Server iniciado com PID: $SERVER_PID"

# 7. Aguardar inicialização e verificar
echo "Aguardando inicialização (30s)..."
sleep 30

if ps -p $SERVER_PID > /dev/null; then
    echo "=================================================="
    echo "Deploy concluído com sucesso!"
    echo "GPU Server rodando em http://localhost:8000"
    echo "Log: $LOG_FILE"
    echo "=================================================="
    echo "Últimas linhas do log:"
    tail -20 "$LOG_FILE"
else
    echo "ERRO: GPU Server não iniciou corretamente"
    echo "Log de erro:"
    cat "$LOG_FILE"
    exit 1
fi
