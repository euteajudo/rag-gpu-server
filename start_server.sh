#!/bin/bash
# Script de inicializacao do GPU Server
# Uso: ./start_server.sh
#
# Este script configura automaticamente:
# - GPU_API_KEYS: Chave de API para VPS
# - VLLM_BASE_URL: URL do servidor vLLM
# - VLLM_MODEL: Nome do modelo (auto-detectado do vLLM)
# - DISABLE_DOCS: Se "true", desabilita /docs, /redoc, /openapi.json (seguranca)

cd /workspace/rag-gpu-server

# Mata processo anterior se existir
pkill -f "uvicorn src.main" 2>/dev/null
sleep 2

# ============================================================
# CONFIGURACOES - Edite aqui se necessario
# ============================================================
export GPU_API_KEYS="${GPU_API_KEYS:-vg_gpu_internal_2025}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8002/v1}"

# Seguranca: desabilitar documentacao em producao
# export DISABLE_DOCS=true  # Descomente para desabilitar /docs, /redoc

# Auto-detecta o modelo do vLLM se nao estiver definido
if [ -z "$VLLM_MODEL" ]; then
    echo "Detectando modelo do vLLM..."
    DETECTED_MODEL=$(curl -s http://localhost:8002/v1/models 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data.get('data') else '')" 2>/dev/null)

    if [ -n "$DETECTED_MODEL" ]; then
        export VLLM_MODEL="$DETECTED_MODEL"
        echo "Modelo detectado: $VLLM_MODEL"
    else
        export VLLM_MODEL="Qwen/Qwen3-8B-AWQ"
        echo "AVISO: Nao foi possivel detectar modelo, usando default: $VLLM_MODEL"
    fi
else
    echo "Usando VLLM_MODEL definido: $VLLM_MODEL"
fi

echo ""
echo "=== Iniciando GPU Server ==="
echo "GPU_API_KEYS: $GPU_API_KEYS"
echo "VLLM_BASE_URL: $VLLM_BASE_URL"
echo "VLLM_MODEL: $VLLM_MODEL"
echo "Log: /workspace/gpu-server.log"
echo ""

# Inicia o servidor
nohup /workspace/venv/bin/python -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    >> /workspace/gpu-server.log 2>&1 &

SERVER_PID=$!
echo "Servidor iniciado (PID: $SERVER_PID)"
echo "Aguardando warmup (pode levar ~30s)..."
sleep 30

# Verifica saude
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo ""
    echo "=== Servidor SAUDAVEL ==="
    curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/health
else
    echo ""
    echo "=== ERRO: Servidor nao respondeu ==="
    echo "Verificando se o processo ainda esta rodando..."
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "Processo ainda rodando (PID: $SERVER_PID)"
        echo "Pode precisar de mais tempo para warmup. Aguarde e execute:"
        echo "  curl http://localhost:8000/health"
    else
        echo "Processo NAO esta rodando!"
    fi
    echo ""
    echo "Ultimas linhas do log:"
    tail -30 /workspace/gpu-server.log
fi
