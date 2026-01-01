#!/usr/bin/env bash
# =============================================================================
# Startup Script - RAG GPU Server (Google Cloud VM)
# =============================================================================
# Este script é executado pelo startup-script da VM (como root)
# Configurar via: --metadata-from-file=startup-script=./startup.sh
#
# Requisitos:
#   - DLVM (Deep Learning VM) com PyTorch + CUDA
#   - GPU NVIDIA L4 ou superior
# =============================================================================

set -euo pipefail

echo "=== RAG GPU Server Startup ==="
echo "Data: $(date)"
echo "Hostname: $(hostname)"

# Evitar prompts interativos do apt
export DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 1. Atualizar sistema (DLVM já vem com CUDA/drivers)
# -----------------------------------------------------------------------------
echo ">>> Atualizando sistema..."
apt-get update -y
apt-get upgrade -y
apt-get install -y git python3-venv curl

# -----------------------------------------------------------------------------
# 2. Configurar GPU Persistence Mode (reduz latência pós-restart)
# -----------------------------------------------------------------------------
echo ">>> Configurando GPU persistence mode..."
nvidia-smi -pm 1 || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# -----------------------------------------------------------------------------
# 3. (Opcional) Instalar Cloud Ops Agent para logs/métricas
# -----------------------------------------------------------------------------
echo ">>> Configurando Cloud Logging..."
if ! command -v google-cloud-ops-agent &> /dev/null; then
    curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
    bash add-google-cloud-ops-agent-repo.sh --also-install || echo "Ops Agent já instalado ou falhou"
    rm -f add-google-cloud-ops-agent-repo.sh
fi

# -----------------------------------------------------------------------------
# 4. Criar usuário de aplicação (segurança)
# -----------------------------------------------------------------------------
APP_USER="ragapp"
APP_DIR="/srv/app"
ENV_FILE="/etc/default/rag-gpu"

echo ">>> Configurando usuário: $APP_USER"
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --create-home --shell /bin/bash "$APP_USER"
fi

mkdir -p "$APP_DIR"
chown "$APP_USER":"$APP_USER" "$APP_DIR"

# -----------------------------------------------------------------------------
# 5. Criar arquivo de configuração (variáveis de ambiente)
# -----------------------------------------------------------------------------
echo ">>> Criando arquivo de configuração: $ENV_FILE"
cat > "$ENV_FILE" <<'ENVFILE'
# =============================================================================
# RAG GPU Server - Configuração de Ambiente
# =============================================================================
# Este arquivo é carregado pelo systemd via EnvironmentFile
# Edite conforme necessário e reinicie: systemctl restart rag-gpu

# Diretório da aplicação
APP_DIR=/srv/app

# HuggingFace cache (modelos baixados)
HF_HOME=/srv/app/.cache/huggingface
HF_HUB_CACHE=/srv/app/.cache/huggingface/hub

# Modelos (opcional, para referência)
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Server
HOST=0.0.0.0
PORT=8000

# GPU (opcional)
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO

# =============================================================================
# Adicione variáveis sensíveis abaixo (não comite no git!)
# =============================================================================
# MILVUS_URI=tcp://10.0.0.5:19530
# OPENAI_API_KEY=sk-xxx
# HF_TOKEN=hf_xxx
ENVFILE

chmod 600 "$ENV_FILE"
echo ">>> Arquivo de configuração criado. Edite conforme necessário."

# -----------------------------------------------------------------------------
# 6. Clonar/atualizar repositório
# -----------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/euteajudo/rag-gpu-server.git}"
echo ">>> Clonando repositório: $REPO_URL"

cd "$APP_DIR"

if [ -d ".git" ]; then
    echo ">>> Repositório existe, atualizando..."
    su - "$APP_USER" -c "cd $APP_DIR && git fetch origin && git reset --hard origin/main"
else
    echo ">>> Clonando repositório..."
    rm -rf "$APP_DIR"/*
    su - "$APP_USER" -c "git clone $REPO_URL $APP_DIR"
fi

chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

# -----------------------------------------------------------------------------
# 7. Configurar ambiente virtual
# -----------------------------------------------------------------------------
echo ">>> Configurando ambiente Python..."

su - "$APP_USER" -c "cd $APP_DIR && python3 -m venv .venv"
su - "$APP_USER" -c "cd $APP_DIR && source .venv/bin/activate && pip install --upgrade pip wheel"

# -----------------------------------------------------------------------------
# 8. Instalar dependências
# -----------------------------------------------------------------------------
echo ">>> Instalando dependências..."
su - "$APP_USER" -c "cd $APP_DIR && source .venv/bin/activate && pip install -r requirements.txt"

# -----------------------------------------------------------------------------
# 9. Criar diretório de cache para modelos
# -----------------------------------------------------------------------------
echo ">>> Configurando cache de modelos..."
mkdir -p "$APP_DIR/.cache/huggingface"
chown -R "$APP_USER":"$APP_USER" "$APP_DIR/.cache"

# -----------------------------------------------------------------------------
# 10. Pré-baixar modelos (warmup)
# -----------------------------------------------------------------------------
echo ">>> Baixando modelos (pode demorar na primeira vez)..."
if [ -f "$APP_DIR/scripts/warmup.py" ]; then
    su - "$APP_USER" -c "cd $APP_DIR && source .venv/bin/activate && \
        HF_HOME=$APP_DIR/.cache/huggingface python scripts/warmup.py"
fi

# -----------------------------------------------------------------------------
# 11. Criar/atualizar serviço systemd
# -----------------------------------------------------------------------------
echo ">>> Configurando serviço systemd..."

cat > /etc/systemd/system/rag-gpu.service <<EOF
[Unit]
Description=RAG GPU Server - Embeddings & Reranking
Documentation=https://github.com/euteajudo/rag-gpu-server
After=network.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR

# Carregar variáveis de ambiente do arquivo
EnvironmentFile=$ENV_FILE

# PATH inclui o venv
Environment="PATH=$APP_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin"

# Comando de execução
ExecStart=$APP_DIR/.venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Restart automático
Restart=always
RestartSec=3

# Limites de recursos (ajuste conforme necessário)
LimitNOFILE=65535

# Segurança
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$APP_DIR

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable rag-gpu

# -----------------------------------------------------------------------------
# 12. Iniciar serviço
# -----------------------------------------------------------------------------
echo ">>> Iniciando serviço..."
systemctl restart rag-gpu

# Aguardar um pouco para o serviço iniciar
sleep 5

# -----------------------------------------------------------------------------
# 13. Verificar status
# -----------------------------------------------------------------------------
echo ""
echo "=== Startup Completo! ==="
echo ""
echo "Status do serviço:"
systemctl status rag-gpu --no-pager || true
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo ""
echo "Comandos úteis:"
echo "  - Status:  systemctl status rag-gpu"
echo "  - Logs:    journalctl -u rag-gpu -f"
echo "  - Config:  vim $ENV_FILE"
echo "  - Health:  curl -s localhost:8000/health | jq"
echo ""
