#!/usr/bin/env bash
# =============================================================================
# Startup Script - RAG GPU Server (Google Cloud VM)
# =============================================================================
# Este script é executado pelo startup-script da VM
# Configurar em: Compute Engine > VM > Editar > Metadados > startup-script

set -euo pipefail

echo "=== RAG GPU Server Startup ==="
echo "Data: $(date)"

# -----------------------------------------------------------------------------
# 1. Atualizar sistema (DLVM já vem com CUDA/drivers)
# -----------------------------------------------------------------------------
echo ">>> Atualizando sistema..."
sudo apt-get update -y
sudo apt-get install -y git python3-venv

# -----------------------------------------------------------------------------
# 2. Configurar diretório da aplicação
# -----------------------------------------------------------------------------
APP_DIR="/srv/app"
echo ">>> Configurando diretório: $APP_DIR"

sudo mkdir -p "$APP_DIR"
sudo chown "$USER":"$USER" "$APP_DIR"
cd "$APP_DIR"

# -----------------------------------------------------------------------------
# 3. Clonar/atualizar repositório
# -----------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/seu-usuario/rag-gpu-server.git}"
echo ">>> Clonando repositório: $REPO_URL"

if [ -d ".git" ]; then
    echo ">>> Repositório existe, atualizando..."
    git fetch origin
    git reset --hard origin/main
else
    echo ">>> Clonando repositório..."
    git clone "$REPO_URL" .
fi

# -----------------------------------------------------------------------------
# 4. Configurar ambiente virtual
# -----------------------------------------------------------------------------
echo ">>> Configurando ambiente Python..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip wheel

# -----------------------------------------------------------------------------
# 5. Instalar dependências
# -----------------------------------------------------------------------------
echo ">>> Instalando dependências..."
pip install -r requirements.txt

# -----------------------------------------------------------------------------
# 6. Pré-baixar modelos (warmup)
# -----------------------------------------------------------------------------
echo ">>> Baixando modelos..."
if [ -f "scripts/warmup.py" ]; then
    python scripts/warmup.py
fi

# -----------------------------------------------------------------------------
# 7. Iniciar serviço
# -----------------------------------------------------------------------------
echo ">>> Iniciando servidor..."

# Opção 1: Usando nohup (simples)
# nohup .venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > /var/log/rag-gpu.log 2>&1 &

# Opção 2: Usando systemd (recomendado)
# Criar serviço se não existir
if [ ! -f "/etc/systemd/system/rag-gpu.service" ]; then
    echo ">>> Criando serviço systemd..."
    sudo tee /etc/systemd/system/rag-gpu.service > /dev/null <<EOF
[Unit]
Description=RAG GPU Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/.venv/bin:/usr/bin"
Environment="HF_HOME=/root/.cache/huggingface"
ExecStart=$APP_DIR/.venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable rag-gpu
fi

# Iniciar serviço
sudo systemctl restart rag-gpu

echo "=== Startup completo! ==="
echo "Serviço: sudo systemctl status rag-gpu"
echo "Logs: sudo journalctl -u rag-gpu -f"
