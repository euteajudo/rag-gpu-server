#!/usr/bin/env bash
# =============================================================================
# Shutdown Script - RAG GPU Server
# =============================================================================
# Este script é executado antes da VM ser desligada
# Configurar em: Compute Engine > VM > Editar > Metadados > shutdown-script

set -euo pipefail

echo "=== RAG GPU Server Shutdown ==="
echo "Data: $(date)"

# Para o serviço graciosamente
if systemctl is-active --quiet rag-gpu; then
    echo ">>> Parando serviço rag-gpu..."
    sudo systemctl stop rag-gpu
    echo ">>> Serviço parado."
fi

# Flush de logs (opcional)
sync

echo "=== Shutdown completo! ==="
exit 0
