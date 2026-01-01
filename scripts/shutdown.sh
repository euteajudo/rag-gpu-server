#!/usr/bin/env bash
# =============================================================================
# Shutdown Script - RAG GPU Server (Google Cloud VM)
# =============================================================================
# Este script é executado antes da VM ser desligada/preemptada
# Configurar via: --metadata-from-file=shutdown-script=./shutdown.sh
#
# Importante para Spot/Preemptible VMs:
#   - Script tem 30 segundos para executar antes de ser SIGKILL
#   - Salva estado importante antes do desligamento
# =============================================================================

set -euo pipefail

echo "=== RAG GPU Server Shutdown ==="
echo "Data: $(date)"
echo "Hostname: $(hostname)"
echo "Motivo: ${PREEMPTED:-manual/scheduled}"

# -----------------------------------------------------------------------------
# 1. Parar serviço graciosamente
# -----------------------------------------------------------------------------
if systemctl is-active --quiet rag-gpu; then
    echo ">>> Parando serviço rag-gpu..."
    systemctl stop rag-gpu --no-block

    # Aguardar até 10 segundos para parada graciosa
    timeout 10 bash -c 'while systemctl is-active --quiet rag-gpu; do sleep 1; done' || true
    echo ">>> Serviço parado."
fi

# -----------------------------------------------------------------------------
# 2. Log de estatísticas finais
# -----------------------------------------------------------------------------
echo ">>> Estatísticas finais:"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "GPU não disponível"

echo ""
echo "Uptime:"
uptime

echo ""
echo "Uso de disco:"
df -h /srv/app 2>/dev/null || true

# -----------------------------------------------------------------------------
# 3. Flush de logs e cache
# -----------------------------------------------------------------------------
echo ">>> Sincronizando filesystem..."
sync

# Flush journald
journalctl --flush 2>/dev/null || true

echo ""
echo "=== Shutdown completo! ==="
echo "Tempo: $(date)"
exit 0
