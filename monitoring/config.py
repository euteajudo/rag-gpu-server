"""
Configurações centralizadas para o Dashboard de Monitoramento.

Este arquivo contém todas as configurações de infraestrutura.
Modifique APENAS este arquivo quando a infraestrutura mudar.

Uso:
    from config import VM_IP, VPS_IP, SSH_USER
"""

import os

# ============================================
# Google Cloud - VM GPU
# ============================================
VM_IP = os.getenv("GPU_VM_IP", "34.26.181.117")
VM_NAME = os.getenv("GPU_VM_NAME", "vectorgov-gpu")
ZONE = os.getenv("GPU_VM_ZONE", "us-east1-b")
PROJECT = os.getenv("GPU_VM_PROJECT", "gen-lang-client-0386547606")

# SSH
SSH_USER = os.getenv("GPU_SSH_USER", "abimael")
SSH_KEY = os.getenv("GPU_SSH_KEY", "~/.ssh/google_compute_engine")

# ============================================
# VPS Hostinger
# ============================================
VPS_IP = os.getenv("VPS_IP", "77.37.43.160")
VPS_SSH_KEY = os.getenv("VPS_SSH_KEY", "~/.ssh/id_rsa")

# ============================================
# Portas dos Serviços
# ============================================
GPU_SERVER_PORT = int(os.getenv("GPU_SERVER_PORT", "8000"))
VLLM_PORT = int(os.getenv("VLLM_PORT", "8001"))
RAG_API_PORT = int(os.getenv("RAG_API_PORT", "8000"))
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

# ============================================
# URLs derivadas
# ============================================
GPU_SERVER_URL = f"http://{VM_IP}:{GPU_SERVER_PORT}"
VLLM_BASE_URL = f"http://{VM_IP}:{VLLM_PORT}/v1"
