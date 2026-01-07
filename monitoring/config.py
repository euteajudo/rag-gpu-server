"""
Configurações centralizadas para o Dashboard de Monitoramento.

Este arquivo contém todas as configurações de infraestrutura.
Modifique APENAS este arquivo quando a infraestrutura mudar.

Uso:
    from config import VM_IP, VPS_IP, SSH_USER
"""

import os

# ============================================
# Provedor GPU Atual (runpod ou gcloud)
# ============================================
GPU_PROVIDER = os.getenv("GPU_PROVIDER", "runpod")

# ============================================
# RunPod - GPU A40 46GB (ATIVO)
# ============================================
RUNPOD_IP = os.getenv("RUNPOD_IP", "194.68.245.138")
RUNPOD_SSH_PORT = int(os.getenv("RUNPOD_SSH_PORT", "22181"))
RUNPOD_SSH_USER = os.getenv("RUNPOD_SSH_USER", "root")
RUNPOD_SSH_KEY = os.getenv("RUNPOD_SSH_KEY", "~/.ssh/id_ed25519")
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "wbfsmrch8rtgc8-644111ef")

# ============================================
# Google Cloud - VM GPU (backup/inativo)
# ============================================
GCLOUD_IP = os.getenv("GCLOUD_VM_IP", "34.26.181.117")
GCLOUD_VM_NAME = os.getenv("GCLOUD_VM_NAME", "vectorgov-gpu")
GCLOUD_ZONE = os.getenv("GCLOUD_VM_ZONE", "us-east1-b")
GCLOUD_PROJECT = os.getenv("GCLOUD_VM_PROJECT", "gen-lang-client-0386547606")
GCLOUD_SSH_USER = os.getenv("GCLOUD_SSH_USER", "abimael")
GCLOUD_SSH_KEY = os.getenv("GCLOUD_SSH_KEY", "~/.ssh/google_compute_engine")

# ============================================
# Configuração ativa baseada no provedor
# ============================================
if GPU_PROVIDER == "runpod":
    VM_IP = RUNPOD_IP
    SSH_USER = RUNPOD_SSH_USER
    SSH_KEY = RUNPOD_SSH_KEY
    SSH_PORT = RUNPOD_SSH_PORT
    VM_NAME = f"RunPod {RUNPOD_POD_ID[:8]}"
    ZONE = "RunPod Cloud"
    PROJECT = "RunPod"
else:
    VM_IP = GCLOUD_IP
    SSH_USER = GCLOUD_SSH_USER
    SSH_KEY = GCLOUD_SSH_KEY
    SSH_PORT = 22
    VM_NAME = GCLOUD_VM_NAME
    ZONE = GCLOUD_ZONE
    PROJECT = GCLOUD_PROJECT

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
