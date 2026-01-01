# =============================================================================
# Dockerfile - RAG GPU Server
# =============================================================================
# Build: docker build -t rag-gpu-server .
# Run:   docker run --gpus all -p 8000:8000 rag-gpu-server

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Alternativa: use python3.11 como padrão
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Workdir
WORKDIR /app

# Cache de modelos
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
RUN mkdir -p /root/.cache/huggingface

# Dependências primeiro (cache de layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Código
COPY src/ ./src/

# Variáveis de ambiente
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEVICE=cuda
ENV USE_FP16=true

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Porta
EXPOSE 8000

# Comando
CMD ["python3", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
