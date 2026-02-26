# Qwen3-VL-8B-Instruct — Configuração de Referência

Documento para restaurar a configuração caso o Qwen3.5-27B-FP8 não atenda.

## Modelo

- **Nome**: Qwen3-VL-8B-Instruct
- **HuggingFace**: `Qwen/Qwen3-VL-8B-Instruct`
- **Arquitetura**: Qwen3VLForConditionalGeneration
- **Tipo**: qwen3_vl (visão + texto)
- **Parâmetros**: 8B
- **Precisão**: Full precision (BF16, sem quantização)
- **Tamanho em disco**: 17GB (11 safetensors)
- **Context window**: 262,144 tokens (configurado com limite de 16K no vLLM)
- **Hidden size**: 4096
- **Layers**: 36
- **Attention heads**: 32
- **Vocab size**: 151,936

## vLLM

- **Versão**: 0.15.1
- **Porta**: 8002
- **GPU**: NVIDIA A40 (48GB VRAM)

## Comando de Inicialização do vLLM

```bash
HF_HOME="/workspace/.cache/huggingface" \
/workspace/venv/bin/python -u -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen3-VL-8B-Instruct \
    --port 8002 \
    --max-model-len 16000 \
    --gpu-memory-utilization 0.80 \
    --trust-remote-code
```

## Parâmetros vLLM

| Parâmetro | Valor | Motivo |
|---|---|---|
| `--port` | 8002 | Porta padrão do vLLM no pod |
| `--max-model-len` | 16000 | Limita context para economizar VRAM |
| `--gpu-memory-utilization` | 0.80 | 80% da VRAM (38.4GB de 48GB) |
| `--trust-remote-code` | sim | Necessário para modelos Qwen |

## Variáveis de Ambiente

No `.env` do rag-gpu-server:
```
VLLM_MODEL=/workspace/models/Qwen3-VL-8B-Instruct
```

No startup do GPU server:
```
VLLM_BASE_URL=http://localhost:8002/v1
```

## Integração com GPU Server

- O FastAPI GPU Server (porta 8000) conecta ao vLLM via `VLLM_BASE_URL`
- Usado para OCR de PDFs (pipeline VLM)
- Chat frontend conecta via túnel Cloudflare (quick tunnel temporário)
- API de ingestão conecta via `gpu.vectorgov.io` (tunnel nomeado permanente)

## Como Restaurar

```bash
# 1. Baixar o modelo novamente
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir /workspace/models/Qwen3-VL-8B-Instruct

# 2. Reverter vLLM para 0.15.1 (se necessário)
/workspace/venv/bin/pip install vllm==0.15.1

# 3. Reiniciar tudo
bash /workspace/start_all.sh
```

## Scripts de Startup

- `/workspace/start_all.sh` — Sobe Redis + Cloudflare + vLLM + GPU Server
- `/workspace/start_all.sh --stop` — Para tudo
- `/workspace/start_all.sh --status` — Verifica status
- `/workspace/restart_services.sh` — Restart dos serviços

## Timestamp

- Data de backup: 2026-02-25
- Motivo: Teste do Qwen3.5-27B-FP8 como substituto
