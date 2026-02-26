# Resumo da Sessao — 2026-02-26

## 1. Correcao do Bug de Ingestao (Lei 14.133)

- **Problema**: Pipeline de ingestao produzia 0 chunks de 74 paginas. O `regex_classifier.py` classificava todos os blocos como "metadata" por causa da fonte Arial antes de verificar dispositivos legais.
- **Fix**: `git pull` trouxe correcao que reordena classificacao (dispositivos ANTES de metadata) e remove deteccao por fonte.
- **Testes**: 396/396 testes passando. Servidor reiniciado com codigo atualizado.

## 2. Migracao de Modelo: Qwen3-VL-8B -> Qwen3.5-27B-AWQ

### Tentativa 1: FP8
- Baixado `Qwen/Qwen3.5-27B-FP8` (29GB)
- Atualizado vLLM de 0.15.1 para **0.16.0rc2 nightly** (necessario para arquitetura Qwen3.5)
- **Problemas**: A40 nao tem hardware FP8 nativo -> emulacao via Marlin, ~38.4GB VRAM, ~14.9 tok/s
- Bug na camada Mamba exigiu `--enforce-eager` (sem CUDA graphs)

### Tentativa 2: AWQ (solucao final)
- Baixado `cyankiwi/Qwen3.5-27B-AWQ-4bit` (20GB)
- Deletado o modelo FP8 para liberar espaco
- **Resultado**: INT4 nativo na A40, **18.65GB VRAM** (metade do FP8), **~14.9 tok/s**
- Thinking mode desabilitado por padrao no `chat_template.jinja`

### Comparativo Final

| | Qwen3-VL-8B (anterior) | Qwen3.5-27B FP8 | Qwen3.5-27B AWQ (atual) |
|---|---|---|---|
| Parametros | 8B | 27B | 27B |
| Quantizacao | Nenhuma (BF16) | FP8 (emulado) | INT4 (nativo A40) |
| VRAM | ~17GB | ~38.4GB | **18.65GB** |
| Disco | 17GB | 29GB | **20GB** |
| Velocidade | Rapido | ~14.9 tok/s | **~14.9 tok/s** |
| Capacidade | Basica | Alta | **Alta** |

## 3. Arquivos Modificados

| Arquivo | Alteracao |
|---|---|
| `.env` | `VLLM_MODEL` -> `/workspace/models/Qwen3.5-27B-AWQ` |
| `src/config.py` | Defaults atualizados para novo modelo |
| `src/extraction/vlm_client.py` | Default e docstring atualizados |
| `/workspace/models/Qwen3.5-27B-AWQ/chat_template.jinja` | Thinking desabilitado por padrao |
| `/workspace/start_all.sh` | Modelo, GPU util (0.85), `--enforce-eager` |

## 4. Backup Criado

- `docs/qwen3-vl-8b-backup.md` — Configuracao completa do Qwen3-VL-8B para restauracao se necessario.

## 5. Teste de Ingestao com AWQ

- Ingestao da PORTARIA-938-2021 (4 paginas): **55 dispositivos extraidos**, pipeline completo em ~3 minutos, sem erros.

## 6. Infraestrutura Atual

- **vLLM** (porta 8002): Qwen3.5-27B-AWQ, `--enforce-eager`, `--gpu-memory-utilization 0.85`
- **GPU Server** (porta 8000): FastAPI com BGE-M3 + Reranker, 9 env vars carregadas
- **Cloudflare Tunnel**: `gpu.vectorgov.io` -> porta 8000 (nomeado, permanente)
- **vLLM nightly**: 0.16.0rc2.dev483
