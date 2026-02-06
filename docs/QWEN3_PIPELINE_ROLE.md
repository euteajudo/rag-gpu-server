# Documentação do Papel do Qwen3 no Pipeline

**Data:** 2026-02-06
**Propósito:** Documentar o papel do modelo Qwen3-8B-AWQ antes de refatoração major
**Tag de Referência:** `stable-pre-refactor-v1`

---

## 1. Visão Geral

O modelo **Qwen/Qwen3-8B-AWQ** é o LLM central do pipeline de ingestão de documentos legais. Ele executa tarefas de extração estruturada que requerem compreensão semântica além do que regex pode oferecer.

### Características do Modelo

| Propriedade | Valor |
|-------------|-------|
| Modelo | `Qwen/Qwen3-8B-AWQ` |
| Quantização | AWQ (4-bit) |
| Contexto | 8192 tokens |
| Servido por | vLLM (container separado) |
| Porta padrão | `8080` (vLLM) → `8000` (API) |
| GPU recomendada | 12GB VRAM |

---

## 2. Onde o Qwen3 é Usado

### 2.1 ArticleOrchestrator (Extração de Hierarquia)

**Arquivo:** `src/parsing/article_orchestrator.py`

O Qwen3 extrai a hierarquia completa de cada artigo:

```
Artigo parseado (regex)
    │
    ▼
[ArticleOrchestrator + Qwen3]
    │
    ├── Identifica parágrafos (PAR-xxx-n)
    ├── Identifica incisos (INC-xxx-I, II, III...)
    ├── Identifica alíneas (ALI-xxx-I-a, b, c...)
    └── Valida cobertura (parser vs LLM)
```

**Prompts utilizados:**
- `ARTICLE_SYSTEM_PROMPT` - Instruções gerais com `/no_think`
- `ARTICLE_USER_PROMPT` - Template para extração inicial
- `ARTICLE_RETRY_PROMPT` - Retry focado por tipo (PAR ou INC)

**Exemplo de chamada:**
```python
orchestrator = ArticleOrchestrator(llm_client, config)
result = orchestrator.extract_all_articles(parsed_doc)
```

### 2.2 VLLMClient (Cliente LLM)

**Arquivo:** `src/llm/vllm_client.py`

O cliente implementa:

1. **Thinking Mode Handling**
   - Qwen3 gera blocos `<think>...</think>`
   - `_strip_thinking_block()` remove esses blocos
   - Prompt `/no_think` desabilita o thinking mode

2. **Presets de Configuração**
   ```python
   LLMConfig.for_extraction()   # max_tokens=12288, temp=0.0
   LLMConfig.for_enrichment()   # max_tokens=1024, temp=0.0
   LLMConfig.for_generation()   # max_tokens=12288, temp=0.3
   ```

3. **Guided JSON**
   - `chat_with_schema()` usa `response_format: json_schema`
   - vLLM força geração de JSON válido

### 2.3 Enrichment Prompts (DEPRECATED)

**Arquivo:** `src/chunking/enrichment_prompts.py`

> **NOTA:** Este módulo foi descontinuado em favor de `retrieval_text` determinístico.

Originalmente usava Qwen3 para:
- Gerar `context_header` (frase de contexto)
- Gerar `thesis_text` (resumo da tese)
- Classificar `thesis_type` (definição, procedimento, prazo...)
- Gerar `synthetic_questions` (perguntas que o chunk responde)

---

## 3. Arquitetura Anti-Alucinação

O pipeline usa várias camadas de proteção:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CAMADAS DE PROTEÇÃO                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. SCHEMA ENUM DINÂMICO                                            │
│     - LLM só pode retornar IDs que existem no documento             │
│     - Schema é gerado por artigo com IDs permitidos                 │
│                                                                     │
│  2. VALIDAÇÃO DE COBERTURA                                          │
│     - Compara spans do parser vs spans do LLM                       │
│     - Threshold padrão: 80%                                         │
│                                                                     │
│  3. RETRY FOCADO POR JANELA                                         │
│     - Se cobertura PAR < 100%: retry focado em parágrafos           │
│     - Se cobertura INC < 100%: retry focado em incisos              │
│     - Max 2 retries (1 PAR + 1 INC)                                 │
│                                                                     │
│  4. VALIDAÇÃO DE IDs                                                │
│     - ID deve existir no ParsedDocument                             │
│     - Detecta duplicatas                                            │
│     - Valida consistência parent-child                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Configuração

### 4.1 Variáveis de Ambiente

```bash
VLLM_BASE_URL=http://localhost:8080/v1
VLLM_MODEL=Qwen/Qwen3-8B-AWQ
```

### 4.2 Config Padrão

**Arquivo:** `src/config.py`

```python
@dataclass
class Config:
    vllm_base_url: str = "http://localhost:8080/v1"
    vllm_model: str = "Qwen/Qwen3-8B-AWQ"
```

### 4.3 OrchestratorConfig

```python
@dataclass
class OrchestratorConfig:
    temperature: float = 0.0
    max_tokens: int = 512
    model_context_limit: int = 8192  # Limite do Qwen3-8B
    chars_per_token: float = 4.0
    context_safety_margin: int = 256
    output_token_budget: int = 1024
    max_ids_per_list: int = 200  # Paginação
    coverage_threshold: float = 0.8
    enable_retry: bool = True
    max_retries: int = 2
```

---

## 5. Fluxo de Processamento

```
PDF
 │
 ▼
Docling (PDF → Markdown)
 │
 ▼
SpanParser (Regex - Determinístico)
 │  ├── CAPÍTULO I, II, III...
 │  ├── Art. 1º, Art. 2º...
 │  ├── § 1º, § 2º, § único
 │  ├── I -, II -, III -
 │  └── a), b), c)
 │
 ▼
ArticleOrchestrator (LLM - Qwen3)
 │  ├── Para cada artigo:
 │  │   ├── Gera documento anotado: [SPAN_ID] texto
 │  │   ├── Qwen3 extrai hierarquia (PAR, INC, ALI)
 │  │   ├── Valida cobertura (parser vs LLM)
 │  │   └── Retry focado se cobertura < 100%
 │  │
 │  └── Curto-circuito: artigos sem filhos não chamam LLM
 │
 ▼
ChunkMaterializer
 │  └── Converte ArticleChunks em chunks indexáveis
 │
 ▼
Milvus + Neo4j
```

---

## 6. Arquivos Críticos

| Arquivo | Função |
|---------|--------|
| `src/llm/vllm_client.py` | Cliente HTTP para vLLM |
| `src/llm/__init__.py` | Exports do módulo LLM |
| `src/config.py` | Configuração central |
| `src/parsing/article_orchestrator.py` | Orquestrador LLM |
| `src/parsing/span_extraction_models.py` | Schemas Pydantic |
| `src/chunking/enrichment_prompts.py` | Prompts (deprecated) |

---

## 7. Dependências do Qwen3

### 7.1 Funcionalidades Específicas

1. **Thinking Mode (`<think>` blocks)**
   - Qwen3 gera blocos de pensamento
   - Código remove com `_strip_thinking_block()`
   - Prompt `/no_think` desabilita (mas nem sempre funciona 100%)

2. **Context Window**
   - 8192 tokens de contexto
   - `_calculate_max_input_chars()` considera este limite
   - Artigos grandes são truncados ou divididos

3. **Guided JSON (vLLM)**
   - `chat_with_schema()` usa `response_format: json_schema`
   - Depende de suporte vLLM, não do modelo

### 7.2 Código Específico para Qwen3

```python
# Em vllm_client.py

def _strip_thinking_block(text: str) -> str:
    """Remove bloco <think>...</think> da resposta do Qwen 3."""
    # Remove blocos completos
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.DOTALL)
    # Remove blocos incompletos
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.DOTALL)
    return text.strip()

# Em article_orchestrator.py (linha 371)
ARTICLE_SYSTEM_PROMPT = """... /no_think
...
"""
```

---

## 8. Procedimento de Rollback

### 8.1 Restaurar Tag

```bash
# Verificar tag existe
git tag -l | grep stable-pre-refactor

# Criar branch a partir da tag
git checkout -b rollback-qwen3 stable-pre-refactor-v1

# OU resetar main para a tag (CUIDADO: perde commits posteriores)
git checkout main
git reset --hard stable-pre-refactor-v1
```

### 8.2 Restaurar Container vLLM

```bash
# Parar container atual
docker stop vllm

# Iniciar com Qwen3
docker run -d --name vllm \
  --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8080:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B-AWQ \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95
```

### 8.3 Verificar Funcionamento

```bash
# Health check vLLM
curl http://localhost:8080/v1/models

# Teste de chat
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [{"role": "user", "content": "O que é ETP?"}],
    "max_tokens": 100
  }'
```

---

## 9. Considerações para Substituição

### 9.1 O Que o Novo Modelo Precisa Suportar

1. **Guided JSON** - vLLM `response_format: json_schema`
2. **Baixa temperatura** - Extração determinística (temp=0.0)
3. **Contexto adequado** - Artigos podem ter 5000+ chars
4. **Português fluente** - Documentos legais brasileiros

### 9.2 O Que Pode Precisar de Ajuste

1. **Thinking mode** - Se o novo modelo não usar `<think>`, remover `_strip_thinking_block()`
2. **Prompt `/no_think`** - Específico do Qwen3, pode não ser necessário
3. **Context limit** - Ajustar `model_context_limit` em `OrchestratorConfig`
4. **Token estimation** - `chars_per_token` pode variar por tokenizer

### 9.3 Testes Necessários Após Substituição

```bash
# Rodar testes do parsing
pytest tests/test_address_mismatch_fix.py -v

# Testar ingestão completa
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"document_id": "LEI-14133-2021-TEST", "path": "/data/lei-14133.pdf"}'

# Verificar cobertura de spans
# (verificar logs para "Extração concluída: X/Y válidos")
```

---

## 10. Histórico de Versões

| Data | Versão | Mudança |
|------|--------|---------|
| 2026-02-06 | 1.0 | Documentação inicial pré-refatoração |

---

**Autor:** Claude Code (RunPod)
**Commit de Referência:** `f8f67a1` (ADDRESS_MISMATCH fix)
**Tag:** `stable-pre-refactor-v1`
