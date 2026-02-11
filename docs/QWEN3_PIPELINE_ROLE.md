# Documentação do Papel do Qwen3-VL no Pipeline

**Data:** 2026-02-11
**Propósito:** Documentar o papel do modelo Qwen3-VL no pipeline dual-entry

---

## 1. Visão Geral

O modelo **Qwen/Qwen3-VL-8B-Instruct** é o VLM (Vision-Language Model) usado na **Entrada 2** do pipeline de ingestão. Ele faz **OCR puro** de páginas de PDF — transcreve o texto visível na imagem e retorna texto bruto. O texto OCR entra no **mesmo Regex Classifier** da Entrada 1.

### Características do Modelo

| Propriedade | Valor |
|-------------|-------|
| Modelo | `Qwen/Qwen3-VL-8B-Instruct` |
| Tipo | Vision-Language Model (multimodal) |
| Contexto | 8192 tokens |
| Servido por | vLLM (container separado) |
| Porta | `8002` (vLLM) |
| GPU | ~12GB VRAM |
| Função | OCR de páginas de PDF (Entrada 2) |

---

## 2. Onde o Qwen3-VL é Usado

### 2.1 OCR de Páginas (`src/extraction/vlm_client.py`)

O método `ocr_page()` envia uma imagem PNG de página e recebe texto bruto:

```
Página PNG (base64)
    │
    ▼
[VLMClient.ocr_page()]
    │
    ├── Envia imagem + OCR prompt ao vLLM
    ├── Recebe texto bruto (não JSON)
    ├── Remove blocos <think> (Qwen3 thinking mode)
    └── Retorna str com texto transcrito
```

**Diferença vs `extract_page()` (legado):**

| Aspecto | `ocr_page()` (atual) | `extract_page()` (legado) |
|---------|----------------------|--------------------------|
| Retorno | `str` (texto bruto) | `dict` (JSON estruturado) |
| Prompt | OCR: transcreva o texto | Classifique os dispositivos |
| Pós-processamento | Nenhum | `_extract_json()` |
| Temperatura | 0.0 (fixo) | 0.0 |

### 2.2 OCR de Documento (`src/extraction/vlm_service.py`)

O método `ocr_document()` orquestra o OCR de todas as páginas:

```
PDF
 │
 ▼
PyMuPDF extract_pages() ──► imagens PNG + dimensões
 │                            (texto nativo descartado)
 │
 ▼
Para cada página (sequencial):
    VLMClient.ocr_page(image_base64) ──► texto OCR
 │
 ▼
split_ocr_into_blocks(ocr_pages) ──► blocos + canonical_text
 │
 ▼
ocr_to_pages_data(pymupdf_pages, blocks, ...) ──► List[PageData]
 │
 ▼
Retorna (pages_data, canonical_text)
    ↑ mesmo formato que PyMuPDFExtractor.extract_pages()
```

O processamento é **sequencial** (uma página por vez) porque:
- `--max-model-len 8192` do vLLM limita o contexto
- Cada página é processada independentemente
- Evita sobrecarga de VRAM com múltiplas imagens simultâneas

### 2.3 Prompts OCR (`src/extraction/vlm_ocr.py`)

```python
OCR_SYSTEM_PROMPT = (
    "Voce e um OCR de alta precisao para documentos legais brasileiros.\n"
    "Transcreva EXATAMENTE o texto visivel na imagem, preservando:\n"
    "- Quebras de linha entre paragrafos\n"
    "- Caracteres especiais (Art., §, º, etc.)\n"
    "- Numeracao e pontuacao exatas\n"
    "- Acentuacao correta\n"
    "NAO adicione comentarios ou formatacao. Retorne APENAS o texto."
)
```

Decisões de design:
- **Sem JSON** — retorno é texto puro (mais confiável para VLMs)
- **Sem classificação** — o Regex Classifier faz isso
- **Temperature = 0.0** — determinismo máximo
- **Sem acentos no prompt** — evita problemas de encoding no system prompt

---

## 3. Arquitetura Dual-Entry

```
  ENTRADA 1 (PyMuPDF nativo)         ENTRADA 2 (VLM OCR)
  ─────────────────────────          ──────────────────────
  PDF                                PDF
   │                                  │
   ▼                                  ▼
  PyMuPDF                            PyMuPDF (imagens)
  extract_pages()                         │
   │                                      ▼
   │                                 Qwen3-VL OCR
   │                                 (página por página)
   │                                      │
   │                                 split_ocr_into_blocks()
   │                                 ocr_to_pages_data()
   │                                      │
   ├── pages_data                    ├── pages_data (sintéticos)
   └── canonical_text                └── canonical_text (OCR)
              │                                │
              └──────────┬─────────────────────┘
                         │
                         ▼
                  MESMO Regex Classifier
                  MESMA _regex_to_processed_chunks()
                  MESMO pipeline downstream
```

> **"A única variável é DE ONDE vem o texto."** — Design doc v3

---

## 4. Quality Gate (`validate_ocr_quality`)

Após o Regex Classifier processar o texto OCR, 3 verificações são executadas:

| Check | Condição de Warning | Significado |
|-------|---------------------|-------------|
| QG1 | 0 artigos encontrados | Texto pode estar corrompido |
| QG2 | < 100 chars/página | OCR pode ter falhado |
| QG3 | < 1 dispositivo/página (>2 páginas) | Classificação incompleta |

Warnings são logados e registrados em `result.quality_issues`, mas **não abortam** o pipeline. O operador decide via inspeção manual.

---

## 5. Código Específico para Qwen3

### 5.1 Thinking Mode

Qwen3 gera blocos `<think>...</think>` antes da resposta. O código os remove:

```python
# Em vlm_client.py
def _strip_thinking_block(text: str) -> str:
    """Remove bloco <think>...</think> da resposta do Qwen 3."""
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.DOTALL)
    return text.strip()
```

### 5.2 Multimodal Input

O VLM recebe imagens via formato OpenAI Vision:

```python
messages = [
    {"role": "system", "content": OCR_SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            {"type": "text", "text": OCR_PAGE_PROMPT},
        ],
    },
]
```

---

## 6. Configuração

### 6.1 Variáveis de Ambiente

```bash
VLLM_BASE_URL=http://localhost:8002/v1
VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

### 6.2 Config (`src/config.py`)

```python
@dataclass
class Config:
    vllm_base_url: str = "http://localhost:8002/v1"
    vllm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
```

---

## 7. Arquivos Relevantes

| Arquivo | Função |
|---------|--------|
| `src/extraction/vlm_client.py` | Cliente HTTP para vLLM (ocr_page + extract_page) |
| `src/extraction/vlm_service.py` | Orquestrador (ocr_document + extract_document) |
| `src/extraction/vlm_ocr.py` | Prompts OCR, split_ocr_into_blocks, quality gate |
| `src/extraction/vlm_models.py` | PageData, BlockData, DocumentExtraction |
| `src/extraction/vlm_prompts.py` | Prompts para classificação VLM (legado) |
| `src/extraction/coord_utils.py` | Conversão coordenadas (img 0-1 ↔ PDF pts) |
| `src/config.py` | Configuração central |

---

## 8. Considerações para Substituição do Modelo

### 8.1 O Que o Novo Modelo Precisa Suportar

1. **Input multimodal** — receber imagem PNG + texto
2. **Baixa temperatura** — OCR determinístico (temp=0.0)
3. **Português fluente** — documentos legais brasileiros
4. **Caracteres especiais** — Art., §, º, acentuação

### 8.2 O Que Pode Precisar de Ajuste

1. **Thinking mode** — se o novo modelo não usar `<think>`, ajustar `_strip_thinking_block()`
2. **Formato de imagem** — verificar se suporta `data:image/png;base64,...`
3. **Prompt** — ajustar OCR prompts para o novo modelo
4. **Context limit** — páginas com muito texto podem precisar de modelo com contexto maior

---

## 9. Histórico de Versões

| Data | Versão | Mudança |
|------|--------|---------|
| 2026-02-06 | 1.0 | Documentação inicial (Qwen3-8B-AWQ como classificador) |
| 2026-02-11 | 2.0 | Reescrita completa — Qwen3-VL como OCR puro |

---

**Autor:** Claude Code (RunPod)
