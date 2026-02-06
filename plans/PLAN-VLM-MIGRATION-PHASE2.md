# Plano: Backend de Extração VLM (Fase 2 do PLAN-VLM-MIGRATION)

## Contexto

O pipeline atual de ingestão usa **Docling** (PDF→Markdown) + **SpanParser** (regex) para extrair estrutura hierárquica de documentos legais. Estamos migrando para **PyMuPDF** (texto determinístico + imagens) + **Qwen3-VL** (visão) para melhor precisão na extração de estrutura.

O vLLM já está rodando com `Qwen3-VL-8B-Instruct` na porta 8002 (confirmado funcional). A Fase 0 (campos VLM no schema) está completa. Agora precisamos criar o backend que:
1. Recebe o PDF (mesmo contrato /ingest atual)
2. Extrai páginas com PyMuPDF (imagens + texto determinístico)
3. Envia cada página ao Qwen3-VL para extração de estrutura
4. Retorna os dados no formato existente (ProcessedChunk)

## Contrato VPS (preservado)

**Entrada** (sem mudanças): `POST /ingest` multipart/form-data com PDF + metadados (document_id, tipo_documento, numero, ano, etc.)

**Saída** (sem mudanças): `GET /ingest/result/{task_id}` retorna `IngestResponse` com lista de `ProcessedChunk`. Os campos VLM já existem no ProcessedChunk (page_number, bbox, confidence) — serão preenchidos pelo novo pipeline.

**Feature flag**: `USE_VLM_PIPELINE=true` no env seleciona o novo pipeline. Default `false` (pipeline legado).

## Arquivos a Criar

### 1. `src/extraction/vlm_models.py` — Modelos Pydantic

```python
class PageExtraction(BaseModel):
    """Resultado da extração VLM de uma página."""
    page_number: int
    devices: list[DeviceExtraction]

class DeviceExtraction(BaseModel):
    """Um dispositivo legal extraído pelo VLM."""
    device_type: str           # "artigo", "paragrafo", "inciso", "alinea"
    identifier: str            # "Art. 5º", "§ 1º", "I", "a)"
    text: str                  # Texto completo do dispositivo
    parent_identifier: str     # Identificador do pai (vazio se artigo)
    bbox: list[float]          # [x0, y0, x1, y1] normalizado 0-1
    confidence: float          # 0.0-1.0

class DocumentExtraction(BaseModel):
    """Resultado completo da extração VLM do documento."""
    document_id: str
    pages: list[PageExtraction]
    canonical_text: str        # Texto PyMuPDF concatenado
    canonical_hash: str        # SHA256 do canonical_text normalizado
    total_devices: int
```

### 2. `src/extraction/vlm_prompts.py` — Prompts para Qwen3-VL

Contém o prompt de sistema e o template de prompt por página. O prompt instrui o VLM a:
- Identificar todos os dispositivos legais na imagem (artigos, parágrafos, incisos, alíneas)
- Retornar JSON estruturado com device_type, identifier, text, parent_identifier, bbox
- Usar bboxes normalizadas [0-1] relativas às dimensões da página
- Manter o texto exatamente como aparece (sem corrigir ou reformatar)

```python
SYSTEM_PROMPT = """Você é um extrator de estrutura de documentos legais brasileiros..."""

PAGE_PROMPT_TEMPLATE = """Analise esta página de documento legal e extraia todos os dispositivos..."""
```

### 3. `src/extraction/vlm_client.py` — Cliente Multimodal

Novo cliente separado do `VLLMClient` existente (que é text-only). O `VLLMClient` existente (src/llm/vllm_client.py:1-625) assume que `content` é string e faz `.endswith("/no_think")` — incompatível com o formato multimodal `[{"type": "image_url", ...}, {"type": "text", ...}]`.

```python
class VLMClient:
    """Cliente para Qwen3-VL via vLLM (multimodal)."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self.model = model

    async def extract_page(self, image_base64: str, prompt: str) -> dict:
        """Envia imagem + prompt, retorna JSON extraído."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                {"type": "text", "text": prompt},
            ]}
        ]
        # POST /chat/completions → parse JSON da resposta
```

**Decisão**: Cliente assíncrono com `httpx.AsyncClient` (padrão do projeto). Retry com backoff (3 tentativas). Strip de blocos `<think>` (reutiliza `_strip_thinking_block` de `src/llm/vllm_client.py`).

### 4. `src/extraction/pymupdf_extractor.py` — Extração PyMuPDF

```python
class PyMuPDFExtractor:
    """Extrai páginas do PDF: imagens (para VLM) + texto (para canonical)."""

    def extract_pages(self, pdf_bytes: bytes) -> list[PageData]:
        """Retorna lista de PageData com imagem PNG e texto por página."""
        # fitz.open(stream=pdf_bytes, filetype="pdf")
        # Para cada página:
        #   - Renderiza como PNG (300 DPI para VLM)
        #   - Extrai texto nativo (.get_text("text"))
        #   - Coleta dimensões (width, height)

class PageData:
    page_number: int
    image_png: bytes           # PNG da página
    image_base64: str          # Base64 do PNG
    text: str                  # Texto nativo PyMuPDF
    width: float               # Largura da página
    height: float              # Altura da página
```

### 5. `src/extraction/vlm_service.py` — Orquestrador

```python
class VLMExtractionService:
    """Orquestra extração: PyMuPDF → Qwen3-VL → DocumentExtraction."""

    def __init__(self, vlm_client: VLMClient, pymupdf_extractor: PyMuPDFExtractor): ...

    async def extract_document(
        self, pdf_bytes: bytes, document_id: str, progress_callback=None
    ) -> DocumentExtraction:
        """
        Pipeline completo de extração VLM:
        1. PyMuPDF: extrai páginas (imagens + texto)
        2. Qwen3-VL: extrai estrutura de cada página (sequencial)
        3. Concatena canonical_text de todas as páginas
        4. Computa canonical_hash
        5. Retorna DocumentExtraction
        """
```

**Processamento sequencial** (uma página por vez): o `--max-model-len 8192` limita o contexto. Cada página é processada independentemente.

## Arquivos a Modificar

### 6. `src/extraction/__init__.py`
Exportar as classes públicas: `VLMClient`, `VLMExtractionService`, `PyMuPDFExtractor`, `DocumentExtraction`.

### 7. `src/config.py`
Adicionar campos:
```python
# VLM Pipeline
use_vlm_pipeline: bool = False          # Feature flag
vlm_page_dpi: int = 300                 # DPI para renderização
vlm_max_retries: int = 3                # Retries por página
```
Atualizar defaults existentes:
```python
vllm_base_url: str = "http://localhost:8002/v1"   # era 8080
vllm_model: str = "Qwen/Qwen3-VL-8B-Instruct"    # era Qwen3-8B-AWQ
```

### 8. `src/ingestion/pipeline.py`
No método `process()`, adicionar branch por feature flag:
```python
if config.use_vlm_pipeline:
    # Novo pipeline: PyMuPDF + Qwen3-VL
    extraction_result = await self.vlm_service.extract_document(pdf_bytes, document_id)
    # Converte DocumentExtraction → list[ProcessedChunk]
else:
    # Pipeline legado: Docling + SpanParser
    # ... código existente ...
```

O pipeline.py já tem padrão de lazy-loading (propriedades singleton como `_docling_converter`). Seguir mesmo padrão para `_vlm_service`.

### 9. `src/main.py`
Nenhuma mudança de rotas necessária. O `/ingest` existente já serve. Apenas log no startup indicando qual pipeline está ativo:
```python
logger.info(f"Pipeline: {'VLM (Qwen3-VL + PyMuPDF)' if config.use_vlm_pipeline else 'Legado (Docling + SpanParser)'}")
```

## Dependências

- `PyMuPDF` (fitz): já deve estar instalado ou precisa instalar
- `httpx`: já no projeto (usado pelo VLLMClient existente)
- Não precisa de novas rotas FastAPI

## Ordem de Implementação

1. **vlm_models.py** — Modelos Pydantic (sem dependências)
2. **vlm_prompts.py** — Prompts (sem dependências)
3. **pymupdf_extractor.py** — Extração PyMuPDF (depende de fitz)
4. **vlm_client.py** — Cliente multimodal (depende de vlm_prompts)
5. **vlm_service.py** — Orquestrador (depende de 3 e 4)
6. **config.py** — Adicionar campos VLM
7. **pipeline.py** — Integrar feature flag + vlm_service
8. **extraction/__init__.py** — Exports
9. **main.py** — Log do pipeline ativo

## Verificação

1. **Teste unitário**: Testar `PyMuPDFExtractor` com PDF de amostra
   ```bash
   cd /workspace/rag-gpu-server && python -m pytest tests/ -k "pymupdf" -v
   ```

2. **Teste de integração VLM**: Enviar uma página ao VLMClient e verificar resposta JSON
   ```python
   client = VLMClient(base_url="http://localhost:8002/v1", model="Qwen/Qwen3-VL-8B-Instruct")
   result = await client.extract_page(image_base64, PAGE_PROMPT_TEMPLATE)
   assert "devices" in result
   ```

3. **Teste end-to-end**: Ativar flag e rodar ingestão completa
   ```bash
   export USE_VLM_PIPELINE=true
   curl -X POST http://localhost:8000/ingest/... # PDF de teste
   ```

4. **Testes existentes**: Devem continuar passando (flag default=false)
   ```bash
   cd /workspace/rag-gpu-server && python -m pytest tests/ -x
   ```

## Riscos e Mitigações

- **Risco**: Qwen3-VL pode retornar JSON malformado
  - **Mitigação**: Parse com fallback, retry com temperatura mais baixa

- **Risco**: DPI 300 gera imagens grandes (~3MB/página), possível timeout
  - **Mitigação**: DPI configurável, timeout generoso (120s default)

- **Risco**: PyMuPDF não instalado
  - **Mitigação**: Verificar na inicialização, erro claro se ausente

- **Risco**: Pipeline legado quebra com mudanças no config.py
  - **Mitigação**: Defaults mantêm comportamento anterior, feature flag default=false
