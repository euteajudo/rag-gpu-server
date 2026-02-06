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

---

## STATUS: Fase 2 COMPLETA + Fase 3 (Integração Pipeline) COMPLETA

### Fase 2 — Backend de Extração VLM (COMPLETA)

Todos os 5 arquivos criados e integrados:
- `src/extraction/vlm_models.py` — PageData, DeviceExtraction, PageExtraction, DocumentExtraction
- `src/extraction/vlm_prompts.py` — SYSTEM_PROMPT + PAGE_PROMPT_TEMPLATE
- `src/extraction/pymupdf_extractor.py` — PyMuPDFExtractor (fitz)
- `src/extraction/vlm_client.py` — VLMClient (httpx async, multimodal)
- `src/extraction/vlm_service.py` — VLMExtractionService (orquestrador)

Arquivos modificados: `config.py`, `pipeline.py`, `extraction/__init__.py`, `main.py`

### Fase 3 — Integração no Pipeline de Ingestão (COMPLETA)

**Problema resolvido**: O path VLM no `pipeline.py` tinha 3 bloqueadores críticos:
1. Offsets sentinela (`canonical_start=-1, canonical_end=-1`) — violava Invariante 7
2. Sem validação `validate_chunk_invariants()` para chunks VLM
3. Sem upload de artefatos (`canonical.md` + `offsets.json`) para a VPS

**Arquivo modificado**: `src/ingestion/pipeline.py` (único)

#### 3.1 Novo método: `_resolve_vlm_offsets()` (~165 linhas)

Computa offsets reais para cada chunk VLM dentro do `canonical_text` PyMuPDF.

**Estratégia em 2 fases:**

- **Fase A — Artigos**: Para cada chunk `device_type == "article"`:
  1. `canonical_text.find(art_text.strip())` — match exato
  2. Fallback: regex do identificador (`Art.\s*N°...`) e estende até próximo artigo ou fim

- **Fase B — Filhos** (paragraph, inciso, alinea): Para cada chunk filho:
  1. Determina `parent_span_id` do `parent_node_id` (`"leis:DOC#SPAN"` → `SPAN`)
  2. Busca range do pai nos offsets já resolvidos (ou do artigo ancestral)
  3. `resolve_child_offsets()` com texto exato dentro do range do pai
  4. Fallback 1: texto com whitespace normalizado
  5. Fallback 2: regex do identificador dentro do range do pai
  6. Fallback 3: `canonical_text.find()` global (se pai não resolvido)

**Reutiliza**: `resolve_child_offsets()` e `OffsetResolutionError` de `src/chunking/canonical_offsets.py`

#### 3.2 Modificação: `_vlm_to_processed_chunks()`

- Comentário atualizado: "sentinela inicial, resolvido por `_resolve_vlm_offsets`"
- Chamada adicionada ao final: `self._resolve_vlm_offsets(chunks, canonical_text, hash, doc_id)`
- Os valores `-1` iniciais são sobrescritos pelo método de resolução

#### 3.3 Novo método: `_phase_vlm_artifacts_upload()` (~100 linhas)

Upload de artefatos para a VPS, análogo a `_phase_artifacts_upload()` do pipeline legado.

**O que envia via `ArtifactsUploader.upload()`:**
- `pdf_content`: bytes do PDF original
- `canonical_md`: `extraction.canonical_text` normalizado
- `offsets_json`: mapa de offsets construído a partir dos chunks resolvidos
- `metadata`: `ArtifactMetadata` com `pipeline_version="1.0.0-vlm"`

**Comportamento em falha**: warning + continua (mesmo padrão do legado, nunca aborta)

**Reutiliza**: `ArtifactsUploader`, `ArtifactMetadata`, `prepare_offsets_map`, `compute_sha256`

#### 3.4 Modificação: `_phase_vlm_extraction()`

Adicionadas 2 etapas entre embeddings e `COMPLETED`:

1. **Artifacts upload** — `self._phase_vlm_artifacts_upload(...)`
2. **Validação de contrato** — `validate_chunk_invariants(chunks, document_id)`

**Progresso atualizado:**

| Fase | Antes | Depois |
|------|-------|--------|
| vlm_extraction | 0.10 → 0.80 | 0.10 → 0.80 (sem mudança) |
| vlm_materialization | 0.85 → 0.90 | 0.80 → 0.88 |
| embedding | 0.90 → 0.95 | 0.88 → 0.94 |
| artifacts_upload | — | 0.94 → 0.97 |
| validation + completed | 0.95 → 1.0 | 0.97 → 1.0 |

#### 3.5 Limpeza menor

- Removido dead code (ternário `if False else`) no fallback de identificador

### Fluxo Atualizado do Pipeline VLM

```
PDF bytes
  │
  ▼
PyMuPDF: extrai páginas (imagens + texto)
  │
  ▼
Qwen3-VL: extrai dispositivos de cada página
  │
  ▼
_vlm_to_processed_chunks(): cria ProcessedChunks (offsets sentinela -1/-1)
  │
  ▼
_resolve_vlm_offsets(): computa offsets reais via find() + resolve_child_offsets()
  │
  ▼
Embeddings (dense + sparse via BGE-M3)
  │
  ▼
_phase_vlm_artifacts_upload(): envia PDF + canonical.md + offsets.json para VPS
  │
  ▼
validate_chunk_invariants(): verifica contrato (Invariantes 1-7)
  │
  ▼
COMPLETED → retorna ProcessedChunks para VPS
```

### Verificação Pendente

1. **Teste com documento real** (requer vLLM + Qwen3-VL rodando):
   ```bash
   export USE_VLM_PIPELINE=true
   python -c "
   from src.ingestion.pipeline import IngestionPipeline, IngestRequest
   pipeline = IngestionPipeline()
   with open('tests/fixtures/sample.pdf', 'rb') as f:
       pdf = f.read()
   req = IngestRequest(document_id='test-vlm', tipo_documento='LEI', numero='1', ano=2024)
   result = pipeline.process(pdf, req)
   print(f'Status: {result.status}')
   for c in result.chunks[:3]:
       print(f'  {c.span_id}: start={c.canonical_start} end={c.canonical_end}')
   "
   ```

2. **Validação**: Pipeline deve passar `validate_chunk_invariants()` sem `ContractViolationError`

3. **Pipeline legado**: `USE_VLM_PIPELINE=false` (default) continua sem regressão

4. **Offsets válidos**: Todos os evidence chunks com `canonical_start >= 0`, `canonical_end > start`, `canonical_hash != ""`

---

## Fase 4 — Coordenadas e Offsets Determinísticos (PENDENTE)

Dois problemas estruturais identificados no código das Fases 2-3 que precisam ser corrigidos
antes de ir para produção.

### Problema 1: Coordenadas — bbox image-space chega ao Milvus como se fosse PDF-space

**Estado atual (errado):**

```
VLM retorna bbox normalizada [0-1] (image space, relativa à imagem 300 DPI)
     │
     ▼
DeviceExtraction.bbox = [x0, y0, x1, y1]   ← normalizado 0-1
     │
     ▼
ProcessedChunk.bbox = [x0, y0, x1, y1]     ← COPIADO SEM CONVERSÃO
     │
     ▼
Milvus campo "bbox" (docstring diz "Bounding box no PDF")  ← MENTIRA
     │
     ▼
Frontend/Evidence Drawer tenta usar como PDF-space → highlight errado
```

O `ProcessedChunk.bbox` documenta "Bounding box no PDF" mas recebe coordenadas
normalizadas 0-1 do image space. O frontend precisa de PDF points (72 DPI) para
highlight. As dimensões da página (`PageData.width`/`height`) existem mas nunca são
usadas na conversão.

**Estado alvo:**

Armazenar coordenadas em AMBOS os espaços, com conversão explícita:

- `bbox_pdf`: coordenadas em PDF points (72 DPI) — para highlight no frontend
- `bbox_img`: coordenadas normalizadas 0-1 — para debug/reprodutibilidade
- `img_width`, `img_height`: dimensões do pixmap em pixels — para reproduzir conversão

#### 4.1 Nova função: `image_bbox_to_pdf_bbox()`

**Arquivo**: `src/extraction/coord_utils.py` (novo)

```python
def image_bbox_to_pdf_bbox(
    bbox_norm: list[float],      # [x0, y0, x1, y1] normalizado 0-1
    page_width_pts: float,       # largura da página em pontos PDF
    page_height_pts: float,      # altura da página em pontos PDF
) -> list[float]:
    """
    Converte bbox normalizada (image space 0-1) para PDF space (pontos 72 DPI).

    A imagem renderizada pelo PyMuPDF é uma escala linear da página PDF:
      pixel_x = pdf_x * (dpi / 72)
    Logo a bbox normalizada (pixel / img_dim) converte para:
      pdf_x = norm_x * page_width_pts

    Returns:
        [x0, y0, x1, y1] em pontos PDF (72 DPI)
    """
    x0, y0, x1, y1 = bbox_norm
    return [
        x0 * page_width_pts,
        y0 * page_height_pts,
        x1 * page_width_pts,
        y1 * page_height_pts,
    ]
```

A conversão é linear porque `get_pixmap(matrix=Matrix(zoom, zoom))` escala
uniformemente. O VLM retorna bbox normalizada 0-1, então:
`pdf_coord = norm_coord × page_dim_pts`.

**Validação**: Função `validate_bbox_pdf()` no mesmo arquivo:
- `x0 < x1`, `y0 < y1` (não degenerado)
- `0 <= x0` e `x1 <= page_width_pts` (dentro da página)
- Área mínima > 0

#### 4.2 Mudanças em modelos

**`src/extraction/vlm_models.py` — `DeviceExtraction`:**
```python
class DeviceExtraction(BaseModel):
    # ... campos existentes ...
    bbox: list[float] = Field(default_factory=list, description="[x0, y0, x1, y1] normalizado 0-1 (image space)")
    # ↑ apenas renomear a description para deixar claro que é image space
```

**`src/extraction/vlm_models.py` — `PageData`:**
```python
@dataclass
class PageData:
    # ... campos existentes ...
    img_width: int = 0    # largura do pixmap em pixels (= width * dpi/72)
    img_height: int = 0   # altura do pixmap em pixels (= height * dpi/72)
```
Preenchido no `pymupdf_extractor.py` a partir de `pixmap.width`/`pixmap.height`.

**`src/ingestion/models.py` — `ProcessedChunk`:**
```python
class ProcessedChunk(BaseModel):
    # Campo existente — renomear para bbox_pdf, ou manter bbox e garantir que é PDF space
    bbox: list[float] = Field(default_factory=list, description="Bounding box em PDF points [x0, y0, x1, y1]")
    bbox_img: list[float] = Field(default_factory=list, description="Bounding box normalizada 0-1 (image space, debug)")
    img_width: int = Field(0, description="Largura do pixmap em pixels")
    img_height: int = Field(0, description="Altura do pixmap em pixels")
```

#### 4.3 Onde chamar a conversão

Em `pipeline.py` → `_vlm_to_processed_chunks()`, na construção de cada `ProcessedChunk`:

```python
from ..extraction.coord_utils import image_bbox_to_pdf_bbox

# Conversão: image space → PDF space
bbox_img = device.bbox  # normalizado 0-1, como o VLM retornou
bbox_pdf = image_bbox_to_pdf_bbox(bbox_img, page_data.width, page_data.height) if bbox_img else []

pc = ProcessedChunk(
    bbox=bbox_pdf,          # PDF points — pronto para highlight
    bbox_img=bbox_img,      # normalizado 0-1 — debug/reprodutibilidade
    img_width=page_data.img_width,
    img_height=page_data.img_height,
    ...
)
```

**Requisito**: `_vlm_to_processed_chunks()` precisa ter acesso às `PageData` originais
(atualmente só recebe `DocumentExtraction` que tem `PageExtraction` sem dimensões).
Solução: passar `pages_data: list[PageData]` como argumento adicional, ou enriquecer
`PageExtraction` com `width`/`height`.

---

### Problema 2: Offsets — `find()` posterior é frágil; canonical_text deve nascer dos blocos

**Estado atual (frágil):**

```
PyMuPDF extractor: page.get_text("text") → texto de cada página
     │
     ▼
vlm_service.py: "\n".join(page.text for page in pages) → canonical_text
     │
     ▼
VLM: retorna device.text (OCR do modelo, pode ter diferenças sutis)
     │
     ▼
pipeline.py _resolve_vlm_offsets(): canonical_text.find(device.text) → offsets
```

Problemas:
1. `get_text("text")` pode ter reading order diferente de `get_text("dict")` em PDFs
   com múltiplas colunas, tabelas, headers/footers
2. `find()` falha se texto duplicado (ambiguidade) ou whitespace diferente
3. Offsets são "mapeamento posterior" — frágeis por natureza

**Estado alvo:**

```
PyMuPDF extractor: page.get_text("dict") → blocos ordenados com bbox
     │
     ├── concatena blocos em reading order → canonical_text
     ├── calcula char_start/char_end DURANTE concatenação
     └── armazena mapa block_index → (char_start, char_end, bbox_pdf)
           │
           ▼
     Offsets são consequência natural da concatenação,
     não mapeamento posterior via find()
```

#### 4.4 Reescrever `PyMuPDFExtractor.extract_pages()`

**Arquivo**: `src/extraction/pymupdf_extractor.py`

Em vez de `page.get_text("text")`, usar `page.get_text("dict")` que retorna blocos
com bboxes. Construir `canonical_text` e offsets incrementalmente:

```python
def extract_pages(self, pdf_bytes: bytes) -> list[PageData]:
    # ...
    for page_idx in range(total_pages):
        page = doc[page_idx]

        # Extrai blocos com bbox (reading order)
        page_dict = page.get_text("dict", sort=True)  # sort=True → reading order
        blocks = page_dict["blocks"]

        # Filtra blocos de texto (type=0), ignora imagens (type=1)
        text_blocks = [b for b in blocks if b["type"] == 0]

        # Concatena texto dos blocos e calcula offsets
        page_text_parts = []
        page_block_map = []  # (char_start, char_end, bbox_pdf)

        for block in text_blocks:
            # Extrai texto de todas as linhas/spans do bloco
            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"]
                block_text += "\n"

            if not block_text.strip():
                continue

            char_start = current_offset  # offset global no canonical_text
            page_text_parts.append(block_text)
            current_offset += len(block_text)
            char_end = current_offset

            # bbox do bloco já está em PDF points (72 DPI)
            bbox_pdf = block["bbox"]  # (x0, y0, x1, y1)

            page_block_map.append((char_start, char_end, bbox_pdf))

        page_text = "".join(page_text_parts)
        # ...
```

#### 4.5 Novo modelo `BlockData` e enriquecer `PageData`

**Arquivo**: `src/extraction/vlm_models.py`

```python
@dataclass
class BlockData:
    """Um bloco de texto extraído pelo PyMuPDF com offset no canonical_text."""
    block_index: int
    char_start: int       # offset início no canonical_text
    char_end: int         # offset fim no canonical_text
    bbox_pdf: list[float] # [x0, y0, x1, y1] em pontos PDF (72 DPI)
    text: str             # texto do bloco
    page_number: int      # página de origem (1-indexed)

@dataclass
class PageData:
    page_number: int
    image_png: bytes
    image_base64: str
    text: str                       # texto concatenado desta página
    width: float                    # largura em pontos PDF
    height: float                   # altura em pontos PDF
    img_width: int = 0              # largura do pixmap em pixels (NOVO)
    img_height: int = 0             # altura do pixmap em pixels (NOVO)
    blocks: list[BlockData] = field(default_factory=list)  # blocos com offsets (NOVO)
    char_start: int = 0             # offset do início desta página no canonical_text (NOVO)
    char_end: int = 0               # offset do fim desta página no canonical_text (NOVO)
```

#### 4.6 Impacto em `_resolve_vlm_offsets()`

Com os blocos tendo offsets nativos, a resolução de offsets pode usar bbox matching
em vez de `find()`:

1. Para cada `DeviceExtraction` do VLM, converter bbox de image-space para PDF-space
2. Encontrar blocos PyMuPDF cuja `bbox_pdf` tem IoU significativo com a bbox do device
3. Os blocos matched já têm `char_start`/`char_end` — compor o range do device
4. Fallback: `find()` dentro do range da página (não global)

Isso elimina a fragilidade do `find()` global e torna os offsets uma consequência
natural da geometria, não de string matching.

#### 4.7 Impacto no `vlm_service.py`

O `canonical_text` passa a ser construído no `pymupdf_extractor.py` (durante a
concatenação dos blocos), não no `vlm_service.py`. O service recebe `pages_data`
que já contém o texto com offsets.

```python
# ANTES (vlm_service.py:162-163):
raw_canonical = "\n".join(page_data.text for page_data in pages_data)

# DEPOIS:
# canonical_text já vem construído pelo extractor
raw_canonical = canonical_text_from_blocks  # concatenação dos blocos de todas as páginas
```

O `PyMuPDFExtractor` retorna tanto `list[PageData]` quanto o `canonical_text` global
(ou `DocumentPages` wrapper com ambos).

#### 4.8 Resumo das mudanças — Fase 4

| Arquivo | Mudança |
|---|---|
| `src/extraction/coord_utils.py` | NOVO — `image_bbox_to_pdf_bbox()`, `validate_bbox_pdf()` |
| `src/extraction/vlm_models.py` | `BlockData` novo, `PageData` +4 campos |
| `src/extraction/pymupdf_extractor.py` | Reescrever para `get_text("dict")` + offsets incrementais |
| `src/extraction/vlm_service.py` | canonical_text vem do extractor, não do service |
| `src/ingestion/models.py` | `ProcessedChunk` +3 campos: `bbox_img`, `img_width`, `img_height` |
| `src/ingestion/pipeline.py` | `_vlm_to_processed_chunks()`: conversão bbox + passar PageData |
| `src/ingestion/pipeline.py` | `_resolve_vlm_offsets()`: refatorar para usar bbox matching em vez de `find()` global |

#### 4.9 Ordem de implementação

1. `coord_utils.py` — funções puras, sem dependências
2. `vlm_models.py` — `BlockData`, campos novos em `PageData`
3. `pymupdf_extractor.py` — reescrever `extract_pages()` com `get_text("dict")`
4. `vlm_service.py` — adaptar construção do canonical_text
5. `models.py` — campos novos em `ProcessedChunk`
6. `pipeline.py` — conversão bbox + refatorar `_resolve_vlm_offsets()`

#### 4.10 Verificação

1. **Idempotência**: Mesmo PDF → mesmo `canonical_text` → mesmos offsets → mesmo `canonical_hash`
2. **Offsets nativos**: `block.char_start`/`char_end` devem corresponder a `canonical_text[start:end]`
3. **Bbox round-trip**: `image_bbox_to_pdf_bbox(norm_bbox, w, h)` deve produzir coordenadas
   que quando usadas pelo frontend geram highlight na posição correta
4. **Consistência**: `canonical_text` de `get_text("dict")` deve ter o mesmo conteúdo
   (possível whitespace diferente) que `get_text("text")` — validar com 10 PDFs de amostra
5. **Regressão**: `_resolve_vlm_offsets()` refatorado deve resolver >= tantos chunks quanto a versão `find()`
