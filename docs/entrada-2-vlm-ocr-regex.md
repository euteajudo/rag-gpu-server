# Entrada 2 — Qwen3-VL OCR + Regex (Leis, Decretos, INs)

Pipeline de ingestão **com GPU** para documentos legislativos cujo texto nativo é deficiente (PDFs escaneados, imagens, texto corrompido). Ativado com `extraction_mode = "vlm"`.

> **Princípio de design:** "A única variável entre Entrada 1 e Entrada 2 é **de onde vem o texto**."
> Entrada 1 usa texto nativo do PyMuPDF; Entrada 2 usa OCR via Qwen3-VL. Todo o pipeline downstream (regex classifier, chunks, embeddings, artifacts) é **idêntico**.

---

## Visão Geral

```
PDF (bytes)
  │
  ├─── PyMuPDFExtractor.extract_pages()     ← imagens das páginas (PNG @ 300 DPI)
  │
  └─── Qwen3-VL OCR (página a página)       ← texto OCR bruto por página
         │
         ▼
       split_ocr_into_blocks()               ← blocos sintéticos com offsets
         │  blocks: List[dict]
         │  canonical_text: str
         ▼
       Idempotency Check                     ← canonical_text == normalize_canonical_text()
         ▼
       RegexClassifier.classify_document()   ← MESMO regex da Entrada 1
         ▼
       Quality Gate OCR                      ← validate_ocr_quality()
         ▼
       _regex_to_processed_chunks()          ← MESMO conversor da Entrada 1
         ▼
       OriginClassifier                      ← MESMO classificador de proveniência
         ▼
       BGE-M3 Embeddings                     ← dense (1024d) + sparse
         ▼
       Artifacts Upload (HTTP POST → VPS)
         ▼
       validate_chunk_invariants()           ← MESMAS invariantes
         ▼
       IngestResponse (chunks + manifest)
```

---

## O Que Difere da Entrada 1

| Aspecto | Entrada 1 (PyMuPDF) | Entrada 2 (VLM OCR) |
|---------|---------------------|----------------------|
| **Fonte do texto** | Texto nativo do PDF | OCR via Qwen3-VL sobre imagens |
| **Construção de blocos** | Blocos nativos do PyMuPDF (com bbox, font, spans) | Blocos sintéticos via `split_ocr_into_blocks()` (sem bbox) |
| **Bounding boxes** | Reais (PDF points, 72 DPI) | Vazias (`bbox: []`) — OCR não retorna geometria por bloco |
| **GPU necessária** | Não | Sim (Qwen3-VL via vLLM na porta 8002) |
| **Quality gate** | Apenas idempotency check | Idempotency check + `validate_ocr_quality()` |
| **Velocidade** | Rápida (~12s para 75 páginas) | Lenta (~120s para 75 páginas, sequencial página a página) |

**Todo o restante é idêntico:** regex classifier, conversão para ProcessedChunk, OriginClassifier, embeddings, artifacts, validação de invariantes.

---

## Fases do Pipeline

### Fase 1 — Extração VLM OCR

A extração OCR envolve dois componentes trabalhando juntos:

#### 1a. PyMuPDFExtractor (imagens)

O mesmo `PyMuPDFExtractor` da Entrada 1 é usado para renderizar as páginas como imagens PNG a 300 DPI. O texto nativo é descartado — apenas as imagens são aproveitadas.

#### 1b. Qwen3-VL OCR (texto)

**Arquivo:** `src/extraction/vlm_client.py`
**Classe:** `VLMClient`
**Método:** `ocr_page(image_base64: str) -> str`

Para cada página:
1. Envia imagem em base64 ao servidor vLLM (Qwen3-VL na porta 8002)
2. Usa prompts especializados para OCR de documentos legais brasileiros
3. Remove bloco `<think>` da resposta (Qwen3 gera raciocínio interno)
4. Retorna texto OCR bruto da página

**Prompts OCR** (`src/extraction/vlm_ocr.py`):

```
System: "Voce e um OCR de alta precisao para documentos legais brasileiros.
         Transcreva EXATAMENTE o texto visivel na imagem, preservando:
         - Quebras de linha entre paragrafos
         - Caracteres especiais (Art., §, º, etc.)
         - Numeracao e pontuacao exatas
         - Acentuacao correta
         NAO adicione comentarios ou formatacao. Retorne APENAS o texto."

User:    "Transcreva todo o texto visivel nesta pagina de legislacao brasileira.
         Mantenha as quebras de linha entre paragrafos."
```

**Orquestração:** `src/extraction/vlm_service.py`
**Método:** `ocr_document(pdf_bytes, document_id, progress_callback) -> Tuple[List[PageData], str]`

Processa páginas sequencialmente (não em paralelo, para estabilidade do vLLM), com retry (até 3 tentativas por página).

#### 1c. split_ocr_into_blocks()

**Arquivo:** `src/extraction/vlm_ocr.py`
**Função:** `split_ocr_into_blocks(ocr_pages: List[Tuple[int, str]]) -> Tuple[List[dict], str, List[Tuple[int, int, int]]]`

Transforma o texto OCR bruto em blocos sintéticos compatíveis com o regex classifier:

1. **Normaliza** cada página: NFC + rstrip por linha (mesma normalização do PyMuPDFExtractor)
2. **Concatena** páginas com separador `\n` (1 newline entre páginas)
3. **Encontra split points** usando regex de início de dispositivo:
   - `Art. \d+` — artigo
   - `§ \d+` — parágrafo
   - `Parágrafo único`
   - `[IVXL]+ [-–—]` — inciso (numeral romano)
   - `[a-z]) ` — alínea
4. **Cria blocos sintéticos** com offsets nativos ao canonical_text

**Output — Bloco sintético:**
```python
{
    "block_index": int,
    "text": str,
    "char_start": int,       # offset no canonical_text
    "char_end": int,          # offset no canonical_text
    "bbox": [],               # vazio — OCR não tem geometria por bloco
    "lines": [],              # vazio — OCR não tem font/span data
    "page_number": int,       # determinado pelos page_boundaries
}
```

**Invariante crítica:**
```python
canonical_text[block["char_start"]:block["char_end"]] == block["text"]
```

### Fase 2 — Idempotency Check

Idêntico à Entrada 1. Verifica que o canonical_text do OCR já está normalizado:
```python
assert canonical_text == normalize_canonical_text(canonical_text)
```
Se divergir, aborta com `RuntimeError`.

### Fase 3 — Classificação Regex

**Idêntico à Entrada 1.** O mesmo `RegexClassifier` processa os blocos sintéticos exatamente como processaria blocos nativos do PyMuPDF.

A classificação funciona porque o OCR preserva os marcadores textuais ("Art.", "§", "I —", "a)") que o regex reconhece.

### Fase 4 — Quality Gate OCR

**Arquivo:** `src/extraction/vlm_ocr.py`
**Função:** `validate_ocr_quality(devices, canonical_text, total_pages, document_id) -> List[str]`

Verificações específicas da Entrada 2 (não existem na Entrada 1):

| Verificação | Critério | Consequência |
|-------------|----------|-------------|
| Artigos encontrados | `len(devices) > 0` | Warning se zero |
| Caracteres por página | `len(canonical_text) / total_pages > threshold` | Warning se baixo |
| Dispositivos por página | `len(devices) / total_pages > threshold` | Warning se baixo |

As warnings são registradas em `result.quality_issues` mas **não abortam** o pipeline. São sinais para revisão humana.

### Fases 5-9 — Pipeline Convergente

A partir da classificação regex, **todo o pipeline é idêntico à Entrada 1**:

5. **Conversão para ProcessedChunk** (`_regex_to_processed_chunks`) — mesmos node_ids `"leis:..."`, mesma hierarquia
6. **OriginClassifier** — mesma detecção de material externo
7. **Embeddings BGE-M3** — mesmos vetores dense + sparse
8. **Artifacts Upload** — mesmos artefatos via HTTP POST
9. **Validação de Invariantes** — mesmas regras, mesmo `ContractViolationError`

### Inspeção

O snapshot de inspeção registrado no Redis diferencia a entrada pela campo `extraction_source`:
- Entrada 1: `extraction_source = "pymupdf_native"`
- Entrada 2: `extraction_source = "vlm_ocr"`

---

## Quando Usar a Entrada 2

| Cenário | Recomendação |
|---------|-------------|
| PDF com texto nativo legível | Entrada 1 (mais rápida, mais precisa) |
| PDF escaneado (imagem only) | **Entrada 2** |
| PDF com texto corrompido/ilegível pelo PyMuPDF | **Entrada 2** |
| PDF com layout complexo (colunas, tabelas) | Testar ambas, comparar resultados |
| Debug/validação cruzada | Rodar ambas e comparar dispositivos extraídos |

---

## Endpoint HTTP

```
POST /ingest
Content-Type: multipart/form-data

Campos:
  file: PDF (binary)
  document_id: "LEI-14133-2021"
  tipo_documento: "LEI"
  numero: "14133"
  ano: 2021
  extraction_mode: "vlm"            # ← ativa Entrada 2
  skip_embeddings: false
```

A resposta tem exatamente a mesma estrutura da Entrada 1, com a diferença na fase registrada:
```json
{
  "phases": [
    {"name": "vlm_ocr_extraction", "method": "vlm_ocr+regex", ...},
    {"name": "embedding", ...},
    {"name": "artifacts_upload", ...}
  ]
}
```

---

## Infraestrutura Necessária

| Componente | Endereço | Papel |
|------------|----------|-------|
| vLLM (Qwen3-VL) | `http://localhost:8002/v1` | Servidor de inferência para OCR |
| GPU | NVIDIA (RunPod) | Execução do modelo VLM |

**Configuração** (`src/config.py`):
```python
vllm_base_url = "http://localhost:8002/v1"
vllm_model = "Qwen/Qwen3-VL-8B-Instruct"
vlm_page_dpi = 300
vlm_max_retries = 3
```

---

## Arquivos-Chave

| Arquivo | Papel |
|---------|-------|
| `src/ingestion/pipeline.py` | Orquestração (`_phase_vlm_extraction`) |
| `src/extraction/vlm_client.py` | Cliente HTTP para Qwen3-VL (método `ocr_page`) |
| `src/extraction/vlm_service.py` | Orquestração de OCR por documento (`ocr_document`) |
| `src/extraction/vlm_ocr.py` | Prompts OCR, `split_ocr_into_blocks()`, `validate_ocr_quality()` |
| `src/extraction/vlm_models.py` | Modelos: `BlockData`, `PageData` |
| `src/extraction/pymupdf_extractor.py` | Renderização de imagens (reusado da Entrada 1) |
| `src/extraction/regex_classifier.py` | Classificação regex (compartilhado com Entrada 1) |
