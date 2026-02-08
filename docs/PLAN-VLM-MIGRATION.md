# VectorGov — Plano de Migração: SpanParser → Qwen3-VL + Reconciliator

**Versão:** 2.1 (Fase 0 concluída)
**Data:** 2026-02-06
**Autor:** Abimael / Claude
**Propósito:** Documento de planejamento para implementação no Claude Code (IDE)

---

## 1. Contexto e Motivação

### 1.1 Problema Original

O pipeline atual usa Docling + SpanParser (regex) para extrair estrutura hierárquica de documentos legais brasileiros. Esse pipeline sofre de:

- **Docling gera texto inconsistente**: line breaks arbitrários, espaçamento não-determinístico
- **Regex é frágil**: o SpanParser falha quando citações legais aparecem no texto (ex: "conforme Art. 23, § 2º" dispara falso-positivo de detecção de artigo)
- **ADDRESS_MISMATCH**: chunks gerados com `span_id` incorreto por falha no regex — problema que motivou a criação de toda a camada de segurança PR10/PR12/PR13

### 1.2 Solução: Paradigma Visual

Substituir extração baseada em texto (Docling → regex) por extração visual (Qwen3-VL) + texto determinístico (PyMuPDF):

```
ANTES:
  PDF → Docling (texto sujo) → SpanParser (regex) → canonical_builder → Milvus

DEPOIS:
  PDF → PyMuPDF (texto determinístico) + Qwen3-VL (estrutura visual) → Reconciliator → Milvus
```

O VLM "vê" a página como um humano — identifica visualmente onde começa um artigo, parágrafo, inciso, sem depender de regex sobre texto corrompido.

### 1.3 Infraestrutura

- **RunPod A40 48GB**: Qwen3-VL-8B-Instruct (~17GB FP16) + BGE-M3 (~2GB) = ~19GB, ~29GB livres para batch
- **VPS (vectorgov.io)**: FastAPI, Milvus, Neo4j, PostgreSQL, Redis, MinIO
- **BYOL**: Cliente traz seu próprio LLM para query/response — RunPod dedicado apenas à ingestão

---

## 2. Arquitetura de Segurança Existente (PRs)

### 2.1 Visão Geral

O pipeline atual possui uma cadeia de módulos de segurança numerados (PR10, PR12, PR13) que garantem integridade das evidence links — o diferencial do VectorGov.

**Princípio fundamental**: Nenhum chunk chega no Milvus sem garantia de que o evidence link vai funcionar.

### 2.2 Módulos e Seus Papéis

#### PR10 — `snippet_extractor.py` (Query-time)

- **Função**: Localiza trecho exato no texto canônico e extrai snippet com janela de contexto (±240 chars)
- **Mecanismo**: Se `canonical_hash` confere, usa slicing puro `canonical_text[start:end]`; senão fallback via `find()`
- **Regra**: Best-effort — se não encontrar, retorna `None` sem quebrar a evidência
- **Camada**: Opera em query-time, não em ingestão

#### PR12 — `canonical_builder.py` (Ingestão — Construção)

- **Função**: Constrói texto canônico markdown enquanto rastreia offsets de cada dispositivo legal
- **Mecanismo**: `normalize_canonical_text()` garante determinismo (NFC, LF, trailing whitespace); `compute_canonical_hash()` gera SHA256 anti-mismatch
- **Output**: `CanonicalResult` com `canonical_text`, `offsets` dict, e `canonical_hash`

#### PR13 — `pr13_validator.py` + `canonical_offsets.py` (Ingestão — Gate)

- **Função**: Gate crítico pré-Milvus — valida o "trio Evidence"
- **Regras obrigatórias**:
  1. `canonical_start >= 0`
  2. `canonical_end >= canonical_start`
  3. `canonical_hash != "" e != None`
- **Comportamento**: Qualquer violação é CRITICAL → aborta documento inteiro (zero rows no Milvus) → cria alarme no PostgreSQL
- **Princípio**: "Validate ALL before inserting ANY"

#### `canonical_offsets.py` (Resolução de Filhos)

- **Função**: Resolve offsets de filhos (§, incisos, alíneas) dentro do range do pai
- **Mecanismo**: `resolve_child_offsets()` busca texto deterministicamente — exatamente 1 ocorrência dentro do range do pai
- **Erros**: `OffsetResolutionError` com atributos `document_id`, `span_id`, `device_type`, `reason` (NOT_FOUND, AMBIGUOUS, EMPTY_TEXT)

#### `canonical_validation.py` (Hardening de Insert)

- **Função**: Valida formato do `node_id` antes do insert no Milvus
- **Regras**: Prefixo correto (`leis:` / `acordaos:`), sem sufixo `@P`, `logical_node_id` preenchido
- **Modos**: `validate_and_fix` (auto-corrige) e `validate_only` (rejeita)

#### `alarm_service.py` (Observabilidade)

- **Função**: Persiste alarmes no PostgreSQL com deduplicação
- **Features**: Stats agregadas, filtros, bulk resolve, bloqueio de evidências de documentos comprometidos
- **Método crítico**: `has_critical_alarms_for_document()` — pode bloquear exibição de evidence links

### 2.3 Fluxo Integrado Atual

```
PDF → Docling → SpanParser
                    ↓
            canonical_builder (PR12: offsets na ingestão)
                    ↓
            canonical_offsets (resolve filhos dentro do pai)
                    ↓
            pr13_validator (gate: trio válido ou aborta tudo)
                    ↓
            canonical_validation (hardening node_id/prefix)
                    ↓
                  Milvus ✅
                    ↓
            snippet_extractor (PR10: slicing puro com hash anti-mismatch)
                    ↓
              Evidence Link ao PDF no MinIO 🎯
```

---

## 3. Impacto da Migração nos Módulos de Segurança

### 3.1 Matriz de Impacto

| Módulo | Status | Justificativa |
|--------|--------|---------------|
| `pr13_validator.py` | ✅ INTACTO | Gate agnóstico — valida trio, não sabe quem produziu os dados |
| `alarm_service.py` | ✅ INTACTO | Pura observabilidade — recebe severity/type/document_id, persiste |
| `canonical_validation.py` | ✅ INTACTO | Validação de formato node_id — agnóstico ao parser |
| `snippet_extractor.py` | ✅ INTACTO | Opera em query-time — não sabe como offsets foram gerados |
| `canonical_builder.py` | 🔄 REDUZ A UTILS | Funções de normalização/hash sobrevivem, construção do canonical morre |
| `canonical_offsets.py` | 🔄 TRANSFORMA | `resolve_child_offsets()` vira validação/fallback do Reconciliator |
| SpanParser | ❌ MORRE | Substituído pelo Qwen3-VL + Reconciliator |
| Docling | ❌ MORRE | Substituído pelo PyMuPDF |

### 3.2 Detalhamento das Transformações

#### `canonical_offsets.py` → `canonical_utils.py` (✅ FEITO na Fase 0)

**Extraído para `src/utils/canonical_utils.py`** (Fase 0 concluída):
```python
normalize_canonical_text()   # NFC + LF + trailing whitespace + final \n
compute_canonical_hash()     # SHA256 determinístico
validate_offsets_hash()      # Anti-mismatch check
```

**NOTA**: O plano original mencionava `canonical_builder.py` mas esse arquivo não existe no codebase.
As funções estavam em `canonical_offsets.py`, que agora re-exporta de `canonical_utils.py` para backward compatibility.

**Morre** (substituído pelo Reconciliator nas fases posteriores):
```python
extract_offsets_from_parsed_doc()  # Não haverá mais ParsedDocument
build_canonical_with_offsets()     # Reconciliator gera offsets via bbox
_format_node_id()                  # Reconciliator constrói node_id do JSON do VLM
```

#### `canonical_offsets.py` — Transformação

**Morre**:
```python
extract_offsets_from_parsed_doc()  # Não haverá mais ParsedDocument
```

**Sobrevive como validação/fallback**:
```python
resolve_child_offsets()            # Double-check do bbox matching OU fallback
OffsetResolutionError              # Diagnóstico de falhas
resolve_offsets_recursive()        # Pode ser reutilizado se Reconciliator precisar
```

---

## 4. Novos Componentes

### 4.1 PyMuPDF Extractor

**Responsabilidade**: Extrair texto determinístico + coordenadas de blocos do PDF.

**Output**:
```python
@dataclass
class PyMuPDFResult:
    canonical_text: str              # Texto completo, determinístico
    canonical_hash: str              # SHA256 do texto normalizado
    pages: list[PageBlocks]          # Blocos por página com coordenadas

@dataclass
class TextBlock:
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    page_number: int
    block_index: int
    char_start: int                  # Offset no canonical_text
    char_end: int                    # Offset no canonical_text
```

**Propriedade crítica**: O canonical_text é DETERMINÍSTICO — mesmo PDF sempre gera mesmo texto, byte a byte. Isso torna o hash anti-mismatch muito mais confiável do que o canonical construído pelo PR12.

**Localização**: `src/extraction/pymupdf_extractor.py`

### 4.2 VLM Service (Qwen3-VL)

**Responsabilidade**: Receber imagem da página do PDF e retornar estrutura hierárquica com bounding boxes.

**Input**: Imagem da página (renderizada via PyMuPDF)
**Output**:
```python
@dataclass
class VLMElement:
    type: str                        # "article", "paragraph", "inciso", "alinea", "chapter"
    number: str                      # "1", "2", "I", "a", "único"
    text: str                        # Texto do dispositivo (OCR do VLM)
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 (coordenadas na página)
    parent: Optional[str]            # Referência ao pai (ex: "ART-001")
    page_number: int
    confidence: float                # Confiança do VLM na classificação

@dataclass
class VLMPageResult:
    page_number: int
    elements: list[VLMElement]
    raw_json: dict                   # Resposta bruta do VLM para debug
```

**Deploy**: vLLM no RunPod A40 — `vllm serve "Qwen/Qwen3-VL-8B-Instruct" --max_model_len 8096`

**Prompt de extração**: Deve solicitar JSON estruturado com type, number, bbox, parent. Instruct variant (sem `<think>` overhead).

**Localização**: `src/extraction/vlm_service.py`

### 4.3 Reconciliator

**Responsabilidade**: Módulo CENTRAL da migração. Ponte entre VLM (estrutura visual) e PyMuPDF (texto determinístico).

**Funções**:
1. Mapear cada bbox do VLM para text blocks do PyMuPDF (matching por sobreposição de coordenadas)
2. Gerar `node_id` canônico a partir do tipo/número/pai retornado pelo VLM
3. Calcular `canonical_start` e `canonical_end` (offsets no canonical_text do PyMuPDF)
4. Construir relação `parent_id` → `child_ids` para hierarquia
5. Validar coerência: o texto no offset corresponde ao que o VLM identificou?

**Interface**:
```python
@dataclass
class ReconciledChunk:
    node_id: str                     # "leis:LEI-14133-2021#ART-023"
    chunk_id: str                    # "LEI-14133-2021#ART-023"
    parent_id: Optional[str]         # "leis:LEI-14133-2021#CAP-V"
    span_id: str                     # "ART-023"
    text: str                        # Texto exato do PyMuPDF (não do VLM OCR)
    device_type: str                 # "article", "paragraph", "inciso", "alinea"
    canonical_start: int             # Offset no canonical_text
    canonical_end: int               # Offset no canonical_text
    canonical_hash: str              # SHA256 do canonical_text
    page_number: int
    bbox: tuple[float, float, float, float]  # Coordenadas físicas no PDF
    confidence: float                # Confiança herdada do VLM

class Reconciliator:
    def __init__(
        self,
        pymupdf_result: PyMuPDFResult,
        vlm_results: list[VLMPageResult],
        document_id: str,
    ): ...

    def reconcile(self) -> list[ReconciledChunk]: ...
    def reconcile_element(self, element: VLMElement) -> Optional[ReconciledChunk]: ...
```

**Estratégia de matching bbox → text blocks**:
```
1. Para cada VLMElement:
   a. Encontrar text blocks do PyMuPDF cuja bbox tem IoU > threshold com bbox do VLM
   b. Concatenar texto desses blocks → texto do chunk
   c. Localizar esse texto no canonical_text → char_start, char_end
   d. Se matching direto falha → usar resolve_child_offsets() como fallback
   e. Se fallback também falha → OffsetResolutionError → alarme
```

**Localização**: `src/extraction/reconciliator.py`

### 4.4 Integrity Validator (Pós-Reconciliação)

**Responsabilidade**: Camada adicional de validação que opera ENTRE o Reconciliator e o PR13. Verifica invariantes que dependem da relação entre VLM output e PyMuPDF output.

**Validações**:
```python
class IntegrityValidator:
    def validate_reconciled_chunks(
        self,
        chunks: list[ReconciledChunk],
        canonical_text: str,
    ) -> IntegrityResult:
        """
        Validações:
        1. Slicing: canonical_text[start:end] começa com primeiras palavras do chunk.text
        2. Hierarquia: todo filho tem start >= parent.start e end <= parent.end
        3. Sem sobreposição: chunks irmãos não se sobrepõem
        4. Cobertura: artigo cobre todos seus filhos
        5. Ordenação: offsets são monotonicamente crescentes dentro de cada nível
        """
```

**Localização**: `src/extraction/integrity_validator.py`

---

## 5. Novo Fluxo Integrado

```
PDF
 │
 ├──→ PyMuPDF Extractor
 │      ├── canonical_text (determinístico)
 │      ├── canonical_hash (SHA256 via canonical_utils)
 │      └── text_blocks[] (texto + bbox + char_start/char_end por bloco)
 │
 └──→ Qwen3-VL Service (RunPod A40)
        └── vlm_elements[] (type + number + bbox + parent + confidence por dispositivo)
                │
                ▼
        ┌─ Reconciliator ──────────────────────┐
        │  1. bbox VLM ↔ text blocks PyMuPDF   │
        │  2. canonical_start/end por chunk     │
        │  3. node_id canônico                  │
        │  4. parent-child hierarchy            │
        │  5. texto = PyMuPDF (não VLM OCR)     │
        └──────────┬───────────────────────────┘
                   │
                   ▼
        ┌─ IntegrityValidator ────────────────┐
        │  Slicing, hierarquia, sobreposição   │
        └──────────┬──────────────────────────┘
                   │
                   ▼
        ┌─ canonical_validation.py ──────────┐  ← EXISTENTE, intacto
        │  node_id format (leis:/acordaos:)  │
        │  sem @P, logical_node_id filled    │
        └──────────┬─────────────────────────┘
                   │
                   ▼
        ┌─ pr13_validator.py ────────────────┐  ← EXISTENTE, intacto
        │  trio Evidence gate                │
        │  abort doc se violação             │
        │  cria alarme via alarm_service     │
        └──────────┬─────────────────────────┘
                   │
                   ▼
        ┌─ alarm_service.py ─────────────────┐  ← EXISTENTE, intacto
        │  persiste no PostgreSQL            │
        │  bloqueia docs com CRITICAL alarms │
        └────────────────────────────────────┘
                   │
                   ▼
        ┌─ BGE-M3 (RunPod A40) ─────────────┐
        │  dense + sparse embeddings         │
        └──────────┬─────────────────────────┘
                   │
                   ▼
                Milvus ✅  +  Neo4j (hierarquia)
                   │
              (query-time)
                   │
                   ▼
        ┌─ snippet_extractor.py ─────────────┐  ← EXISTENTE, intacto
        │  slicing puro + hash anti-mismatch │
        │  fallback find() se necessário     │
        └──────────┬─────────────────────────┘
                   │
                   ▼
             Evidence Link 🎯
             (page_number + bbox → highlight no PDF no MinIO)
```

---

## 6. Chunk Schema Final

```python
{
    # Identificação
    "node_id": "leis:LEI-14133-2021#ART-023-PAR-002",
    "chunk_id": "LEI-14133-2021#ART-023-PAR-002",
    "logical_node_id": "leis:LEI-14133-2021#ART-023-PAR-002",
    "document_id": "LEI-14133-2021",
    "span_id": "PAR-023-2",

    # Texto (fonte: PyMuPDF, NÃO VLM OCR)
    "text": "§ 2º O processo de contratação direta...",

    # Hierarquia
    "parent_id": "leis:LEI-14133-2021#ART-023",
    "device_type": "paragraph",
    "chunk_level": 3,  # 1=cap, 2=art, 3=§/inciso

    # Evidence Trio (PR13)
    "canonical_start": 1847,
    "canonical_end": 2103,
    "canonical_hash": "a1b2c3d4...",  # SHA256 do canonical_text

    # Localização física no PDF (NOVO — não existia no pipeline anterior)
    "page_number": 12,
    "bbox": [72.0, 340.5, 520.3, 410.8],  # x0, y0, x1, y1

    # Embeddings (BGE-M3)
    "dense_vector": [...],
    "sparse_vector": {...},

    # Metadados
    "tipo_documento": "LEI",
    "numero": "14133",
    "ano": 2021,
    "confidence": 0.97,  # Confiança do VLM na classificação
}
```

**Diferenças vs. schema anterior**:
- `page_number` e `bbox`: NOVOS — permitem highlight visual direto no PDF (MinIO), sem depender apenas de char offsets
- `confidence`: NOVO — permite filtrar/alertar chunks com baixa confiança do VLM
- `text`: fonte muda de "canonical_builder markdown" para "PyMuPDF raw text"

---

## 7. Fases de Implementação

### FASE 0 — Preparação ✅ CONCLUÍDA (2026-02-06)

**Objetivo**: Reorganizar código existente sem quebrar nada.

**Status**: CONCLUÍDA. Todas as tarefas essenciais foram executadas. Testes: 294 passed, 34 skipped, 0 failures.

**Tarefas realizadas**:

1. ✅ **Extrair `canonical_utils.py`** de `canonical_offsets.py` (NÃO de `canonical_builder.py` — esse arquivo não existe no codebase):
   - Criado `src/utils/canonical_utils.py` com `normalize_canonical_text()`, `compute_canonical_hash()`, `validate_offsets_hash()`
   - `canonical_offsets.py` agora importa e re-exporta de `canonical_utils.py` (backward compatibility)
   - Import usa try/except para suportar testes que carregam módulos diretamente (sem package)
   - Atualizado `src/utils/__init__.py` com novos exports
   - **Arquivos modificados**: `src/utils/canonical_utils.py` (NOVO), `src/utils/__init__.py`, `src/chunking/canonical_offsets.py`

2. ✅ **Criar estrutura de diretórios**:
   - Criado `src/extraction/__init__.py` com docstring descrevendo os módulos futuros
   ```
   src/
   ├── extraction/          # NOVO (placeholder para Fases 1-3)
   │   └── __init__.py
   ├── utils/
   │   ├── canonical_utils.py  # NOVO (extraído de canonical_offsets)
   │   └── normalization.py    # Existente
   ├── chunking/            # EXISTENTE (deprecar gradualmente)
   │   ├── canonical_offsets.py  # Agora re-exporta de canonical_utils
   │   └── chunk_materializer.py
   ├── ingestion/           # EXISTENTE
   │   └── models.py        # Atualizado com campos VLM
   └── services/            # EXISTENTE
   ```

3. ✅ **Atualizar `ProcessedChunk` model** em `ingestion/models.py`:
   - Adicionados campos: `page_number: int = Field(-1)`, `bbox: list[float] = Field(default_factory=list)`, `confidence: float = Field(0.0)`
   - Todos os campos existentes mantidos intactos
   - **Arquivo modificado**: `src/ingestion/models.py`

4. ✅ **Atualizar schema Milvus `leis_v4`** (mantém nome `leis_v4`, NÃO renomear — hardcoded em 12+ locais):
   - Adicionados ao `docs/leis_v4.json`: `page_number` (Int64, fieldID 125), `bbox` (VarChar max 256 para JSON, fieldID 126), `confidence` (Float, fieldID 127)
   - Adicionados ao dict de inserção Milvus em `pipeline.py` (linhas ~1310-1316)
   - **IMPORTANTE**: A collection Milvus precisa ser recriada para incluir os novos campos. O JSON é a referência para recriação.
   - **Arquivos modificados**: `docs/leis_v4.json`, `src/ingestion/pipeline.py`

5. ⏳ **Compatibilidade MinIO Evidence Storage** — ADIADA para fases posteriores:
   - `integrity_validator.py` Layer 6 NÃO existe no codebase atual
   - `EvidenceResponse` em `evidence/models.py` NÃO existe no codebase atual
   - `docs/ANALISE_MINIO_EVIDENCE_STORAGE.md` NÃO existe no codebase atual
   - Esses componentes serão criados nas Fases 3-4 quando o pipeline VLM estiver funcional

**Discrepâncias encontradas entre plano e codebase**:
- `canonical_builder.py` não existe — funções estavam em `canonical_offsets.py`
- `pr13_validator.py` não existe como módulo separado — validação está em `pipeline.py:validate_chunk_invariants()`
- `snippet_extractor.py` não existe — é `extract_snippet_by_offsets()` dentro de `canonical_offsets.py`
- `alarm_service.py` não existe no codebase atual
- `evidence/models.py` e `EvidenceResponse` não existem no codebase atual

**Testes executados**:
- `test_pr13_canonical_offsets.py`: 27 passed ✅
- `test_pr13_offset_resolution.py`: todos passed ✅
- `test_pr13_span_parser_offsets.py`: todos passed ✅
- `test_pr13_materializer_integration.py`: todos passed ✅
- `test_ingestion_pipeline.py`: 17 passed, 2 skipped ✅
- Suite completa: **294 passed, 34 skipped, 0 failures** ✅
- Erros de coleção pré-existentes (não causados pela Fase 0): `test_article_validator.py` (fastapi ausente), `test_chunk_materializer_split.py` (import direto), `test_rel_type_classification.py` (prometheus registry)

**Critério de conclusão**: ✅ Todos os testes existentes passam, nova estrutura de diretórios existe, model atualizado com campos VLM, schema Milvus documentado com novos campos.

---

### FASE 1 — PyMuPDF Extractor (2-3 dias)

**Objetivo**: Substituir Docling como fonte de texto. Ter canonical_text determinístico.

**Tarefas**:
1. **Implementar `pymupdf_extractor.py`**:
   ```python
   class PyMuPDFExtractor:
       def extract(self, pdf_path: str) -> PyMuPDFResult:
           """
           1. Abrir PDF com fitz (PyMuPDF)
           2. Para cada página:
              a. Extrair text blocks com get_text("dict") → blocos com bbox
              b. Extrair texto corrido com get_text("text") → para canonical_text
           3. Construir canonical_text concatenando texto de todas as páginas
           4. Calcular char_start/char_end de cada bloco no canonical_text
           5. Normalizar via normalize_canonical_text()
           6. Computar canonical_hash via compute_canonical_hash()
           """

       def render_page_image(self, pdf_path: str, page_num: int, dpi: int = 300) -> bytes:
           """Renderiza página como imagem para enviar ao VLM."""
   ```

2. **Testes**:
   - Determinismo: extrair mesmo PDF 3x → mesmo canonical_text e hash
   - Offsets: char_start/char_end de cada bloco apontam pro texto correto via slicing
   - Cobertura: todos os blocos de texto estão representados no canonical_text
   - Edge cases: PDFs com múltiplas colunas, tabelas, headers/footers

3. **Teste de comparação**: Extrair mesmo documento com Docling e PyMuPDF, comparar qualidade do texto

4. **Definir formato do canonical.md**:
   - O canonical.md será texto puro extraído pelo PyMuPDF (não markdown construído pelo canonical_builder)
   - Confirmar que `normalize_canonical_text()` (em `canonical_utils.py`) funciona com texto PyMuPDF — a função já é agnóstica ao formato (strip + lower)

5. **Teste de determinismo do canonical.md**:
   - Mesmo PDF extraído 3x pelo PyMuPDF → mesmo texto → mesmo SHA-256 hash
   - Texto extraído deve ser estável entre versões do PyMuPDF (fixar versão no requirements)

**Critério de conclusão**: PyMuPDF extrai texto determinístico de PDFs legais brasileiros com offsets corretos por bloco. Hash é estável. Formato do canonical.md definido como texto puro PyMuPDF.

**Dependências**: `pip install PyMuPDF` (já disponível no ambiente)

---

### FASE 2 — VLM Service (3-5 dias)

**Objetivo**: Ter o Qwen3-VL rodando no RunPod e retornando hierarquia estruturada com bbox.

**Tarefas**:
1. **Deploy do Qwen3-VL no RunPod**:
   ```bash
   vllm serve "Qwen/Qwen3-VL-8B-Instruct" --max_model_len 8096 --gpu-memory-utilization 0.35
   ```
   - Configurar endpoint HTTP
   - Testar health check e throughput

2. **Implementar `vlm_service.py`**:
   ```python
   class VLMService:
       def __init__(self, endpoint_url: str): ...

       def extract_structure(self, page_image: bytes, page_number: int) -> VLMPageResult:
           """
           1. Enviar imagem da página para Qwen3-VL via API (vLLM OpenAI-compatible)
           2. Prompt: solicitar JSON com dispositivos legais, seus tipos, números, bbox, hierarquia
           3. Parsear resposta JSON
           4. Validar: cada elemento tem type, number, bbox válidos
           5. Retornar VLMPageResult
           """

       def extract_document(self, page_images: list[bytes]) -> list[VLMPageResult]:
           """Extrai estrutura de todas as páginas (batch processing)."""
   ```

3. **Engenharia de Prompt** — Desenvolver e testar prompt que retorne:
   ```json
   {
     "elements": [
       {
         "type": "article",
         "number": "23",
         "text": "Art. 23. O processo de contratação direta...",
         "bbox": [72.0, 120.5, 520.3, 180.8],
         "parent": null,
         "confidence": 0.98
       },
       {
         "type": "paragraph",
         "number": "1",
         "text": "§ 1º A contratação direta será processada...",
         "bbox": [72.0, 185.0, 520.3, 240.2],
         "parent": "ART-023",
         "confidence": 0.96
       }
     ]
   }
   ```

4. **Testes**:
   - Accuracy: VLM identifica corretamente artigos, §§, incisos, alíneas de trechos da Lei 14.133
   - Bbox precision: bbox retornado cobre o texto visualmente (sem cortar, sem excesso)
   - Hierarquia: parent correto (inciso do § aponta pro §, não direto pro artigo)
   - Edge cases: artigos longos que continuam na página seguinte, tabelas legais

**Critério de conclusão**: VLM extrai hierarquia de páginas legais com >90% accuracy em tipo/número e bbox utilizável para matching.

**Dependências**: RunPod com A40, vLLM ≥ 0.11.0, Qwen3-VL-8B-Instruct

---

### FASE 3 — Reconciliator (5-7 dias)

**Objetivo**: Módulo central que une VLM + PyMuPDF em chunks com offsets válidos.

**Tarefas**:
1. **Implementar bbox matching**:
   ```python
   def match_bbox_to_text_blocks(
       self,
       vlm_bbox: tuple[float, float, float, float],
       page_blocks: list[TextBlock],
       iou_threshold: float = 0.3,
   ) -> list[TextBlock]:
       """
       Encontra text blocks do PyMuPDF cuja bbox tem IoU (Intersection over Union)
       acima do threshold com a bbox do VLM.

       Retorna blocos ordenados por posição vertical (top→bottom).
       """
   ```

2. **Implementar construção de offsets**:
   ```python
   def build_offsets(
       self,
       matched_blocks: list[TextBlock],
       canonical_text: str,
   ) -> tuple[int, int]:
       """
       Dado os blocos matchados, determina char_start e char_end no canonical_text.

       Estratégia:
       1. Se blocos são contíguos: start = primeiro.char_start, end = último.char_end
       2. Se há gap entre blocos: concatenar texto e usar resolve_child_offsets() como validação
       3. Sempre validar: canonical_text[start:end] contém texto esperado
       """
   ```

3. **Implementar construção de node_id**:
   ```python
   def build_node_id(
       self,
       element: VLMElement,
       document_id: str,
   ) -> str:
       """
       Constrói node_id canônico a partir do output do VLM.

       Exemplos:
       - type="article", number="23" → "leis:LEI-14133-2021#ART-023"
       - type="paragraph", number="1", parent="ART-023" → "leis:LEI-14133-2021#PAR-023-1"
       - type="inciso", number="V", parent="ART-023" → "leis:LEI-14133-2021#INC-023-V"
       """
   ```

4. **Implementar resolução de conflitos**:
   ```python
   def resolve_conflicts(self, chunks: list[ReconciledChunk]) -> list[ReconciledChunk]:
       """
       Resolve conflitos quando:
       - Dois chunks têm offsets sobrepostos
       - VLM classifica mesmo trecho de duas formas (baixa confiança)
       - Artigo continua na próxima página (cross-page)
       """
   ```

5. **Integrar `resolve_child_offsets()` como fallback/validação**:
   ```python
   def reconcile_element(self, element: VLMElement) -> Optional[ReconciledChunk]:
       # Passo 1: bbox matching
       matched = self.match_bbox_to_text_blocks(element.bbox, page_blocks)

       if matched:
           start, end = self.build_offsets(matched, self.canonical_text)

           # Passo 2: validação via resolve_child_offsets (double-check)
           if element.parent:
               parent_start, parent_end = self.get_parent_range(element.parent)
               try:
                   verified_start, verified_end = resolve_child_offsets(
                       canonical_text=self.canonical_text,
                       parent_start=parent_start,
                       parent_end=parent_end,
                       chunk_text=self.canonical_text[start:end],
                   )
                   return ReconciledChunk(start=verified_start, end=verified_end, ...)
               except OffsetResolutionError:
                   logger.warning(f"Double-check falhou para {element.type} {element.number}")
                   # Usa offset do bbox matching mesmo assim
                   return ReconciledChunk(start=start, end=end, ...)

       # Passo 3: fallback puro via find()
       if element.parent:
           parent_start, parent_end = self.get_parent_range(element.parent)
           start, end = resolve_child_offsets(
               canonical_text=self.canonical_text,
               parent_start=parent_start,
               parent_end=parent_end,
               chunk_text=element.text,
           )
           return ReconciledChunk(start=start, end=end, ...)

       return None  # Falha total → alarme
   ```

6. **Testes (nível industrial)**:
   - Matching: bbox com IoU > 0.3 retorna blocos corretos
   - Offsets: slicing `canonical_text[start:end]` retorna texto esperado para cada tipo de dispositivo
   - Hierarquia: filhos dentro do range do pai (reutilizar lógica dos testes PR13 existentes)
   - Cross-page: artigo que começa na página 5 e termina na 6
   - Edge cases: artigo com 20+ incisos, alínea dupla, parágrafo único
   - Trecho real da Lei 14.133/2021 (reutilizar `LEI_14133_EXCERPT` dos testes existentes)
   - Fallback: quando bbox matching falha, `resolve_child_offsets()` resolve

7. **Gerar offsets.json**:
   - Reconciliator deve produzir offsets.json com objetos por dispositivo contendo: `start`, `end`, `page_number`, `bbox`, `confidence`, `device_type`, `parent_id`
   - Incluir `extraction_method: "pymupdf+qwen3vl"` no header do JSON
   - Validar que todo node_id dos chunks tem entrada correspondente no offsets.json

**Critério de conclusão**: Reconciliator gera chunks com trio Evidence válido (testado pelo PR13 validator) para >95% dos dispositivos de um trecho real da Lei 14.133. offsets.json contém campos extras por dispositivo.

---

### FASE 4 — Integração no Pipeline (3-4 dias)

**Objetivo**: Conectar novos componentes ao pipeline de ingestão existente, passando pelos gates de segurança.

**Tarefas**:
1. **Implementar `IntegrityValidator`**:
   - Slicing validation: `canonical_text[start:end]` contém primeiras palavras do chunk
   - Hierarquia: filhos dentro do range do pai
   - Sem sobreposição entre irmãos
   - Cobertura: artigo cobre todos seus filhos
   - Ordenação monotônica de offsets

2. **Criar endpoint de ingestão**:
   ```python
   @router.post("/ingest")
   async def ingest_document(pdf_file: UploadFile):
       # 1. PyMuPDF extrai texto + coordenadas
       pymupdf_result = pymupdf_extractor.extract(pdf_path)

       # 2. Renderiza páginas como imagens
       page_images = [pymupdf_extractor.render_page_image(pdf_path, i) for i in range(num_pages)]

       # 3. VLM extrai estrutura hierárquica
       vlm_results = vlm_service.extract_document(page_images)

       # 4. Reconciliator une VLM + PyMuPDF
       reconciliator = Reconciliator(pymupdf_result, vlm_results, document_id)
       chunks = reconciliator.reconcile()

       # 5. IntegrityValidator
       integrity_result = integrity_validator.validate(chunks, pymupdf_result.canonical_text)
       if not integrity_result.valid:
           # Alarme WARNING (não bloqueia, mas registra)
           alarm_service.create_alarm(...)

       # 6. canonical_validation (EXISTENTE)
       for chunk in chunks:
           fixed, warnings = validate_and_fix_chunk(chunk.to_dict())

       # 7. PR13 gate (EXISTENTE)
       pr13_result = pr13_validator.validate_chunks(document_id, [c.to_dict() for c in chunks])
       if not pr13_result.valid:
           pr13_validator.create_alarm(db, pr13_result)
           raise HTTPException(422, pr13_result.to_error_response())

       # 8. BGE-M3 embeddings
       embeddings = bge_m3.encode([c.text for c in chunks])

       # 9. Insert Milvus
       milvus_service.insert(collection="leis_v4", chunks=chunks, embeddings=embeddings)

       # 10. Insert Neo4j (hierarquia)
       neo4j_service.insert_hierarchy(chunks)

       return {"status": "ok", "chunks": len(chunks)}
   ```

3. **Atualizar Evidence Link flow** para usar bbox:
   ```python
   # Query-time: snippet_extractor (EXISTENTE) + bbox highlight (NOVO)
   def build_evidence_response(chunk, canonical_text):
       # Snippet via offsets (existente)
       snippet_result = get_snippet_from_chunk(
           canonical_text=canonical_text,
           chunk_text=chunk.text,
           stored_start=chunk.canonical_start,
           stored_end=chunk.canonical_end,
           stored_hash=chunk.canonical_hash,
       )

       # PDF highlight via bbox (NOVO)
       pdf_url = f"{MINIO_URL}/{chunk.document_id}.pdf#page={chunk.page_number}"

       return {
           "snippet": snippet_result.snippet if snippet_result else None,
           "evidence": {
               "document_id": chunk.document_id,
               "page_number": chunk.page_number,
               "bbox": chunk.bbox,           # Para highlight visual no frontend
               "char_start": chunk.canonical_start,
               "char_end": chunk.canonical_end,
               "pdf_url": pdf_url,
           }
       }
   ```

4. **Testes de integração end-to-end**:
   - PDF real da Lei 14.133 → pipeline completo → chunks no Milvus com trio válido
   - Query → busca → snippet + bbox → evidence link funcional
   - Documento inválido → PR13 rejeita → alarme no PostgreSQL → zero rows no Milvus

5. **Atualizar código de upload MinIO** (ref: `docs/ANALISE_MINIO_EVIDENCE_STORAGE.md`):
   - GPU Server (`storage/object_storage.py`): upload do canonical.md (texto PyMuPDF) para bucket `rag-documents`
   - VPS (`evidence/storage_service.py`): upload do canonical.md + offsets.json para bucket `vectorgov-evidence`
   - Ambos os serviços são agnósticos ao formato (aceitam bytes) — a mudança é no conteúdo, não no código de upload

6. **Validar evidence link completo**:
   - Verificar que `canonical_text[start:end]` retorna snippet correto (sem fallback `find()`)
   - Verificar que `IntegrityValidator` não gera alarmes
   - Verificar que frontend recebe `page_number` e `bbox` para highlight no PDF

**Critério de conclusão**: Pipeline funciona end-to-end com pelo menos 1 documento real. Gates de segurança rejeitam corretamente chunks inválidos. Evidence links funcionam com snippet + bbox highlight.

---

### FASE 5 — Validação e Produção (2-3 dias)

**Objetivo**: Validar qualidade em escala e preparar para produção.

**Tarefas**:
1. **Benchmark de qualidade**:
   - Processar 10+ documentos legais reais (leis, decretos, acórdãos)
   - Medir: % de dispositivos corretamente identificados pelo VLM
   - Medir: % de offsets válidos (trio PR13 passa)
   - Medir: % de evidence links funcionais (snippet + bbox)

2. **Stress test no RunPod**:
   - Processar documento de 200+ páginas
   - Medir throughput (páginas/minuto)
   - Monitorar VRAM (Qwen3-VL + BGE-M3 simultâneos)
   - Testar recovery de falhas (VLM timeout, OOM)

3. **Monitoramento em produção**:
   - Dashboard de alarmes (alarm_service stats)
   - Métricas de confiança do VLM por documento
   - Alertas quando % de chunks com `confidence < 0.8` ultrapassa threshold

**Critério de conclusão**: Pipeline em produção, processando documentos reais, com métricas atingindo targets definidos.

---

## 8. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| VLM bbox impreciso para texto pequeno (alíneas) | Média | Alto | `resolve_child_offsets()` como fallback; threshold de confiança |
| Artigo cross-page (começa pág. 5, termina pág. 6) | Alta | Médio | Reconciliator deve juntar elementos de páginas adjacentes quando VLM detecta continuação |
| Tabelas legais (tabela de valores em lei) | Média | Médio | Tratar como dispositivo especial; PyMuPDF `get_text("dict")` lida bem com tabelas |
| VRAM insuficiente no A40 para batch grande | Baixa | Alto | `--max_model_len 8096` limita contexto; processar 1 página por vez se necessário |
| Latência alta do VLM (>5s por página) | Média | Baixo | Ingestão é batch, não real-time; aceitável para processamento de documentos |
| VLM alucina dispositivo inexistente | Baixa | Alto | IntegrityValidator + PR13 gate rejeitam chunks com offsets inválidos |
| Normalização markdown-specific no IntegrityValidator Layer 6 | Média | Médio | Remover na Fase 0: strip `"- "` prefix e regex de espaços em hífens romanos são artefatos do canonical_builder, não existem em texto PyMuPDF |

---

## 9. Métricas de Sucesso

| Métrica | Target | Como medir |
|---------|--------|------------|
| Dispositivos identificados | >95% | Manual review de 5 documentos |
| ADDRESS_MISMATCH rate | <0.5% | Alarmes PR13 / total chunks |
| Evidence link funcional | >98% | snippet_extractor retorna `found=True` |
| Offsets determinísticos | Sim | Hash anti-mismatch estável em re-ingestão |
| Throughput | >5 págs/min | Cronometrar ingestão de documento |

---

## 10. Testes a Reutilizar

Os testes PR13 existentes são valiosos e devem ser adaptados:

| Arquivo de Teste | Status | Adaptação |
|-----------------|--------|-----------|
| `test_pr13_canonical_offsets.py` | ✅ Reutilizar | Trocar MockParsedDocument por MockReconciledChunks |
| `test_pr13_materializer_integration.py` | 🔄 Adaptar | Substituir ChunkMaterializer por Reconciliator nos testes |
| `test_pr13_offset_resolution.py` | ✅ Reutilizar | `resolve_child_offsets()` continua existindo como fallback |
| `test_pr13_span_parser_offsets.py` | 🔄 Adaptar | Trocar SpanParser por VLM+Reconciliator; manter mesmas asserções de hierarquia |

**Princípio**: As ASSERÇÕES dos testes (filhos dentro do pai, slicing correto, hash determinístico) são independentes do parser. Mudam os fixtures, não as validações.

---

## 11. Ordem de Execução para Claude Code

```
FASE 0: Preparação ✅ CONCLUÍDA (2026-02-06)
├── ✅ Criar src/utils/canonical_utils.py (extraído de canonical_offsets.py)
├── ✅ Atualizar imports com backward compatibility (re-export + try/except)
├── ✅ Criar src/extraction/__init__.py
├── ✅ Atualizar ProcessedChunk model (page_number, bbox, confidence)
├── ✅ Atualizar schema Milvus leis_v4 (mantém nome, adiciona campos VLM)
├── ✅ Atualizar dict de inserção Milvus em pipeline.py
├── ⏳ Compatibilidade MinIO: ADIADA (módulos não existem ainda)
└── ✅ Rodar testes existentes → 294 passed, 0 failures

FASE 1: PyMuPDF Extractor
├── Implementar src/extraction/pymupdf_extractor.py
├── Definir formato do canonical.md (texto puro PyMuPDF)
├── Testes de determinismo e offsets
└── Teste de comparação com Docling

FASE 2: VLM Service
├── Implementar src/extraction/vlm_service.py
├── Desenvolver prompt de extração estruturada
├── Testes de accuracy e bbox
└── Deploy no RunPod (pode ser em paralelo com Fase 3)

FASE 3: Reconciliator
├── Implementar src/extraction/reconciliator.py
├── Bbox matching + offset building + node_id construction
├── Integrar resolve_child_offsets() como fallback
├── Gerar offsets.json com campos extras por dispositivo
├── Testes nível industrial com trecho real Lei 14.133
└── Implementar src/extraction/integrity_validator.py

FASE 4: Integração
├── Endpoint /ingest
├── Conectar ao PR13 + canonical_validation + alarm_service
├── Atualizar evidence link flow com bbox
├── Atualizar código de upload MinIO (GPU Server + VPS)
├── Testes end-to-end
└── Validar evidence link completo (snippet + bbox highlight)

FASE 5: Validação e Produção
├── Benchmark com 10+ documentos
├── Stress test RunPod
└── Monitoramento
```

---

*Documento gerado para consumo pelo Claude Code na IDE. Cada fase é independente e testável. Os gates de segurança (PR13, alarm_service, canonical_validation, snippet_extractor) não são alterados — o Reconciliator é o único componente novo que produz dados para esses gates.*
