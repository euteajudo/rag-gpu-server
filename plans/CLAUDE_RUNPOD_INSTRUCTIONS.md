# Pipeline Inspector — Briefing RunPod (Atualizado 2026-02-06)

> **Ultima atualizacao**: 2026-02-06
> **Status geral**: Fases 1-5 do inspection pipeline IMPLEMENTADAS. Falta: instalar Redis, configurar MINIO_ENDPOINT, testar end-to-end.

---

## 0. Resumo do que ja foi feito

### 0.1 Extraction Pipeline (src/extraction/) — COMPLETO
Criados 5 arquivos novos para o pipeline VLM:

| Arquivo | Linhas | Descricao |
|---------|--------|-----------|
| `vlm_models.py` | 53 | Modelos: PageData, DeviceExtraction, PageExtraction, DocumentExtraction |
| `vlm_prompts.py` | 54 | SYSTEM_PROMPT e PAGE_PROMPT_TEMPLATE para Qwen3-VL |
| `vlm_client.py` | 239 | Cliente async multimodal (httpx) para vLLM com retry e strip de `<think>` |
| `pymupdf_extractor.py` | 103 | Extrai paginas: PNG 300 DPI + texto nativo get_text("text") |
| `vlm_service.py` | 184 | Orquestrador: PyMuPDF -> VLM -> DocumentExtraction |

Modificados:
- `__init__.py` — exports publicos
- `src/config.py` — campos VLM (use_vlm_pipeline, vlm_page_dpi, vlm_max_retries)
- `src/ingestion/pipeline.py` — branch por feature flag USE_VLM_PIPELINE
- `src/main.py` — log do pipeline ativo no startup

### 0.2 Inspection Pipeline (src/inspection/) — COMPLETO
O pipeline.py foi **completamente reescrito** (614 -> ~1074 linhas):

| Fase | Status | O que faz |
|------|--------|-----------|
| 1. PyMuPDF | FUNCIONAL | Extrai blocos com bboxes via PageRenderer (150 DPI display) |
| 2. VLM | **IMPLEMENTADO** | Qwen3-VL real via VLMClient (300 DPI), bbox normalizadas -> PDF points |
| 3. Reconciliation | **IMPLEMENTADO** | Bbox IoU matching PyMuPDF <-> VLM, texto SEMPRE do PyMuPDF |
| 4. Integrity | **EXPANDIDO** | 5 checks (adicionado vlm_matches check) |
| 5. Chunks | **REESCRITO** | Gera ChunkPreview a partir de VLM devices + PyMuPDF text |

**Docling e SpanParser foram 100% removidos** do inspection pipeline.

### 0.3 Correcoes no router.py
- Adicionado `task_id=task_id` no call a `pipeline.run()` (era bug)
- Thresholds de progresso atualizados: pymupdf=0.25, vlm=0.55, reconciliation=0.70, integrity=0.80, chunks=0.95

---

## 1. Arquitetura de Infraestrutura

### 1.1 Tres camadas

```
Frontend (Next.js)                VPS (FastAPI)               GPU Server (RunPod)
 https://vectorgov.io             77.37.43.160                gpu.vectorgov.io
        |                              |                              |
   React UI                       Proxy + MinIO               Pipeline + Redis
   (upload PDF,                   + Milvus + Neo4j            + vLLM (Qwen3-VL)
    visualizar fases,             + PostgreSQL                + BGE-M3
    aprovar)                                                  + PyMuPDF
```

### 1.2 Onde cada servico roda

| Servico | Onde roda | Porta | Usado por |
|---------|-----------|-------|-----------|
| **Redis** | **RunPod (localhost)** | 6379 | Inspection storage temporario (TTL 2h, DB 2) |
| **MinIO** | **VPS** | 9000 | Storage permanente (PDFs, artefatos aprovados) |
| **vLLM** | RunPod (GPU) | 8002 | Qwen3-VL para extracao VLM |
| **BGE-M3** | RunPod (GPU) | interno | Embeddings dense+sparse |
| **Milvus** | VPS | 19530 | Vector DB para chunks |
| **Neo4j** | VPS | 7687 | Hierarquia de dispositivos |
| **PostgreSQL** | VPS | 5432 | Alarmes, metadados |

### 1.3 Decisoes de infraestrutura

**Redis NO RunPod (nao na VPS)**:
- Artefatos temporarios podem ser grandes (imagens base64, ~3-5MB por pagina)
- TTL 2h — nao precisa persistencia
- Latencia zero (localhost) durante processamento das 5 fases
- Se RunPod reinicia, dados temporarios se perdem (OK — usuario re-submete)

**MinIO SO na VPS**:
- Artefatos aprovados sao permanentes
- O upload de documentos sempre vem da interface React (VPS)
- Re-ingestao: usuario busca doc ja aprovado no MinIO da VPS e envia pro RunPod
- `MINIO_ENDPOINT` no RunPod aponta para VPS (ex: `77.37.43.160:9000`)

---

## 2. Fluxo de dados detalhado

### 2.1 Inspecao (dry-run)

```
[RunPod GPU Server - tudo local]

PDF bytes (recebido via HTTP da VPS)
  |
  v
Fase 1: PyMuPDF (CPU)
  |-- get_text("dict") -> blocos com bbox (PDF points, 72 dpi)
  |-- get_pixmap(dpi=150) -> PNG para display
  |-- Salva PyMuPDFArtifact no Redis (localhost) --> retorna objeto em memoria
  |
  v
Fase 2: VLM (GPU via vLLM localhost:8002)
  |-- pymupdf_extractor.extract_pages(pdf_bytes) -> PNG 300 DPI + texto
  |-- Para cada pagina: HTTP POST localhost:8002/v1/chat/completions
  |     image_base64 -> Qwen3-VL "ve" a pagina -> retorna JSON com devices
  |-- Converte bbox normalizada [0-1] -> PDF points (* page_width, * page_height)
  |-- Renderiza bboxes verdes nas imagens
  |-- Salva VLMArtifact no Redis --> retorna objeto em memoria
  |
  v
Fase 3: Reconciliation (CPU)
  |-- canonical_text = join dos textos PyMuPDF (ja armazenados em self._page_texts)
  |-- Para cada VLM element, busca melhor PyMuPDF block por bbox IoU
  |     IoU >= 0.5 = "exact", IoU >= 0.1 = "partial", else "unmatched"
  |-- Texto reconciliado = SEMPRE do PyMuPDF
  |-- Salva ReconciliationArtifact no Redis --> retorna objeto em memoria
  |
  v
Fase 4: Integrity (CPU)
  |-- 5 checks: content_not_empty, has_articles, min_length, clean_text, vlm_matches
  |-- Score 0.0-1.0
  |-- Salva IntegrityArtifact no Redis --> retorna objeto em memoria
  |
  v
Fase 5: Chunks Preview (CPU)
  |-- Gera ChunkPreview a partir dos VLM devices + texto PyMuPDF
  |-- node_id, span_id, device_type, parent_node_id, canonical_start/end
  |-- Salva ChunksPreviewArtifact no Redis --> DONE
```

### 2.2 Aprovacao

```
Frontend clica "Aprovar"
  |
  v
POST /inspect/approve/{task_id}
  |
  v
ApprovalService:
  1. Valida que inspecao esta COMPLETED
  2. Gera offsets.json a partir do canonical_text
  3. Busca artefatos do Redis (localhost)
  4. Persiste no MinIO (VPS, via MINIO_ENDPOINT remoto)
     -> inspections/{document_id}/metadata.json
     -> inspections/{document_id}/canonical.md
     -> inspections/{document_id}/pymupdf_result.json
     -> inspections/{document_id}/vlm_result.json
     -> inspections/{document_id}/pages/page_001.png ...
  5. Limpa Redis
  6. Retorna ApprovalResult
```

### 2.3 Re-ingestao (futuro)

```
Usuario quer reingerir doc ja aprovado:
  1. Frontend busca do MinIO (VPS): inspections/{document_id}/metadata.json
  2. Compara pdf_hash com PDF atual
  3. Se hash confere -> envia artefatos aprovados pro RunPod
  4. RunPod pula fases 1-5 -> vai direto pra embedding + indexacao
  5. Economiza GPU time (Qwen3-VL nao precisa rodar de novo)
```

---

## 3. Endpoints da API (GPU Server)

Todos sob prefixo `/inspect`:

| Metodo | Path | Descricao |
|--------|------|-----------|
| POST | `/inspect` | Inicia dry-run, retorna task_id |
| GET | `/inspect/status/{task_id}` | Polling progresso (frontend faz a cada 3s) |
| GET | `/inspect/artifacts/{task_id}/{stage}` | Artefato de uma fase (do Redis) |
| GET | `/inspect/metadata/{task_id}` | Metadados da inspecao |
| POST | `/inspect/approve/{task_id}` | Aprova -> Redis -> MinIO |
| GET | `/inspect/health` | Health check (redis + stats) |

Stages validos: `pymupdf`, `vlm`, `reconciliation`, `integrity`, `chunks`

Thresholds de progresso:

| Fase | Threshold |
|------|-----------|
| pymupdf | >= 0.25 |
| vlm | >= 0.55 |
| reconciliation | >= 0.70 |
| integrity | >= 0.80 |
| chunks | >= 0.95 |

---

## 4. Formato dos artefatos por fase

### 4.1 PyMuPDF (stage = "pymupdf")

```json
{
  "blocks": [
    {"text": "Art. 5 ...", "bbox": [72.0, 150.3, 540.0, 165.8], "font_size": 10.0, "is_bold": false, "page": 1}
  ],
  "total_pages": 15,
  "page_images": {"1": "data:image/png;base64,...", "2": "data:image/png;base64,..."}
}
```

### 4.2 VLM (stage = "vlm")

```json
{
  "elements": [
    {"element_type": "article", "bbox": [72.0, 150.3, 540.0, 320.0], "confidence": 0.95, "text": "Art. 5 ...", "parent_id": null, "page": 1}
  ],
  "page_images": {"1": "data:image/png;base64,..."}
}
```

### 4.3 Reconciliation (stage = "reconciliation")

```json
{
  "matches": [
    {"pymupdf_block_id": 0, "vlm_element_id": 0, "match_quality": "exact", "reconciled_text": "Art. 5 ..."}
  ],
  "canonical_text": "# LEI ...\n\nArt. 1 ...",
  "stats": {"total_conflicts": 2, "total_matches": 45, "coverage_pct": 98.5}
}
```

### 4.4 Integrity (stage = "integrity")

```json
{
  "checks": [
    {"check_name": "content_not_empty", "passed": true, "message": "15432 caracteres", "severity": "critical"},
    {"check_name": "vlm_matches", "passed": true, "message": "87% matched", "severity": "warning"}
  ],
  "score": 0.92
}
```

### 4.5 Chunks (stage = "chunks")

```json
{
  "chunks": [
    {"node_id": "leis:LEI-14133-2021#ART-005", "span_id": "ART-005", "device_type": "article", "text": "Art. 5 ...", "canonical_start": 1500, "canonical_end": 2300, "parent_node_id": null}
  ],
  "total_chunks": 47
}
```

---

## 5. Storage

### 5.1 Redis (RunPod localhost:6379, DB 2)

- **Temporario**: TTL 7200s (2h)
- **Key pattern**: `inspect:{task_id}:{suffix}`
- **Suffixes**: metadata, pymupdf, vlm, reconciliation, integrity, chunks

### 5.2 MinIO (VPS 77.37.43.160:9000)

- **Permanente**: apos aprovacao
- **Bucket**: `vectorgov`
- **Path**: `inspections/{document_id}/`
- **Conteudo**: metadata.json, canonical.md, offsets.json, pymupdf_result.json, vlm_result.json, pages/*.png

---

## 6. Variaveis de ambiente necessarias no RunPod

```bash
# Redis (local)
REDIS_URL=redis://localhost:6379        # default ja e localhost

# MinIO (remoto - VPS)
MINIO_ENDPOINT=77.37.43.160:9000       # aponta para VPS
MINIO_ACCESS_KEY=<chave>
MINIO_SECRET_KEY=<chave>
MINIO_BUCKET=vectorgov

# vLLM (local)
VLLM_BASE_URL=http://localhost:8002/v1  # ja configurado
VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct  # ja configurado

# Feature flags
USE_VLM_PIPELINE=true                   # ativa pipeline VLM na ingestao
```

---

## 7. O que falta fazer

### 7.1 Infraestrutura RunPod — FEITO
- [x] **Instalar Redis** no RunPod (apt install redis-server)
- [x] Configurar Redis: bind localhost, port 6379, sem senha
- [x] Testar: `redis-cli ping` -> PONG
- [x] ~~Configurar MINIO_ENDPOINT~~ NAO NECESSARIO — RunPod nao acessa MinIO direto

### 7.2 Approval via HTTP — FEITO
- [x] Criar `src/sinks/inspection_uploader.py` — POST multipart para VPS
- [x] Reescrever `src/inspection/approval.py` — usa InspectionUploader em vez de MinIO direto
- [x] Limpar `src/inspection/storage.py` — removidos metodos de MinIO
- [x] Atualizar `src/sinks/__init__.py` — exports do InspectionUploader
- [x] Escrever instrucoes para VPS: `plans/VPS_INSTRUCTIONS_INSPECT_ARTIFACTS.md`

### 7.3 VPS — PENDENTE (Claude VPS)
- [ ] **Criar endpoint** `POST /api/v1/inspect/artifacts` na VPS
- [ ] Receber multipart (metadata + artefatos JSON + page images PNG)
- [ ] Gravar no MinIO local (127.0.0.1:9100, bucket vectorgov-evidence)
- [ ] Auth via X-Ingest-Key (mesma key do /ingest/artifacts)
- [ ] Ver instrucoes completas em: `plans/VPS_INSTRUCTIONS_INSPECT_ARTIFACTS.md`

### 7.4 Teste end-to-end do Inspector
- [ ] Subir o FastAPI server no RunPod
- [ ] Enviar PDF de teste via curl ou frontend
- [ ] Verificar que todas as 5 fases completam
- [ ] Verificar artefatos no Redis (redis-cli GET inspect:...)
- [ ] Testar aprovacao (POST /inspect/approve) -> HTTP -> VPS -> MinIO

### 7.5 Integracao com pipeline de ingestao
- [ ] Testar USE_VLM_PIPELINE=true com POST /ingest
- [ ] Verificar que chunks gerados passam pr13_validator
- [ ] Verificar indexacao no Milvus

### 7.6 Futuro
- [ ] Pipeline de re-ingestao (VPS busca artefatos aprovados do MinIO e envia pro RunPod)
- [ ] Expandir checks de integridade (invariantes T1-T3, H1-H4, etc.)
- [ ] Otimizacao: pipeline pagina-a-pagina (PyMuPDF page N+1 enquanto VLM processa page N)

---

## 8. Arquivos de referencia

### Inspection (src/inspection/)

| Arquivo | Linhas | Status |
|---------|--------|--------|
| `__init__.py` | ~10 | Pronto |
| `models.py` | 254 | Pronto — todos os modelos Pydantic |
| `storage.py` | ~160 | **LIMPO** — Redis-only, MinIO removido |
| `page_renderer.py` | ~150 | Pronto — PNG com bboxes coloridos |
| `pipeline.py` | ~1074 | **REESCRITO** — todas 5 fases implementadas, sem Docling |
| `approval.py` | ~336 | **REESCRITO** — usa InspectionUploader (HTTP POST), sem MinIO direto |
| `router.py` | 431 | **CORRIGIDO** — task_id + thresholds |

### Extraction (src/extraction/)

| Arquivo | Linhas | Status |
|---------|--------|--------|
| `vlm_models.py` | 53 | Pronto — PageData, DeviceExtraction, etc. |
| `vlm_prompts.py` | 54 | Pronto — prompts para Qwen3-VL |
| `vlm_client.py` | 239 | Pronto — cliente async multimodal |
| `pymupdf_extractor.py` | 103 | Pronto — PNG 300 DPI + texto |
| `vlm_service.py` | 184 | Pronto — orquestrador PyMuPDF -> VLM |

### Sinks (src/sinks/)

| Arquivo | Linhas | Status |
|---------|--------|--------|
| `inspection_uploader.py` | ~284 | **NOVO** — POST multipart para VPS /api/v1/inspect/artifacts |
| `artifacts_uploader.py` | ~314 | Pronto — POST para /api/v1/ingest/artifacts (ingestao) |

### Planejamento (plans/)

| Arquivo | Descricao |
|---------|-----------|
| `CLAUDE_RUNPOD_INSTRUCTIONS.md` | **ESTE DOCUMENTO** — briefing principal |
| `TASK-EXTRACTION-PIPELINE-FLOW.md` | Diagrama completo ANTES/DEPOIS do pipeline |
| `PLAN-VLM-MIGRATION-PHASE2.md` | Plano original da fase 2 (extraction backend) |
| `VPS_INSTRUCTIONS_INSPECT_ARTIFACTS.md` | Spec do endpoint que a VPS deve implementar |
