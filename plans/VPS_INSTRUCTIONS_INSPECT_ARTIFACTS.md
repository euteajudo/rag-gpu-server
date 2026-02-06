# Instrucoes para VPS: Endpoint POST /api/v1/inspect/artifacts

> **Data**: 2026-02-06
> **De**: Claude Code (RunPod)
> **Para**: Claude Code (VPS)
> **Contexto**: O RunPod NAO acessa MinIO diretamente. Quando o usuario aprova uma inspecao, o RunPod envia os artefatos via HTTP POST para a VPS, e a VPS grava no MinIO local.

---

## 1. O que precisa ser criado

Um novo endpoint na VPS: `POST /api/v1/inspect/artifacts`

Este endpoint recebe artefatos de inspecao aprovada do RunPod e os persiste no MinIO local.

**Onde criar**: `extracao/src/api/routers/inspect.py` (ja existe este arquivo como proxy). Adicionar a rota neste arquivo, ou criar um arquivo separado se preferir.

**Registrar**: O router ja deve estar registrado no app. Confirmar que o prefix e `/api/v1/inspect`.

---

## 2. Assinatura do endpoint

```python
@router.post("/artifacts")
async def upload_inspection_artifacts(
    # Form data
    document_id: str = Form(...),

    # Arquivos (multipart)
    metadata_file: UploadFile = File(..., description="metadata.json"),
    canonical_md_file: Optional[UploadFile] = File(None, description="canonical.md"),
    offsets_json_file: Optional[UploadFile] = File(None, description="offsets.json"),
    pymupdf_file: Optional[UploadFile] = File(None, description="pymupdf_result.json"),
    vlm_file: Optional[UploadFile] = File(None, description="vlm_result.json"),
    reconciliation_file: Optional[UploadFile] = File(None, description="reconciliation_result.json"),
    integrity_file: Optional[UploadFile] = File(None, description="integrity_result.json"),
    chunks_file: Optional[UploadFile] = File(None, description="chunks_preview.json"),
    page_images: Optional[list[UploadFile]] = File(None, description="PNGs das paginas anotadas"),

    # Auth
    x_ingest_key: Optional[str] = Header(None, alias="X-Ingest-Key"),
):
```

---

## 3. O que o endpoint faz

### 3.1 Autenticacao

Mesma logica do `/api/v1/ingest/artifacts`:
```python
expected_key = os.getenv("INGEST_API_KEY", "")
if expected_key and x_ingest_key != expected_key:
    raise HTTPException(status_code=403, detail="Invalid X-Ingest-Key")
```

### 3.2 Gravar no MinIO

Usar o `StorageService` existente (`extracao/src/evidence/storage_service.py`) ou o Minio client diretamente.

**Bucket**: `vectorgov-evidence` (mesmo bucket existente)
**Path prefix**: `inspections/{document_id}/`

Mapa de arquivos:

| Campo multipart | Filename recebido | Key no MinIO |
|-----------------|-------------------|-------------|
| `metadata_file` | metadata.json | `inspections/{document_id}/metadata.json` |
| `canonical_md_file` | canonical.md | `inspections/{document_id}/canonical.md` |
| `offsets_json_file` | offsets.json | `inspections/{document_id}/offsets.json` |
| `pymupdf_file` | pymupdf_result.json | `inspections/{document_id}/pymupdf_result.json` |
| `vlm_file` | vlm_result.json | `inspections/{document_id}/vlm_result.json` |
| `reconciliation_file` | reconciliation_result.json | `inspections/{document_id}/reconciliation_result.json` |
| `integrity_file` | integrity_result.json | `inspections/{document_id}/integrity_result.json` |
| `chunks_file` | chunks_preview.json | `inspections/{document_id}/chunks_preview.json` |
| `page_images[i]` | page_001.png, page_002.png, ... | `inspections/{document_id}/pages/page_001.png`, ... |

Content types:
- JSON: `application/json; charset=utf-8`
- Markdown: `text/markdown; charset=utf-8`
- PNG: `image/png`

### 3.3 Exemplo de implementacao

```python
from minio import Minio
from io import BytesIO

minio_client = Minio("127.0.0.1:9100", access_key=..., secret_key=..., secure=False)
bucket = "vectorgov-evidence"
base = f"inspections/{document_id}"

# Metadata (obrigatorio)
metadata_bytes = await metadata_file.read()
minio_client.put_object(
    bucket, f"{base}/metadata.json",
    data=BytesIO(metadata_bytes), length=len(metadata_bytes),
    content_type="application/json; charset=utf-8",
)
persisted = [f"{base}/metadata.json"]

# Arquivos opcionais
optional_files = [
    (canonical_md_file, "canonical.md", "text/markdown; charset=utf-8"),
    (offsets_json_file, "offsets.json", "application/json; charset=utf-8"),
    (pymupdf_file, "pymupdf_result.json", "application/json; charset=utf-8"),
    (vlm_file, "vlm_result.json", "application/json; charset=utf-8"),
    (reconciliation_file, "reconciliation_result.json", "application/json; charset=utf-8"),
    (integrity_file, "integrity_result.json", "application/json; charset=utf-8"),
    (chunks_file, "chunks_preview.json", "application/json; charset=utf-8"),
]

for upload_file, filename, content_type in optional_files:
    if upload_file is not None:
        data = await upload_file.read()
        if data:
            key = f"{base}/{filename}"
            minio_client.put_object(
                bucket, key,
                data=BytesIO(data), length=len(data),
                content_type=content_type,
            )
            persisted.append(key)

# Imagens de paginas
if page_images:
    for img_file in page_images:
        if img_file and img_file.filename:
            data = await img_file.read()
            if data:
                key = f"{base}/pages/{img_file.filename}"
                minio_client.put_object(
                    bucket, key,
                    data=BytesIO(data), length=len(data),
                    content_type="image/png",
                )
                persisted.append(key)
```

### 3.4 Resposta

```python
return {
    "success": True,
    "document_id": document_id,
    "artifacts_persisted": persisted,  # lista de keys no MinIO
    "message": f"{len(persisted)} artefatos persistidos",
}
```

Status codes:
- `200`: Sucesso
- `400`: document_id ausente ou metadata_file ausente
- `403`: X-Ingest-Key invalido
- `500`: Erro interno (MinIO down, etc.)

---

## 4. Diagrama do fluxo

```
RunPod (GPU Server)                    VPS (Hostinger)
┌──────────────────────┐              ┌────────────────────────────────┐
│ approval.py          │              │ inspect.py (router)            │
│   approve()          │   POST       │   upload_inspection_artifacts()│
│     ↓                │  /api/v1/    │     ↓                         │
│ InspectionUploader   │  inspect/    │   StorageService / Minio      │
│   .upload()          │──artifacts──→│     ↓                         │
│                      │  multipart:  │   MinIO (127.0.0.1:9100)      │
│  metadata.json       │  + metadata  │   Bucket: vectorgov-evidence  │
│  canonical.md        │  + canonical │     ├── inspections/{id}/     │
│  offsets.json        │  + offsets   │     │   ├── metadata.json     │
│  pymupdf_result.json │  + pymupdf  │     │   ├── canonical.md      │
│  vlm_result.json     │  + vlm      │     │   ├── offsets.json      │
│  reconciliation.json │  + recon    │     │   ├── pymupdf_result.json│
│  integrity.json      │  + integrity│     │   ├── vlm_result.json   │
│  chunks_preview.json │  + chunks   │     │   ├── pages/            │
│  page_*.png          │  + images   │     │   │   ├── page_001.png  │
│                      │  + X-Ingest │     │   │   └── page_002.png  │
│                      │    -Key     │     │   └── ...               │
└──────────────────────┘              └────────────────────────────────┘
```

---

## 5. Notas importantes

1. **O RunPod ja implementou o lado enviador**:
   - `src/sinks/inspection_uploader.py` — InspectionUploader (POST multipart)
   - `src/inspection/approval.py` — reescrito para usar InspectionUploader
   - `src/inspection/storage.py` — removidos metodos de MinIO direto

2. **Env vars esperadas na VPS**:
   - `INGEST_API_KEY` — mesma key ja usada pelo `/api/v1/ingest/artifacts`

3. **Tamanho dos payloads**: artefatos podem ser grandes (imagens base64 em JSONs + PNGs). Um documento de 15 paginas pode gerar ~50-100MB de artefatos. Configurar timeouts e limites de upload adequados.

4. **Idempotencia**: Se o mesmo `document_id` for enviado novamente (re-aprovacao), os arquivos no MinIO devem ser sobrescritos (put_object faz isso por padrao).

5. **O proxy existente** (`inspect.py` na VPS) que repassa requests para o RunPod continua funcionando normalmente. Este novo endpoint `/artifacts` e ADICIONAL — nao substitui nada existente.

6. **Bucket pode ser diferente**: Se o bucket `vectorgov-evidence` nao for adequado para inspecoes, pode usar `vectorgov` ou criar um sub-path. O importante e que o frontend consiga acessar os artefatos aprovados depois.
