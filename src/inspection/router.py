"""
Router FastAPI para o Pipeline Inspector v4 (Regex).

Endpoints:
    GET  /inspect/inspector                - Frontend SPA
    GET  /inspect/inspector/recent         - Lista inspeções recentes
    GET  /inspect/inspector/{task_id}      - Detalhe de uma inspeção
    GET  /inspect/health                   - Health check do módulo
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from .models import InspectionStage
from .storage import InspectionStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inspect", tags=["Inspection"])

# Lazy init storage
_storage = None


def _get_storage() -> InspectionStorage:
    global _storage
    if _storage is None:
        _storage = InspectionStorage()
    return _storage


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get("/inspector", response_class=HTMLResponse)
async def inspector_ui():
    """Serve o frontend do Pipeline Inspector."""
    html_path = Path(__file__).parent / "static" / "inspector.html"
    if not html_path.exists():
        raise HTTPException(404, "Inspector frontend not found")
    return html_path.read_text(encoding="utf-8")


@router.get("/inspector/recent")
async def list_recent_inspections():
    """Lista inspeções regex recentes (últimas 2h, TTL do Redis)."""
    storage = _get_storage()
    try:
        r = storage._get_redis()
    except Exception:
        return []

    # Scan por chaves inspect:*:regex_classification
    keys = list(r.scan_iter(match="inspect:*:regex_classification"))

    results = []
    for key in keys:
        key_str = key.decode() if isinstance(key, bytes) else key
        parts = key_str.split(":")
        if len(parts) >= 2:
            task_id = parts[1]
            metadata = storage.get_metadata(task_id)
            if metadata:
                results.append({
                    "task_id": task_id,
                    "document_id": metadata.document_id,
                    "tipo_documento": metadata.tipo_documento,
                    "status": metadata.status.value,
                    "started_at": metadata.started_at,
                    "total_pages": metadata.total_pages,
                })

    results.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return results


@router.get("/inspector/{task_id}")
async def get_inspection_detail(task_id: str):
    """Retorna artefatos completos de uma inspeção regex."""
    storage = _get_storage()

    pymupdf_raw = storage.get_artifact(task_id, InspectionStage.PYMUPDF)
    regex_raw = storage.get_artifact(task_id, InspectionStage.REGEX_CLASSIFICATION)

    if not regex_raw:
        raise HTTPException(
            status_code=404,
            detail=f"Inspeção {task_id} não encontrada ou expirada (TTL 2h)",
        )

    metadata = storage.get_metadata(task_id)

    return {
        "task_id": task_id,
        "metadata": metadata.model_dump() if metadata else None,
        "pymupdf": json.loads(pymupdf_raw) if pymupdf_raw else None,
        "regex_classification": json.loads(regex_raw),
    }


@router.get("/health")
async def inspect_health():
    """Health check do módulo de inspeção."""
    storage = _get_storage()

    redis_ok = False
    try:
        storage._get_redis().ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
    }
