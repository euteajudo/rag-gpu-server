"""
Router FastAPI para o Pipeline Inspector.

Versao async com background processing (mesma arquitetura do ingestion/router.py).

Endpoints:
    POST /inspect                          - Inicia dry-run, retorna task_id
    GET  /inspect/status/{task_id}         - Polling de progresso por fase
    GET  /inspect/artifacts/{task_id}/{stage} - Artefatos de uma fase
    POST /inspect/approve/{task_id}        - Aprova e persiste no MinIO
    GET  /inspect/health                   - Health check do modulo
"""

import hashlib
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .approval import ApprovalService
from .models import (
    ApprovalResult,
    InspectionStage,
    InspectionStatus,
)
from .pipeline import InspectionPipeline
from .storage import InspectionStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inspect", tags=["Inspection"])


# ============================================================================
# TASK STORAGE (in-memory — mesmo padrão do ingestion/router.py)
# ============================================================================


class InspectTaskInfo(BaseModel):
    """Informações de uma task de inspeção."""

    task_id: str
    document_id: str
    status: str = "processing"  # processing, completed, failed
    progress: float = 0.0
    current_phase: str = "starting"
    stages_completed: list[str] = Field(default_factory=list)
    started_at: str = ""
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


_tasks: Dict[str, InspectTaskInfo] = {}
_tasks_lock = threading.Lock()


def _generate_task_id(document_id: str, pdf_content: bytes) -> str:
    """Gera um ID unico para a task de inspeção."""
    hash_input = f"inspect-{document_id}-{len(pdf_content)}-{time.time()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def _get_task(task_id: str) -> Optional[InspectTaskInfo]:
    with _tasks_lock:
        return _tasks.get(task_id)


def _update_task(task_id: str, **kwargs) -> None:
    with _tasks_lock:
        if task_id in _tasks:
            task = _tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)


def _add_completed_stage(task_id: str, stage: str) -> None:
    with _tasks_lock:
        if task_id in _tasks:
            task = _tasks[task_id]
            if stage not in task.stages_completed:
                task.stages_completed.append(stage)


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class InspectStartResponse(BaseModel):
    """Resposta do POST /inspect."""

    task_id: str
    document_id: str
    message: str = "Inspeção iniciada em background"


class InspectStatusResponse(BaseModel):
    """Resposta do GET /inspect/status/{task_id}."""

    task_id: str
    document_id: str
    status: str
    progress: float
    current_phase: str
    stages_completed: list[str]
    started_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class ApproveRequest(BaseModel):
    """Body do POST /inspect/approve/{task_id}."""

    approved_by: str = Field("admin", description="Nome/email de quem aprovou")


# ============================================================================
# SHARED SERVICES (lazy init)
# ============================================================================

_storage: Optional[InspectionStorage] = None
_pipeline: Optional[InspectionPipeline] = None
_approval: Optional[ApprovalService] = None


def _get_storage() -> InspectionStorage:
    global _storage
    if _storage is None:
        _storage = InspectionStorage()
    return _storage


def _get_pipeline() -> InspectionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InspectionPipeline(storage=_get_storage())
    return _pipeline


def _get_approval() -> ApprovalService:
    global _approval
    if _approval is None:
        _approval = ApprovalService(storage=_get_storage())
    return _approval


# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================


def _background_inspect(
    task_id: str,
    pdf_content: bytes,
    document_id: str,
    tipo_documento: str,
    numero: str,
    ano: int,
) -> None:
    """
    Executa o dry-run em background (roda em thread separada).
    Atualiza status da task conforme progride.
    """
    try:
        logger.info(f"[Inspect {task_id}] Iniciando inspeção de {document_id}")
        _update_task(task_id, current_phase="initializing", progress=0.05)

        pipeline = _get_pipeline()

        # Callback para atualizar progresso
        def progress_callback(phase: str, progress: float) -> None:
            _update_task(task_id, current_phase=phase, progress=progress)

            # Detecta fase completada pelo progresso
            stage_thresholds = {
                "pymupdf": 0.30,
                "vlm": 0.40,
                "reconciliation": 0.60,
                "integrity": 0.75,
                "chunks": 0.95,
            }
            for stage_name, threshold in stage_thresholds.items():
                if progress >= threshold:
                    _add_completed_stage(task_id, stage_name)

            logger.info(f"[Inspect {task_id}] {phase}: {progress * 100:.1f}%")

        # Executa dry-run
        pipeline.run(
            pdf_bytes=pdf_content,
            document_id=document_id,
            tipo_documento=tipo_documento,
            numero=numero,
            ano=ano,
            progress_callback=progress_callback,
        )

        # Sucesso
        _update_task(
            task_id,
            status="completed",
            progress=1.0,
            current_phase="completed",
            completed_at=datetime.now().isoformat(),
        )
        logger.info(f"[Inspect {task_id}] Inspeção concluída com sucesso")

    except Exception as e:
        logger.exception(f"[Inspect {task_id}] Erro na inspeção: {e}")
        _update_task(
            task_id,
            status="failed",
            error_message=str(e),
            completed_at=datetime.now().isoformat(),
        )


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("", response_model=InspectStartResponse)
async def start_inspection(
    file: UploadFile = File(..., description="Arquivo PDF para inspecionar"),
    document_id: str = Form(..., description="ID do documento (ex: LEI-14133-2021)"),
    tipo_documento: str = Form(..., description="Tipo: LEI, DECRETO, IN, etc"),
    numero: str = Form(..., description="Número do documento"),
    ano: int = Form(..., ge=1900, le=2100, description="Ano do documento"),
):
    """
    Inicia inspeção (dry-run) de um PDF em background.

    Retorna imediatamente um task_id para polling via GET /inspect/status/{task_id}.

    O dry-run processa o PDF por todas as fases (PyMuPDF, VLM, Reconciliação,
    Integridade, Chunks) mas NÃO indexa no Milvus. Os artefatos ficam
    temporariamente no Redis (TTL 2h) até aprovação.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser PDF")

    pdf_content = await file.read()
    if len(pdf_content) == 0:
        raise HTTPException(status_code=400, detail="Arquivo PDF vazio")

    logger.info(
        f"Inspeção recebida: {file.filename} ({len(pdf_content)} bytes) "
        f"doc={document_id}"
    )

    task_id = _generate_task_id(document_id, pdf_content)

    with _tasks_lock:
        _tasks[task_id] = InspectTaskInfo(
            task_id=task_id,
            document_id=document_id,
            status="processing",
            progress=0.0,
            current_phase="queued",
            started_at=datetime.now().isoformat(),
        )

    thread = threading.Thread(
        target=_background_inspect,
        args=(task_id, pdf_content, document_id, tipo_documento, numero, ano),
        daemon=True,
    )
    thread.start()

    return InspectStartResponse(
        task_id=task_id,
        document_id=document_id,
        message=(
            "Inspeção iniciada em background. "
            "Use GET /inspect/status/{task_id} para acompanhar."
        ),
    )


@router.get("/status/{task_id}", response_model=InspectStatusResponse)
async def get_inspection_status(task_id: str):
    """
    Verifica o status de uma inspeção.

    Use para polling até status ser 'completed' ou 'failed'.
    Intervalo recomendado: 3 segundos.
    """
    task = _get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404, detail=f"Inspeção {task_id} não encontrada"
        )

    return InspectStatusResponse(
        task_id=task.task_id,
        document_id=task.document_id,
        status=task.status,
        progress=task.progress,
        current_phase=task.current_phase,
        stages_completed=task.stages_completed,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error_message=task.error_message,
    )


@router.get("/artifacts/{task_id}/{stage}")
async def get_inspection_artifacts(task_id: str, stage: str):
    """
    Retorna os artefatos de uma fase da inspeção.

    Stages válidos: pymupdf, vlm, reconciliation, integrity, chunks

    Os artefatos ficam no Redis com TTL de 2h. Após aprovação,
    são persistidos no MinIO.
    """
    # Valida stage
    try:
        inspection_stage = InspectionStage(stage)
    except ValueError:
        valid = [s.value for s in InspectionStage]
        raise HTTPException(
            status_code=400,
            detail=f"Stage '{stage}' inválido. Válidos: {valid}",
        )

    # Verifica que a task existe
    task = _get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=404, detail=f"Inspeção {task_id} não encontrada"
        )

    # Busca artefato no Redis
    storage = _get_storage()
    artifact_json = storage.get_artifact(task_id, inspection_stage)

    if artifact_json is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Artefato '{stage}' não encontrado para inspeção {task_id}. "
                "A fase pode não ter sido executada ainda ou o TTL expirou (2h)."
            ),
        )

    # Retorna o JSON parseado
    import json

    try:
        return json.loads(artifact_json)
    except json.JSONDecodeError:
        return {"raw": artifact_json}


@router.get("/metadata/{task_id}")
async def get_inspection_metadata(task_id: str):
    """
    Retorna os metadados completos de uma inspeção.

    Inclui: document_id, tipo_documento, status, pdf_hash, timestamps, etc.
    """
    storage = _get_storage()
    metadata = storage.get_metadata(task_id)

    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Metadados de inspeção {task_id} não encontrados "
                "(podem ter expirado após 2h)."
            ),
        )

    return metadata.model_dump()


@router.post("/approve/{task_id}", response_model=ApprovalResult)
async def approve_inspection(task_id: str, body: Optional[ApproveRequest] = None):
    """
    Aprova uma inspeção, persistindo artefatos no MinIO.

    Fluxo:
    1. Valida que a inspeção está COMPLETED
    2. Gera offsets.json a partir do canonical text
    3. Persiste todos os artefatos no MinIO (permanente)
    4. Limpa artefatos temporários do Redis
    5. Retorna ApprovalResult com lista de artefatos persistidos

    Após aprovação, o pipeline de ingestão pode reutilizar os artefatos
    aprovados sem reprocessar o PDF.
    """
    approved_by = body.approved_by if body else "admin"

    approval = _get_approval()
    result = approval.approve(task_id=task_id, approved_by=approved_by)

    if not result.success:
        raise HTTPException(status_code=400, detail=result.message)

    return result


@router.get("/health")
async def inspect_health():
    """Health check do módulo de inspeção."""
    storage = _get_storage()

    # Testa conexão Redis
    redis_ok = False
    try:
        storage._get_redis().ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
        "active_inspections": len(
            [t for t in _tasks.values() if t.status == "processing"]
        ),
        "total_inspections": len(_tasks),
    }
