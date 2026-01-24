"""
Router FastAPI para ingestao de PDFs.

Versao async com background processing para evitar timeout do Cloudflare.

Endpoints:
    POST /ingest  - Inicia processamento e retorna task_id imediatamente
    GET  /ingest/status/{task_id}  - Verifica status do processamento
    GET  /ingest/health  - Health check do modulo
"""

import asyncio
import logging
import hashlib
import threading
import time
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from .models import IngestRequest, ProcessedChunk, IngestStatus, IngestError
from .pipeline import get_pipeline, PipelineResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


# ============================================================================
# TASK STORAGE (in-memory)
# ============================================================================

class TaskInfo(BaseModel):
    """Informacoes de uma task de ingestao."""
    task_id: str
    document_id: str
    status: str = "processing"  # processing, completed, failed
    progress: float = 0.0
    current_phase: str = "starting"
    started_at: str = ""
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[dict] = None


# Armazenamento global de tasks (em producao, usar Redis)
_tasks: Dict[str, TaskInfo] = {}
_tasks_lock = threading.Lock()


def _generate_task_id(document_id: str, pdf_content: bytes) -> str:
    """Gera um ID unico para a task."""
    hash_input = f"{document_id}-{len(pdf_content)}-{time.time()}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def _get_task(task_id: str) -> Optional[TaskInfo]:
    """Busca uma task pelo ID."""
    with _tasks_lock:
        return _tasks.get(task_id)


def _update_task(task_id: str, **kwargs):
    """Atualiza campos de uma task."""
    with _tasks_lock:
        if task_id in _tasks:
            task = _tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)


def _set_task_result(task_id: str, result: PipelineResult):
    """Salva o resultado completo de uma task."""
    with _tasks_lock:
        if task_id in _tasks:
            task = _tasks[task_id]
            task.status = "completed" if result.status == IngestStatus.COMPLETED else "failed"
            task.progress = 1.0
            task.completed_at = datetime.now().isoformat()
            # CORREÇÃO: Define error_message se houver erros (para VPS poder ler)
            if result.errors:
                task.error_message = "; ".join([e.message for e in result.errors])
            task.result = {
                "success": result.status == IngestStatus.COMPLETED,
                "document_id": result.document_id,
                "status": result.status.value,
                "total_chunks": len(result.chunks),
                "phases": result.phases,
                "errors": [{"phase": e.phase, "message": e.message, "details": e.details} for e in result.errors],
                "total_time_seconds": result.total_time_seconds,
                "chunks": [c.model_dump() for c in result.chunks],
                "document_hash": result.document_hash,
            }


def _set_task_error(task_id: str, error_message: str):
    """Marca uma task como falha."""
    with _tasks_lock:
        if task_id in _tasks:
            task = _tasks[task_id]
            task.status = "failed"
            task.error_message = error_message
            task.completed_at = datetime.now().isoformat()


class IngestStartResponse(BaseModel):
    """Resposta do endpoint de inicio de ingestao (async)."""
    task_id: str
    document_id: str
    message: str = "Processamento iniciado em background"


class IngestStatusResponse(BaseModel):
    """Resposta do endpoint de status."""
    task_id: str
    document_id: str
    status: str  # processing, completed, failed
    progress: float
    current_phase: str
    started_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class IngestResponse(BaseModel):
    """Resposta completa com chunks (quando task completa)."""
    success: bool
    document_id: str
    status: str
    total_chunks: int = 0
    phases: List[dict] = []
    errors: List[dict] = []
    total_time_seconds: float = 0.0
    chunks: List[dict] = []
    document_hash: str = ""


def _background_process(task_id: str, pdf_content: bytes, request: IngestRequest):
    """
    Processa o PDF em background (roda em thread separada).
    Atualiza o status da task conforme progride.
    """
    try:
        logger.info(f"[Task {task_id}] Iniciando processamento de {request.document_id}")
        _update_task(task_id, current_phase="initializing", progress=0.05)

        pipeline = get_pipeline()

        # Callback para atualizar progresso
        def progress_callback(phase: str, progress: float):
            _update_task(task_id, current_phase=phase, progress=progress)
            logger.info(f"[Task {task_id}] {phase}: {progress*100:.1f}%")

        # Processa
        _update_task(task_id, current_phase="processing", progress=0.1)
        result = pipeline.process(pdf_content, request, progress_callback=progress_callback)

        # Salva resultado
        _set_task_result(task_id, result)
        logger.info(f"[Task {task_id}] Concluido: {len(result.chunks)} chunks")

    except Exception as e:
        logger.exception(f"[Task {task_id}] Erro no processamento: {e}")
        _set_task_error(task_id, str(e))


@router.post("", response_model=IngestStartResponse)
async def ingest_pdf(
    file: UploadFile = File(..., description="Arquivo PDF para processar"),
    document_id: str = Form(..., description="ID unico do documento (ex: LEI-14133-2021)"),
    tipo_documento: str = Form(..., description="Tipo: LEI, DECRETO, IN, ACORDAO, etc"),
    numero: str = Form(..., description="Numero do documento"),
    ano: int = Form(..., ge=1900, le=2100, description="Ano do documento"),
    titulo: Optional[str] = Form(None, description="Titulo do documento (opcional)"),
    # Campos especificos para Acordaos TCU
    colegiado: Optional[str] = Form(None, description="Colegiado: P (Plenario), 1C, 2C"),
    processo: Optional[str] = Form(None, description="Numero do processo (TC xxx.xxx/xxxx-x)"),
    relator: Optional[str] = Form(None, description="Nome do Ministro Relator"),
    data_sessao: Optional[str] = Form(None, description="Data da sessao (DD/MM/YYYY)"),
    unidade_tecnica: Optional[str] = Form(None, description="Unidade tecnica responsavel"),
    unidade_jurisdicionada: Optional[str] = Form(None, description="Orgao/Entidade objeto da deliberacao"),
    # Configuracoes opcionais
    skip_embeddings: bool = Form(False, description="Pular geracao de embeddings"),
    max_articles: Optional[int] = Form(None, description="Limite de artigos (debug)"),
):
    """
    Inicia processamento de um PDF em background.

    Retorna imediatamente um task_id para polling via GET /ingest/status/{task_id}.

    Isso evita timeout do Cloudflare (~100s) para documentos grandes.
    O processamento completo pode levar varios minutos.
    """
    # Valida arquivo
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser PDF")

    # Le conteudo (I/O async do FastAPI)
    pdf_content = await file.read()
    if len(pdf_content) == 0:
        raise HTTPException(status_code=400, detail="Arquivo PDF vazio")

    logger.info(f"Recebido PDF: {file.filename} ({len(pdf_content)} bytes)")
    logger.info(f"Documento: {document_id} ({tipo_documento} {numero}/{ano})")

    # Cria request
    request = IngestRequest(
        document_id=document_id,
        tipo_documento=tipo_documento,
        numero=numero,
        ano=ano,
        titulo=titulo,
        # Campos especificos para Acordaos TCU
        colegiado=colegiado,
        processo=processo,
        relator=relator,
        data_sessao=data_sessao,
        unidade_tecnica=unidade_tecnica,
        unidade_jurisdicionada=unidade_jurisdicionada,
        # Configuracoes opcionais
        skip_embeddings=skip_embeddings,
        max_articles=max_articles,
    )

    # Gera task_id
    task_id = _generate_task_id(document_id, pdf_content)

    # Registra task
    with _tasks_lock:
        _tasks[task_id] = TaskInfo(
            task_id=task_id,
            document_id=document_id,
            status="processing",
            progress=0.0,
            current_phase="queued",
            started_at=datetime.now().isoformat(),
        )

    # Inicia processamento em thread separada
    thread = threading.Thread(
        target=_background_process,
        args=(task_id, pdf_content, request),
        daemon=True,
    )
    thread.start()

    logger.info(f"Task {task_id} iniciada para {document_id}")

    return IngestStartResponse(
        task_id=task_id,
        document_id=document_id,
        message="Processamento iniciado em background. Use GET /ingest/status/{task_id} para acompanhar.",
    )


@router.get("/status/{task_id}", response_model=IngestStatusResponse)
async def get_ingest_status(task_id: str):
    """
    Verifica o status de uma task de ingestao.

    Use este endpoint para polling ate status ser 'completed' ou 'failed'.
    """
    task = _get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} nao encontrada")

    return IngestStatusResponse(
        task_id=task.task_id,
        document_id=task.document_id,
        status=task.status,
        progress=task.progress,
        current_phase=task.current_phase,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error_message=task.error_message,
    )


@router.get("/result/{task_id}", response_model=IngestResponse)
async def get_ingest_result(task_id: str):
    """
    Retorna o resultado completo de uma task de ingestao.

    So disponivel quando status == 'completed'.
    """
    task = _get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} nao encontrada")

    if task.status == "processing":
        raise HTTPException(status_code=202, detail="Processamento ainda em andamento")

    if task.status == "failed":
        raise HTTPException(status_code=500, detail=f"Processamento falhou: {task.error_message}")

    if not task.result:
        raise HTTPException(status_code=500, detail="Resultado nao disponivel")

    return IngestResponse(**task.result)


@router.get("/health")
async def ingest_health():
    """
    Health check do modulo de ingestao.

    CORRIGIDO: Usa asyncio.to_thread() para verificacoes que podem bloquear.
    """

    def _do_health_check() -> dict:
        """Executa health check (sync - roda em thread)."""
        pipeline = get_pipeline()
        return {
            "status": "healthy",
            "docling_loaded": pipeline._docling_converter is not None,
            "span_parser_loaded": pipeline._span_parser is not None,
            "llm_client_loaded": pipeline._llm_client is not None,
            "embedder_loaded": pipeline._embedder is not None,
        }

    return await asyncio.to_thread(_do_health_check)
