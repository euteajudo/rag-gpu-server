"""
Router FastAPI para ingestao de PDFs com processamento assincrono.

Endpoints:
    POST /ingest          - Inicia processamento, retorna task_id imediatamente
    GET  /ingest/status/{task_id}  - Status e resultado do processamento

"""

import logging
import uuid
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks

from .models import IngestRequest, IngestResponse, IngestStatus, PhaseResult
from .pipeline import get_pipeline, PipelineResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Storage para tasks em andamento (em producao, usar Redis)
_tasks: Dict[str, Dict[str, Any]] = {}

# ThreadPool para processamento em background
_executor = ThreadPoolExecutor(max_workers=2)


def _process_task(task_id: str, pdf_content: bytes, request: IngestRequest):
    """Processa PDF em background (roda em thread separada)."""
    logger.info(f"[{task_id}] Iniciando processamento em background")

    try:
        _tasks[task_id]["status"] = "processing"
        _tasks[task_id]["current_phase"] = "docling"
        _tasks[task_id]["started_at"] = datetime.utcnow().isoformat()

        pipeline = get_pipeline()
        result: PipelineResult = pipeline.process(pdf_content, request)

        # Converte phases para dict
        phases = []
        for p in result.phases:
            phases.append({
                "phase": p.get("name", ""),
                "duration_ms": p.get("duration_seconds", 0) * 1000,
                "success": p.get("success", True),
                "items_processed": 1,
                "message": p.get("output", ""),
            })

        # Atualiza task com resultado
        _tasks[task_id].update({
            "status": "completed" if result.status == IngestStatus.COMPLETED else "failed",
            "current_phase": None,
            "completed_at": datetime.utcnow().isoformat(),
            "result": {
                "success": result.status == IngestStatus.COMPLETED,
                "document_id": result.document_id,
                "total_chunks": len(result.chunks),
                "phases": phases,
                "errors": result.errors,
                "total_duration_ms": result.total_time_seconds * 1000,
                "chunks": [c.model_dump() for c in result.chunks] if result.chunks else [],
            }
        })

        logger.info(f"[{task_id}] Processamento concluido: {len(result.chunks)} chunks")

    except Exception as e:
        logger.exception(f"[{task_id}] Erro no processamento: {e}")
        _tasks[task_id].update({
            "status": "failed",
            "current_phase": None,
            "completed_at": datetime.utcnow().isoformat(),
            "error": str(e),
        })


@router.post("")
async def ingest_pdf(
    file: UploadFile = File(description="Arquivo PDF para processar"),
    document_id: str = Form(description="ID unico do documento (ex: LEI-14133-2021)"),
    tipo_documento: str = Form(description="Tipo: LEI, DECRETO, IN, etc"),
    numero: str = Form(description="Numero do documento"),
    ano: int = Form(ge=1900, le=2100, description="Ano do documento"),
    skip_embeddings: bool = Form(default=False, description="Pular geracao de embeddings"),
    max_articles: Optional[int] = Form(default=None, description="Limite de artigos (debug)"),
):
    """
    Inicia processamento assincrono de PDF.

    Retorna imediatamente com task_id.
    Use GET /ingest/status/{task_id} para verificar progresso.
    """
    # Valida arquivo
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser PDF")

    # Le conteudo
    pdf_content = await file.read()
    if len(pdf_content) == 0:
        raise HTTPException(status_code=400, detail="Arquivo PDF vazio")

    # Gera task_id
    task_id = str(uuid.uuid4())

    logger.info(f"[{task_id}] Recebido PDF: {file.filename} ({len(pdf_content)} bytes)")
    logger.info(f"[{task_id}] Documento: {document_id} ({tipo_documento} {numero}/{ano})")

    # Cria request
    request = IngestRequest(
        document_id=document_id,
        tipo_documento=tipo_documento,
        numero=numero,
        ano=ano,
        skip_embeddings=skip_embeddings,
        max_articles=max_articles,
    )

    # Registra task
    _tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "document_id": document_id,
        "filename": file.filename,
        "file_size": len(pdf_content),
        "created_at": datetime.utcnow().isoformat(),
        "current_phase": None,
        "result": None,
        "error": None,
    }

    # Inicia processamento em background
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _process_task, task_id, pdf_content, request)

    # Retorna imediatamente
    return {
        "task_id": task_id,
        "status": "pending",
        "message": f"Processamento iniciado para {document_id}",
    }


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Retorna status e resultado do processamento.

    Status possiveis:
    - pending: Aguardando inicio
    - processing: Em andamento
    - completed: Concluido com sucesso
    - failed: Falhou
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task nao encontrada")

    task = _tasks[task_id]

    response = {
        "task_id": task_id,
        "status": task["status"],
        "document_id": task.get("document_id"),
        "filename": task.get("filename"),
        "file_size": task.get("file_size"),
        "current_phase": task.get("current_phase"),
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "completed_at": task.get("completed_at"),
        "error": task.get("error"),
    }

    # Inclui resultado completo se concluido
    if task["status"] == "completed" and task.get("result"):
        response["result"] = task["result"]

    return response


@router.get("/health")
async def ingest_health():
    """Health check do modulo de ingestao."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "docling_loaded": pipeline._docling_converter is not None,
        "span_parser_loaded": pipeline._span_parser is not None,
        "llm_client_loaded": pipeline._llm_client is not None,
        "embedder_loaded": pipeline._embedder is not None,
        "pending_tasks": len([t for t in _tasks.values() if t["status"] == "pending"]),
        "processing_tasks": len([t for t in _tasks.values() if t["status"] == "processing"]),
    }
