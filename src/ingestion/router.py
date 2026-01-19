"""
Router FastAPI para ingestao de PDFs.

Versao async-safe: usa asyncio.to_thread() para pipeline blocking.

Endpoints:
    POST /ingest  - Processa PDF e retorna chunks prontos para indexacao
    GET  /ingest/health  - Health check do modulo
"""

import asyncio
import logging
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field

from .models import IngestRequest, ProcessedChunk, IngestStatus, IngestError
from .pipeline import get_pipeline, PipelineResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestResponse(BaseModel):
    """Resposta do endpoint de ingestao."""
    success: bool
    document_id: str
    status: IngestStatus
    total_chunks: int = 0
    phases: List[dict] = []
    errors: List[IngestError] = []
    total_time_seconds: float = 0.0
    chunks: List[ProcessedChunk] = []
    document_hash: str = ""


@router.post("", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(..., description="Arquivo PDF para processar"),
    document_id: str = Form(..., description="ID unico do documento (ex: LEI-14133-2021)"),
    tipo_documento: str = Form(..., description="Tipo: LEI, DECRETO, IN, etc"),
    numero: str = Form(..., description="Numero do documento"),
    ano: int = Form(..., ge=1900, le=2100, description="Ano do documento"),
    skip_embeddings: bool = Form(False, description="Pular geracao de embeddings"),
    max_articles: Optional[int] = Form(None, description="Limite de artigos (debug)"),
):
    """
    Processa um PDF e retorna chunks prontos para indexacao.

    O PDF e processado pelo pipeline completo:
    1. Docling (PDF -> Markdown)
    2. SpanParser (Markdown -> Spans)
    3. ArticleOrchestrator (LLM extraction)
    4. ChunkMaterializer (parent-child chunks)
    5. Embeddings (BGE-M3) - opcional

    CORRIGIDO: Usa asyncio.to_thread() para nao bloquear o event loop.
    O pipeline pode levar varios minutos para documentos grandes.
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
        skip_embeddings=skip_embeddings,
        max_articles=max_articles,
    )

    def _do_process(pdf_bytes: bytes, ingest_req: IngestRequest) -> PipelineResult:
        """Executa pipeline (sync - roda em thread)."""
        pipeline = get_pipeline()
        return pipeline.process(pdf_bytes, ingest_req)

    # Executa em thread para nao bloquear event loop
    # IMPORTANTE: pipeline.process() pode levar minutos!
    result: PipelineResult = await asyncio.to_thread(_do_process, pdf_content, request)

    # Monta resposta
    return IngestResponse(
        success=result.status == IngestStatus.COMPLETED,
        document_id=result.document_id,
        status=result.status,
        total_chunks=len(result.chunks),
        phases=result.phases,
        errors=result.errors,
        total_time_seconds=result.total_time_seconds,
        chunks=result.chunks,
        document_hash=result.document_hash,
    )


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
