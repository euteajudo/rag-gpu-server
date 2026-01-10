"""
Router FastAPI para ingestao de PDFs.

Endpoints:
    POST /ingest  - Processa PDF e retorna chunks prontos para indexacao
    GET  /ingest/status/{task_id}  - Status do processamento (futuro)
"""

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

    A VPS recebe os chunks processados e faz a indexacao no Milvus.
    """
    # Valida arquivo
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser PDF")

    # Le conteudo
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

    # Processa
    pipeline = get_pipeline()
    result: PipelineResult = pipeline.process(pdf_content, request)

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
    """Health check do modulo de ingestao."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "docling_loaded": pipeline._docling_converter is not None,
        "span_parser_loaded": pipeline._span_parser is not None,
        "llm_client_loaded": pipeline._llm_client is not None,
        "embedder_loaded": pipeline._embedder is not None,
    }
