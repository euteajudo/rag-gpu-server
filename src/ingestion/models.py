"""
Modelos Pydantic para o pipeline de ingestão.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class IngestStatus(str, Enum):
    """Status do processamento."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestError(BaseModel):
    """Erro durante processamento."""
    phase: str
    message: str
    details: Optional[str] = None


class IngestRequest(BaseModel):
    """Request para processamento de PDF."""

    # Metadados do documento
    document_id: str = Field(description="ID único do documento (ex: LEI-123-2006)")
    tipo_documento: str = Field(description="Tipo: LEI, DECRETO, IN, etc")
    numero: str = Field(description="Número do documento")
    ano: int = Field(ge=1900, le=2100, description="Ano do documento")

    # Configurações opcionais
    skip_enrichment: bool = Field(default=False, description="Pular enriquecimento LLM")
    skip_embeddings: bool = Field(default=False, description="Pular geração de embeddings")
    max_articles: Optional[int] = Field(default=None, description="Limite de artigos (para debug)")


class ProcessedChunk(BaseModel):
    """Chunk processado, pronto para indexação."""

    # Identificação
    chunk_id: str = Field(description="ID único: DOC#SPAN_ID")
    parent_chunk_id: str = Field(default="", description="ID do chunk pai (vazio se artigo)")
    span_id: str = Field(description="ID do span: ART-005, PAR-005-1, etc")
    device_type: str = Field(description="article, paragraph, inciso, alinea")
    chunk_level: str = Field(description="article ou device")

    # Conteúdo
    text: str = Field(description="Texto original do dispositivo")
    enriched_text: str = Field(default="", description="Texto enriquecido para embedding")
    context_header: str = Field(default="", description="Frase de contexto")
    thesis_text: str = Field(default="", description="Resumo/tese do dispositivo")
    thesis_type: str = Field(default="", description="Tipo: definicao, procedimento, etc")
    synthetic_questions: str = Field(default="", description="Perguntas relacionadas")

    # Metadados do documento
    document_id: str
    tipo_documento: str
    numero: str
    ano: int
    article_number: str = Field(default="", description="Número do artigo")

    # Vetores (opcionais, preenchidos se skip_embeddings=False)
    dense_vector: Optional[list[float]] = Field(default=None, description="Embedding dense 1024d")
    sparse_vector: Optional[dict[int, float]] = Field(default=None, description="Embedding sparse")
    thesis_vector: Optional[list[float]] = Field(default=None, description="Embedding da thesis")

    # Proveniência
    citations: list[str] = Field(default_factory=list, description="Spans citados")
    schema_version: str = Field(default="1.0.0")


class PhaseResult(BaseModel):
    """Resultado de uma fase do pipeline."""
    phase: str
    duration_ms: float
    success: bool
    items_processed: int = 0
    message: str = ""


class IngestResponse(BaseModel):
    """Response do processamento completo."""

    success: bool
    document_id: str
    status: IngestStatus

    # Resultados
    chunks: list[ProcessedChunk] = Field(default_factory=list)
    total_chunks: int = 0

    # Métricas por fase
    phases: list[PhaseResult] = Field(default_factory=list)
    total_duration_ms: float = 0

    # Erros (se houver)
    errors: list[IngestError] = Field(default_factory=list)

    # Estatísticas
    articles_extracted: int = 0
    paragraphs_extracted: int = 0
    incisos_extracted: int = 0
