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
    document_id: str = Field(..., description="ID único do documento (ex: LEI-123-2006)")
    tipo_documento: str = Field(..., description="Tipo: LEI, DECRETO, IN, ACORDAO, etc")
    numero: str = Field(..., description="Número do documento")
    ano: int = Field(..., ge=1900, le=2100, description="Ano do documento")
    titulo: Optional[str] = Field(None, description="Título do documento (opcional)")

    # Campos específicos para Acórdãos TCU
    colegiado: Optional[str] = Field(None, description="Colegiado: P (Plenário), 1C, 2C")
    processo: Optional[str] = Field(None, description="Número do processo (TC xxx.xxx/xxxx-x)")
    relator: Optional[str] = Field(None, description="Nome do Ministro Relator")
    data_sessao: Optional[str] = Field(None, description="Data da sessão (DD/MM/YYYY)")
    unidade_tecnica: Optional[str] = Field(None, description="Unidade técnica responsável")
    unidade_jurisdicionada: Optional[str] = Field(None, description="Órgão/Entidade objeto da deliberação")

    # Configurações opcionais
    skip_enrichment: bool = Field(False, description="Pular enriquecimento LLM")
    skip_embeddings: bool = Field(False, description="Pular geração de embeddings")
    max_articles: Optional[int] = Field(None, description="Limite de artigos (para debug)")

    # Validação de artigos
    validate_articles: bool = Field(False, description="Habilita validação de artigos")
    expected_first_article: Optional[int] = Field(None, description="Primeiro artigo esperado (ex: 1)")
    expected_last_article: Optional[int] = Field(None, description="Último artigo esperado (ex: 193)")

    # PDF será enviado como multipart/form-data


class ProcessedChunk(BaseModel):
    """Chunk processado, pronto para indexação."""

    # Identificação
    node_id: str = Field(..., description="PK física: leis:DOC#SPAN_ID@Pxx")
    chunk_id: str = Field(..., description="ID único: DOC#SPAN_ID")
    parent_node_id: str = Field("", description="ID lógico do pai: leis:DOC#SPAN_ID (sem @Pxx, vazio se artigo)")
    span_id: str = Field(..., description="ID do span: ART-005, PAR-005-1, etc")
    device_type: str = Field(..., description="article, paragraph, inciso, alinea")
    chunk_level: str = Field(..., description="article ou device")

    # Conteúdo
    text: str = Field(..., description="Texto original do dispositivo")
    parent_text: str = Field("", description="Texto do artigo pai (caput) para dispositivos filhos")
    retrieval_text: str = Field("", description="Texto determinístico para embedding")
    enriched_text: str = Field("", description="[DEPRECATED] Texto enriquecido por LLM")
    context_header: str = Field("", description="[DEPRECATED] Frase de contexto")
    thesis_text: str = Field("", description="Resumo/tese do dispositivo")
    thesis_type: str = Field("", description="Tipo: definicao, procedimento, etc")
    synthetic_questions: str = Field("", description="Perguntas relacionadas")

    # Metadados do documento
    document_id: str
    tipo_documento: str
    numero: str
    ano: int
    article_number: str = Field("", description="Número do artigo")

    # Campos específicos para Acórdãos TCU (string vazia para LEIs)
    colegiado: str = Field("", description="Colegiado: P (Plenário), 1C, 2C")
    processo: str = Field("", description="Número do processo (TC xxx.xxx/xxxx-x)")
    relator: str = Field("", description="Nome do Ministro Relator")
    data_sessao: str = Field("", description="Data da sessão (DD/MM/YYYY)")
    unidade_tecnica: str = Field("", description="Unidade técnica responsável")

    # Vetores (opcionais, preenchidos se skip_embeddings=False)
    dense_vector: Optional[list[float]] = Field(None, description="Embedding dense 1024d")
    sparse_vector: Optional[dict[int, float]] = Field(None, description="Embedding sparse")
    thesis_vector: Optional[list[float]] = Field(None, description="Embedding da thesis")

    # Proveniência
    # PR5: citations agora inclui rel_type e rel_type_confidence
    # Formato: [{"target_node_id": "...", "rel_type": "CITA", "rel_type_confidence": 0.85}, ...]
    # Ou formato legado: ["target_node_id1", "target_node_id2", ...]
    citations: list = Field(default_factory=list, description="Citações com rel_type")
    schema_version: str = Field("1.0.0")

    # Campos adicionais para Milvus leis_v4
    aliases: str = Field("", description="Aliases/termos alternativos do chunk")
    sparse_source: str = Field("", description="Texto fonte para sparse embedding")

    # PR13: Offsets verdadeiros no canonical_text (zero fallback find)
    # Quando canonical_hash == hash_atual E start/end >= 0:
    #   → usa slicing puro: canonical_text[start:end]
    # Caso contrário:
    #   → fallback best-effort via find()
    canonical_start: int = Field(-1, description="Offset início no canonical_text (-1 se desconhecido)")
    canonical_end: int = Field(-1, description="Offset fim no canonical_text (-1 se desconhecido)")
    canonical_hash: str = Field("", description="SHA256 do canonical_text para anti-mismatch")

    # OriginClassifier: Classificação de origem do material
    # "self" = material da própria lei, "external" = material citado de outra lei
    origin_type: str = Field("self", description="Origem: self (própria lei) ou external (outra lei)")
    origin_confidence: str = Field("high", description="Confiança: high, medium, low")
    origin_reference: str = Field("", description="Referência externa (ex: DL-2848-1940) se origin_type=external")
    origin_reference_name: str = Field("", description="Nome da referência externa (ex: Código Penal)")
    is_external_material: bool = Field(False, description="True se material é de outra lei")
    origin_reason: str = Field("", description="Regra que determinou a classificação (ex: rule:codigo_penal_art337)")

    # VLM: Campos do pipeline Qwen3-VL + PyMuPDF
    page_number: int = Field(-1, description="Página do dispositivo no PDF (-1 se desconhecido)")
    bbox: list[float] = Field(default_factory=list, description="Bounding box no PDF [x0, y0, x1, y1]")
    confidence: float = Field(0.0, description="Confiança do VLM na classificação (0.0-1.0)")

    class Config:
        json_schema_extra = {
            "example": {
                "node_id": "leis:LEI-123-2006#ART-005",
                "chunk_id": "LEI-123-2006#ART-005",
                "parent_node_id": "",
                "span_id": "ART-005",
                "device_type": "article",
                "chunk_level": "article",
                "text": "Art. 5º Esta Lei estabelece...",
                "enriched_text": "[CONTEXTO: Este artigo...] Art. 5º...",
                "context_header": "Este artigo da Lei 123/2006 estabelece...",
                "thesis_text": "Define os critérios para...",
                "thesis_type": "definicao",
                "synthetic_questions": "O que estabelece o Art. 5?",
                "document_id": "LEI-123-2006",
                "tipo_documento": "LEI",
                "numero": "123",
                "ano": 2006,
                "article_number": "5",
                "dense_vector": [0.1, 0.2, ...],
                "sparse_vector": {1234: 0.5, 5678: 0.3},
                "citations": ["ART-005"],
                "schema_version": "1.0.0",
                "canonical_start": 245,
                "canonical_end": 412,
                "canonical_hash": "abc123def456...",
            }
        }


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

    # Validação de artigos (Fase Docling)
    validation_docling: Optional[dict] = Field(None, description="Resultado da validação de artigos")

    # Hash do documento
    document_hash: str = Field("", description="Hash do documento para deduplicação")

    # Tempo total (alias para compatibilidade)
    total_time_seconds: float = Field(0.0, description="Tempo total em segundos")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_id": "LEI-123-2006",
                "status": "completed",
                "chunks": [...],
                "total_chunks": 47,
                "phases": [
                    {"phase": "docling", "duration_ms": 5000, "success": True, "items_processed": 1},
                    {"phase": "parsing", "duration_ms": 50, "success": True, "items_processed": 50},
                ],
                "total_duration_ms": 15000,
                "errors": [],
                "articles_extracted": 11,
                "paragraphs_extracted": 19,
                "incisos_extracted": 17,
            }
        }
