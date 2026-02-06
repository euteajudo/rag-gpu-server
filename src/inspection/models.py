"""
Modelos Pydantic para o Pipeline Inspector.

Define os artefatos de cada fase de inspeção:
  1. PyMuPDF  — blocos de texto extraídos com bboxes
  2. VLM      — elementos detectados pelo Qwen3-VL
  3. Reconciliação — matches entre PyMuPDF e VLM
  4. Integridade   — resultado do IntegrityValidator
  5. Chunks        — preview dos chunks que seriam criados
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class InspectionStage(str, Enum):
    """Fases do pipeline de inspeção."""
    PYMUPDF = "pymupdf"
    VLM = "vlm"
    RECONCILIATION = "reconciliation"
    INTEGRITY = "integrity"
    CHUNKS = "chunks"


class InspectionStatus(str, Enum):
    """Status de uma inspeção."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    APPROVED = "approved"
    FAILED = "failed"


# ============================================================================
# Request
# ============================================================================

class InspectRequest(BaseModel):
    """Request para iniciar inspeção de um PDF."""
    document_id: str = Field(..., description="ID único do documento (ex: LEI-123-2006)")
    tipo_documento: str = Field(..., description="Tipo: LEI, DECRETO, IN, ACORDAO, etc")
    numero: str = Field(..., description="Número do documento")
    ano: int = Field(..., ge=1900, le=2100, description="Ano do documento")
    titulo: Optional[str] = Field(None, description="Título do documento (opcional)")


# ============================================================================
# Fase 1: PyMuPDF — Blocos de texto extraídos
# ============================================================================

class BBox(BaseModel):
    """Bounding box de um elemento na página."""
    x0: float = Field(..., description="Coordenada X esquerda")
    y0: float = Field(..., description="Coordenada Y topo")
    x1: float = Field(..., description="Coordenada X direita")
    y1: float = Field(..., description="Coordenada Y base")


class PyMuPDFBlock(BaseModel):
    """Bloco de texto extraído pelo PyMuPDF."""
    block_index: int = Field(..., description="Índice do bloco na página")
    text: str = Field(..., description="Texto extraído")
    bbox: BBox = Field(..., description="Bounding box do bloco")
    font_size: float = Field(0.0, description="Tamanho da fonte predominante")
    is_bold: bool = Field(False, description="Se o texto é negrito")
    page: int = Field(..., description="Número da página (0-indexed)")


class PyMuPDFPageResult(BaseModel):
    """Resultado do PyMuPDF para uma página."""
    page_number: int = Field(..., description="Número da página (0-indexed)")
    width: float = Field(..., description="Largura da página em pontos")
    height: float = Field(..., description="Altura da página em pontos")
    blocks: list[PyMuPDFBlock] = Field(default_factory=list)
    image_base64: str = Field("", description="Imagem PNG da página com bboxes anotados (base64)")


class PyMuPDFArtifact(BaseModel):
    """Artefato completo da fase PyMuPDF."""
    pages: list[PyMuPDFPageResult] = Field(default_factory=list)
    total_blocks: int = Field(0)
    total_pages: int = Field(0)
    total_chars: int = Field(0)
    duration_ms: float = Field(0.0)


# ============================================================================
# Fase 2: VLM — Elementos detectados pelo Qwen3-VL
# ============================================================================

class VLMElement(BaseModel):
    """Elemento detectado pelo VLM (Qwen3-VL)."""
    element_id: str = Field(..., description="ID do elemento (ex: article_1, paragraph_2)")
    element_type: str = Field(..., description="Tipo: article, paragraph, inciso, alinea, header, table")
    text: str = Field("", description="Texto detectado pelo VLM")
    bbox: Optional[BBox] = Field(None, description="Bounding box na página")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confiança da detecção")
    page: int = Field(..., description="Número da página")
    parent_id: Optional[str] = Field(None, description="ID do elemento pai (hierarquia)")
    children_ids: list[str] = Field(default_factory=list, description="IDs dos filhos")


class VLMPageResult(BaseModel):
    """Resultado do VLM para uma página."""
    page_number: int
    elements: list[VLMElement] = Field(default_factory=list)
    image_base64: str = Field("", description="Imagem PNG com bboxes VLM anotados (base64)")


class VLMArtifact(BaseModel):
    """Artefato completo da fase VLM."""
    pages: list[VLMPageResult] = Field(default_factory=list)
    total_elements: int = Field(0)
    total_pages: int = Field(0)
    hierarchy_depth: int = Field(0, description="Profundidade máxima da hierarquia")
    duration_ms: float = Field(0.0)


# ============================================================================
# Fase 3: Reconciliação — Matches entre PyMuPDF e VLM
# ============================================================================

class ReconciliationMatch(BaseModel):
    """Match entre um bloco PyMuPDF e um elemento VLM."""
    pymupdf_block_index: int = Field(..., description="Índice do bloco PyMuPDF")
    vlm_element_id: str = Field(..., description="ID do elemento VLM")
    match_quality: str = Field(..., description="exact, partial, conflict, unmatched_pymupdf, unmatched_vlm")
    text_pymupdf: str = Field("", description="Texto do bloco PyMuPDF")
    text_vlm: str = Field("", description="Texto do elemento VLM")
    text_reconciled: str = Field("", description="Texto reconciliado final")
    bbox_overlap: float = Field(0.0, ge=0.0, le=1.0, description="IoU das bounding boxes")
    page: int = Field(...)


class ReconciliationStats(BaseModel):
    """Estatísticas da reconciliação."""
    total_matches: int = 0
    exact_matches: int = 0
    partial_matches: int = 0
    conflicts: int = 0
    unmatched_pymupdf: int = 0
    unmatched_vlm: int = 0
    coverage_pymupdf: float = 0.0
    coverage_vlm: float = 0.0


class ReconciliationPageResult(BaseModel):
    """Resultado da reconciliação para uma página."""
    page_number: int
    matches: list[ReconciliationMatch] = Field(default_factory=list)
    image_base64: str = Field("", description="Imagem PNG com matches anotados (base64)")


class ReconciliationArtifact(BaseModel):
    """Artefato completo da fase de reconciliação."""
    pages: list[ReconciliationPageResult] = Field(default_factory=list)
    stats: ReconciliationStats = Field(default_factory=lambda: ReconciliationStats())
    canonical_text: str = Field("", description="Texto canônico reconciliado completo")
    duration_ms: float = Field(0.0)


# ============================================================================
# Fase 4: Integridade — Resultado do IntegrityValidator
# ============================================================================

class IntegrityCheck(BaseModel):
    """Resultado de uma verificação de integridade."""
    check_name: str = Field(..., description="Nome: slicing, hierarchy, overlap, coverage, ordering")
    passed: bool = Field(...)
    message: str = Field("")
    details: Optional[dict] = Field(None, description="Detalhes adicionais do check")


class IntegrityArtifact(BaseModel):
    """Artefato completo da fase de integridade."""
    checks: list[IntegrityCheck] = Field(default_factory=list)
    overall_score: float = Field(0.0, ge=0.0, le=1.0, description="Score geral de integridade")
    passed: bool = Field(False, description="Se passou em todos os checks críticos")
    total_checks: int = Field(0)
    passed_checks: int = Field(0)
    failed_checks: int = Field(0)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float = Field(0.0)


# ============================================================================
# Fase 5: Chunks — Preview dos chunks que seriam criados
# ============================================================================

class ChunkPreview(BaseModel):
    """Preview de um chunk que seria criado na ingestão."""
    node_id: str = Field(..., description="PK física: leis:DOC#SPAN_ID")
    chunk_id: str = Field(..., description="ID: DOC#SPAN_ID")
    parent_node_id: str = Field("", description="ID do pai (vazio se artigo)")
    span_id: str = Field(..., description="ART-005, PAR-005-1, etc")
    device_type: str = Field(..., description="article, paragraph, inciso, alinea")
    chunk_level: str = Field(..., description="article ou device")
    text: str = Field(..., description="Texto do chunk")
    canonical_start: int = Field(-1, description="Offset início no canonical_text")
    canonical_end: int = Field(-1, description="Offset fim no canonical_text")
    children_count: int = Field(0, description="Número de filhos diretos")


class ChunksPreviewArtifact(BaseModel):
    """Artefato completo da fase de preview de chunks."""
    chunks: list[ChunkPreview] = Field(default_factory=list)
    total_chunks: int = Field(0)
    articles_count: int = Field(0)
    paragraphs_count: int = Field(0)
    incisos_count: int = Field(0)
    alineas_count: int = Field(0)
    max_depth: int = Field(0, description="Profundidade máxima da hierarquia")
    duration_ms: float = Field(0.0)


# ============================================================================
# Metadados e Aprovação
# ============================================================================

class InspectionMetadata(BaseModel):
    """Metadados de uma inspeção."""
    inspection_id: str = Field(..., description="ID único da inspeção (task_id)")
    document_id: str = Field(...)
    tipo_documento: str = Field("")
    numero: str = Field("")
    ano: int = Field(0)
    pdf_hash: str = Field("", description="SHA-256 do PDF original")
    pdf_size_bytes: int = Field(0)
    total_pages: int = Field(0)
    started_at: str = Field("")
    completed_at: Optional[str] = Field(None)
    approved_at: Optional[str] = Field(None)
    approved_by: Optional[str] = Field(None)
    status: InspectionStatus = Field(InspectionStatus.PENDING)


class ApprovalResult(BaseModel):
    """Resultado da aprovação de uma inspeção."""
    success: bool = Field(...)
    inspection_id: str = Field(...)
    document_id: str = Field(...)
    minio_path: str = Field("", description="Path base no MinIO: inspections/{document_id}/")
    artifacts_persisted: list[str] = Field(
        default_factory=list,
        description="Lista de artefatos persistidos no MinIO",
    )
    canonical_md_size: int = Field(0, description="Tamanho do canonical.md em bytes")
    message: str = Field("")
