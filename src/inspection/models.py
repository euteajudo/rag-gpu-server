"""
Modelos Pydantic para o Pipeline Inspector v4 (Regex).

Artefatos da inspeção regex (Entrada 1 — PyMuPDF + Regex):
  - RegexClassificationArtifact — resultado completo da classificação
  - PyMuPDF stage — blocos extraídos (reusa InspectionStage.PYMUPDF)
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
    REGEX_CLASSIFICATION = "regex_classification"
    VLM_CLASSIFICATION = "vlm_classification"


class InspectionStatus(str, Enum):
    """Status de uma inspeção."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Fase Regex — Classificação de dispositivos via regex (Entrada 1)
# ============================================================================

class RegexDevice(BaseModel):
    """Dispositivo classificado pelo regex."""
    span_id: str = Field(..., description="ART-001, PAR-003-1, INC-009-III, ALI-009-III-a")
    device_type: str = Field(..., description="article, paragraph, inciso, alinea")
    identifier: str = Field(..., description="Art. 1º, § 1º, I, a")
    parent_span_id: str = Field("", description="Span ID do pai (vazio para artigos)")
    children_span_ids: list[str] = Field(default_factory=list)
    hierarchy_depth: int = Field(0, description="0=art, 1=§/inc, 2=inc sob §, 3=alínea")
    text: str = Field(..., description="Texto completo do dispositivo")
    text_preview: str = Field("", description="Primeiros 120 chars")
    char_start: int = Field(..., description="Offset início no canonical_text")
    char_end: int = Field(..., description="Offset fim no canonical_text")
    page_number: int = Field(...)
    bbox: list[float] = Field(default_factory=list, description="[x0, y0, x1, y1] PDF points")


class RegexFilteredBlock(BaseModel):
    """Bloco filtrado (não normativo)."""
    block_index: int
    page_number: int
    filter_type: str = Field(..., description="metadata, cabecalho, preambulo")
    reason: str = Field("")
    text_preview: str = Field("")


class RegexUnclassifiedBlock(BaseModel):
    """Bloco não classificado."""
    block_index: int
    page_number: int
    reason: str = Field("")
    text_preview: str = Field("")


class RegexClassificationStats(BaseModel):
    """Estatísticas da classificação regex."""
    total_blocks: int = Field(0)
    devices: int = Field(0)
    filtered: int = Field(0)
    unclassified: int = Field(0)
    by_device_type: dict[str, int] = Field(default_factory=dict)
    by_filter_type: dict[str, int] = Field(default_factory=dict)
    max_hierarchy_depth: int = Field(0)


class RegexOffsetCheck(BaseModel):
    """Resultado da verificação de um offset."""
    span_id: str
    page: int
    char_start: int
    char_end: int
    match: bool
    expected_preview: str = Field("")
    got_preview: str = Field("")


class RegexIntegrityChecks(BaseModel):
    """Verificações de integridade do canonical_text."""
    all_pass: bool = Field(False)
    offsets_pass: bool = Field(False)
    offsets_total: int = Field(0)
    offsets_matches: int = Field(0)
    offsets_details: list[RegexOffsetCheck] = Field(default_factory=list)
    normalization_idempotent: bool = Field(False)
    no_trailing_spaces: bool = Field(False)
    trailing_space_violations: int = Field(0)
    unicode_nfc: bool = Field(False)
    trailing_newline: bool = Field(False)


class RegexClassificationArtifact(BaseModel):
    """Artefato completo da classificação regex (Entrada 1 e 2)."""
    devices: list[RegexDevice] = Field(default_factory=list)
    filtered: list[RegexFilteredBlock] = Field(default_factory=list)
    unclassified: list[RegexUnclassifiedBlock] = Field(default_factory=list)
    stats: RegexClassificationStats = Field(default_factory=RegexClassificationStats)
    checks: RegexIntegrityChecks = Field(default_factory=RegexIntegrityChecks)
    canonical_text: str = Field("", description="Texto canônico completo")
    canonical_hash: str = Field("")
    canonical_length: int = Field(0)
    duration_ms: float = Field(0.0)
    extraction_source: str = Field("pymupdf_native", description="pymupdf_native ou vlm_ocr")


# ============================================================================
# Fase VLM — Classificação de dispositivos via Qwen3-VL (Entrada 2)
# ============================================================================

class VLMDevice(BaseModel):
    """Dispositivo classificado pelo VLM (Qwen3-VL)."""
    span_id: str = Field(..., description="ART-005, PAR-005-1, etc.")
    device_type: str = Field(..., description="article, paragraph, inciso, alinea")
    identifier: str = Field(..., description="Span ID usado como identifier")
    parent_span_id: str = Field("", description="Span ID do pai")
    text_preview: str = Field("", description="Primeiros 120 chars")
    char_start: int = Field(-1, description="Offset resolvido (-1 se sentinel)")
    char_end: int = Field(-1, description="Offset resolvido (-1 se sentinel)")
    page_number: int = Field(0)
    bbox_pdf: list[float] = Field(default_factory=list, description="PDF points")
    bbox_img: list[float] = Field(default_factory=list, description="0-1 normalized")
    confidence: float = Field(0.0, description="VLM confidence")
    resolution_phase: str = Field("", description="A_bbox, B_find, C_parent, etc.")
    is_cross_page: bool = Field(False)


class VLMResolutionSummary(BaseModel):
    """Sumário da resolução de offsets VLM."""
    total_chunks: int = Field(0)
    resolved: int = Field(0)
    sentinel: int = Field(0, description="Chunks sem offset resolvido")
    by_phase: dict[str, int] = Field(default_factory=dict, description="Ex: A_bbox=30, B_find=8")
    resolution_rate_pct: float = Field(0.0)


class VLMIntegrityChecks(BaseModel):
    """Verificações de integridade para pipeline VLM."""
    all_pass: bool = Field(False)
    offsets_pass: bool = Field(False)
    offsets_total: int = Field(0)
    offsets_matches: int = Field(0)
    normalization_idempotent: bool = Field(False)
    unicode_nfc: bool = Field(False)
    trailing_newline: bool = Field(False)
    cross_page_count: int = Field(0)
    orphan_drops: int = Field(0, description="Filhos sem parent_identifier dropados")


class VLMClassificationArtifact(BaseModel):
    """Artefato completo da classificação VLM (Entrada 2)."""
    devices: list[VLMDevice] = Field(default_factory=list)
    resolution: VLMResolutionSummary = Field(default_factory=VLMResolutionSummary)
    checks: VLMIntegrityChecks = Field(default_factory=VLMIntegrityChecks)
    canonical_text: str = Field("", description="Texto canônico completo")
    canonical_hash: str = Field("")
    canonical_length: int = Field(0)
    duration_ms: float = Field(0.0)
    vlm_model: str = Field("", description="Ex: Qwen/Qwen3-VL-8B-Instruct")
    total_pages: int = Field(0)


# ============================================================================
# PyMuPDF — Blocos extraídos (usado pelo inspector para exibir páginas)
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
    page: int = Field(..., description="Número da página (0-indexed)")


class PyMuPDFPageResult(BaseModel):
    """Resultado do PyMuPDF para uma página."""
    page_number: int = Field(..., description="Número da página (0-indexed)")
    width: float = Field(..., description="Largura da página em pontos")
    height: float = Field(..., description="Altura da página em pontos")
    blocks: list[PyMuPDFBlock] = Field(default_factory=list)
    image_base64: str = Field("", description="Imagem PNG da página (base64)")


class PyMuPDFArtifact(BaseModel):
    """Artefato completo da fase PyMuPDF."""
    pages: list[PyMuPDFPageResult] = Field(default_factory=list)
    total_blocks: int = Field(0)
    total_pages: int = Field(0)
    total_chars: int = Field(0)
    duration_ms: float = Field(0.0)


# ============================================================================
# Metadados
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
    status: InspectionStatus = Field(InspectionStatus.PENDING)
