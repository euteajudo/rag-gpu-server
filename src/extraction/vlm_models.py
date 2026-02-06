"""
Modelos Pydantic para o pipeline VLM de extração de estrutura.

Define os modelos de dados para:
- BlockData: bloco de texto PyMuPDF com offset no canonical_text
- PageData: dados brutos de uma página extraídos via PyMuPDF
- DeviceExtraction: um dispositivo legal extraído pelo VLM
- PageExtraction: resultado da extração VLM de uma página
- DocumentExtraction: resultado completo da extração VLM do documento
"""

from dataclasses import dataclass, field
from pydantic import BaseModel, Field


@dataclass
class BlockData:
    """Um bloco de texto extraído pelo PyMuPDF com offset no canonical_text."""

    block_index: int          # índice do bloco na página
    char_start: int           # offset início no canonical_text global
    char_end: int             # offset fim no canonical_text global
    bbox_pdf: list[float]     # [x0, y0, x1, y1] em pontos PDF (72 DPI)
    text: str                 # texto do bloco
    page_number: int          # página de origem (1-indexed)


@dataclass
class PageData:
    """Dados brutos de uma página extraídos via PyMuPDF."""

    page_number: int           # 1-indexed
    image_png: bytes           # PNG da página renderizada
    image_base64: str          # Base64 do PNG
    text: str                  # Texto concatenado dos blocos desta página
    width: float               # Largura da página em pontos PDF
    height: float              # Altura da página em pontos PDF
    img_width: int = 0         # Largura do pixmap em pixels
    img_height: int = 0        # Altura do pixmap em pixels
    blocks: list[BlockData] = field(default_factory=list)  # Blocos com offsets
    char_start: int = 0        # Offset do início desta página no canonical_text
    char_end: int = 0          # Offset do fim desta página no canonical_text


class DeviceExtraction(BaseModel):
    """Um dispositivo legal extraído pelo VLM."""

    device_type: str = Field(..., description="artigo, paragrafo, inciso, alinea")
    identifier: str = Field(..., description="Art. 5º, § 1º, I, a)")
    text: str = Field(..., description="Texto completo do dispositivo")
    parent_identifier: str = Field("", description="Identificador do pai (vazio se artigo)")
    bbox: list[float] = Field(default_factory=list, description="[x0, y0, x1, y1] normalizado 0-1 (image space)")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confiança do VLM 0.0-1.0")


class PageExtraction(BaseModel):
    """Resultado da extração VLM de uma página."""

    page_number: int = Field(..., description="Número da página (1-indexed)")
    devices: list[DeviceExtraction] = Field(default_factory=list)


class DocumentExtraction(BaseModel):
    """Resultado completo da extração VLM do documento."""

    document_id: str
    pages: list[PageExtraction] = Field(default_factory=list)
    canonical_text: str = Field("", description="Texto PyMuPDF concatenado dos blocos")
    canonical_hash: str = Field("", description="SHA256 do canonical_text normalizado")
    total_devices: int = Field(0, description="Total de dispositivos extraídos")
    pages_data: list = Field(default_factory=list, description="list[PageData] com blocos e offsets (não serializado)")
