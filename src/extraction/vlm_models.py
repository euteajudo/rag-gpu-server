"""
Modelos Pydantic para o pipeline VLM de extração de estrutura.

Define os modelos de dados para:
- PageData: dados brutos de uma página extraídos via PyMuPDF
- DeviceExtraction: um dispositivo legal extraído pelo VLM
- PageExtraction: resultado da extração VLM de uma página
- DocumentExtraction: resultado completo da extração VLM do documento
"""

from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class PageData:
    """Dados brutos de uma página extraídos via PyMuPDF."""

    page_number: int           # 1-indexed
    image_png: bytes           # PNG da página renderizada
    image_base64: str          # Base64 do PNG
    text: str                  # Texto nativo PyMuPDF
    width: float               # Largura da página em pontos
    height: float              # Altura da página em pontos


class DeviceExtraction(BaseModel):
    """Um dispositivo legal extraído pelo VLM."""

    device_type: str = Field(..., description="artigo, paragrafo, inciso, alinea")
    identifier: str = Field(..., description="Art. 5º, § 1º, I, a)")
    text: str = Field(..., description="Texto completo do dispositivo")
    parent_identifier: str = Field("", description="Identificador do pai (vazio se artigo)")
    bbox: list[float] = Field(default_factory=list, description="[x0, y0, x1, y1] normalizado 0-1")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confiança do VLM 0.0-1.0")


class PageExtraction(BaseModel):
    """Resultado da extração VLM de uma página."""

    page_number: int = Field(..., description="Número da página (1-indexed)")
    devices: list[DeviceExtraction] = Field(default_factory=list)


class DocumentExtraction(BaseModel):
    """Resultado completo da extração VLM do documento."""

    document_id: str
    pages: list[PageExtraction] = Field(default_factory=list)
    canonical_text: str = Field("", description="Texto PyMuPDF concatenado")
    canonical_hash: str = Field("", description="SHA256 do canonical_text normalizado")
    total_devices: int = Field(0, description="Total de dispositivos extraídos")
