"""
VLM Extraction Service - Orquestra a extração VLM de documentos legais.

Pipeline completo:
1. PyMuPDF: extrai páginas (imagens + texto determinístico)
2. Qwen3-VL: extrai estrutura de cada página (sequencial, uma por vez)
3. Concatena canonical_text de todas as páginas
4. Computa canonical_hash
5. Retorna DocumentExtraction

O processamento é SEQUENCIAL (uma página por vez) porque:
- O --max-model-len 8192 do vLLM limita o contexto
- Cada página é processada independentemente
- Evita sobrecarga de VRAM com múltiplas imagens simultâneas
"""

import logging
from typing import Optional, Callable

from ..utils.canonical_utils import normalize_canonical_text, compute_canonical_hash
from .pymupdf_extractor import PyMuPDFExtractor
from .vlm_client import VLMClient
from .vlm_models import (
    DeviceExtraction,
    DocumentExtraction,
    PageExtraction,
)

logger = logging.getLogger(__name__)


class VLMExtractionService:
    """Orquestra extração: PyMuPDF -> Qwen3-VL -> DocumentExtraction."""

    def __init__(
        self,
        vlm_client: VLMClient,
        pymupdf_extractor: PyMuPDFExtractor,
    ):
        """
        Args:
            vlm_client: Cliente multimodal para Qwen3-VL
            pymupdf_extractor: Extrator PyMuPDF para páginas
        """
        self.vlm_client = vlm_client
        self.pymupdf_extractor = pymupdf_extractor

    async def extract_document(
        self,
        pdf_bytes: bytes,
        document_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> DocumentExtraction:
        """
        Pipeline completo de extração VLM.

        1. PyMuPDF: extrai páginas (imagens + texto)
        2. Qwen3-VL: extrai estrutura de cada página (sequencial)
        3. Concatena canonical_text de todas as páginas
        4. Computa canonical_hash
        5. Retorna DocumentExtraction

        Args:
            pdf_bytes: Conteúdo binário do PDF
            document_id: ID do documento (ex: LEI-14133-2021)
            progress_callback: Callback (phase, progress) para reportar progresso

        Returns:
            DocumentExtraction com todos os dispositivos extraídos
        """
        def report(phase: str, progress: float):
            if progress_callback:
                try:
                    progress_callback(phase, progress)
                except Exception as e:
                    logger.warning(f"Erro no progress_callback: {e}")

        # === Etapa 1: PyMuPDF ===
        report("pymupdf_extraction", 0.10)
        logger.info(f"VLM Pipeline: Etapa 1 - Extraindo páginas com PyMuPDF ({document_id})")

        pages_data = self.pymupdf_extractor.extract_pages(pdf_bytes)
        total_pages = len(pages_data)

        if total_pages == 0:
            logger.warning(f"PyMuPDF retornou 0 páginas para {document_id}")
            return DocumentExtraction(
                document_id=document_id,
                pages=[],
                canonical_text="",
                canonical_hash="",
                total_devices=0,
            )

        logger.info(f"PyMuPDF: {total_pages} páginas extraídas")
        report("pymupdf_extraction", 0.20)

        # === Etapa 2: Qwen3-VL (sequencial, uma página por vez) ===
        logger.info(f"VLM Pipeline: Etapa 2 - Extraindo estrutura com Qwen3-VL ({total_pages} páginas)")

        page_extractions: list[PageExtraction] = []
        total_devices = 0

        for i, page_data in enumerate(pages_data):
            page_num = page_data.page_number
            progress = 0.20 + (0.60 * (i / total_pages))
            report("vlm_extraction", progress)

            logger.info(f"VLM: processando página {page_num}/{total_pages}")

            try:
                vlm_result = await self.vlm_client.extract_page(
                    image_base64=page_data.image_base64,
                )

                # Converte resultado VLM para modelo Pydantic
                devices = []
                for raw_device in vlm_result.get("devices", []):
                    try:
                        device = DeviceExtraction(
                            device_type=raw_device.get("device_type", ""),
                            identifier=raw_device.get("identifier", ""),
                            text=raw_device.get("text", ""),
                            parent_identifier=raw_device.get("parent_identifier", ""),
                            bbox=raw_device.get("bbox", []),
                            confidence=float(raw_device.get("confidence", 0.0)),
                        )
                        devices.append(device)
                    except Exception as e:
                        logger.warning(
                            f"Erro ao parsear dispositivo VLM na página {page_num}: {e}"
                        )

                page_extraction = PageExtraction(
                    page_number=page_num,
                    devices=devices,
                )
                page_extractions.append(page_extraction)
                total_devices += len(devices)

                logger.info(
                    f"VLM página {page_num}: {len(devices)} dispositivos extraídos"
                )

            except Exception as e:
                logger.error(
                    f"Erro VLM na página {page_num}/{total_pages}: {e}",
                    exc_info=True,
                )
                # Adiciona página vazia para manter a contagem
                page_extractions.append(PageExtraction(
                    page_number=page_num,
                    devices=[],
                ))

        report("vlm_extraction", 0.80)

        # === Etapa 3: Construir canonical_text ===
        logger.info("VLM Pipeline: Etapa 3 - Construindo canonical_text")

        # Concatena texto de todas as páginas (texto do PyMuPDF, determinístico)
        raw_canonical = "\n".join(
            page_data.text for page_data in pages_data
        )

        # Normaliza e computa hash
        canonical_text = normalize_canonical_text(raw_canonical)
        canonical_hash = compute_canonical_hash(canonical_text)

        report("building_canonical", 0.90)

        logger.info(
            f"VLM Pipeline: {total_devices} dispositivos em {total_pages} páginas, "
            f"canonical_text={len(canonical_text)} chars, "
            f"hash={canonical_hash[:16]}..."
        )

        return DocumentExtraction(
            document_id=document_id,
            pages=page_extractions,
            canonical_text=canonical_text,
            canonical_hash=canonical_hash,
            total_devices=total_devices,
        )
