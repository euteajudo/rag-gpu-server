"""
Canonicalizador: PDF → Markdown.

PR3 v2 - Hard Reset RAG Architecture

Usa Docling para converter PDF para markdown estruturado.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CanonicalizationResult:
    """Resultado da canonicalização."""

    markdown: str
    sha256: str
    page_count: int
    char_count: int


class Canonicalizer:
    """
    Converte PDF para markdown canônico usando Docling.

    O markdown gerado é a "fonte de verdade" para extração de spans.
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        language: str = "pt",
    ):
        """
        Inicializa o canonicalizador.

        Args:
            enable_ocr: Se True, usa OCR para PDFs escaneados
            language: Idioma para OCR (default: português)
        """
        self.enable_ocr = enable_ocr
        self.language = language
        self._converter = None

    def _get_converter(self):
        """Lazy-load do DocumentConverter do Docling."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter

                self._converter = DocumentConverter()
            except ImportError as e:
                raise ImportError(
                    "Docling não instalado. Instale com: pip install docling"
                ) from e
        return self._converter

    def canonicalize(
        self,
        pdf_bytes: bytes,
        filename: Optional[str] = None,
    ) -> CanonicalizationResult:
        """
        Converte PDF para markdown canônico.

        Args:
            pdf_bytes: Bytes do PDF
            filename: Nome do arquivo (opcional, para logging)

        Returns:
            CanonicalizationResult com markdown, sha256, métricas
        """
        import tempfile
        import os

        logger.info(f"Canonicalizando PDF: {filename or 'unknown'} ({len(pdf_bytes)} bytes)")

        # Salva em arquivo temporário (Docling requer arquivo)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            temp_path = f.name

        try:
            converter = self._get_converter()
            result = converter.convert(temp_path)

            # Exporta para markdown
            markdown = result.document.export_to_markdown()

            # Calcula hash
            sha256 = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

            # Métricas
            page_count = len(result.document.pages) if result.document.pages else 0
            char_count = len(markdown)

            logger.info(
                f"Canonicalização concluída: {page_count} páginas, "
                f"{char_count} caracteres, sha256={sha256[:16]}..."
            )

            return CanonicalizationResult(
                markdown=markdown,
                sha256=sha256,
                page_count=page_count,
                char_count=char_count,
            )

        finally:
            # Remove arquivo temporário
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def canonicalize_from_file(
        self,
        pdf_path: str,
    ) -> CanonicalizationResult:
        """
        Converte PDF de um arquivo para markdown canônico.

        Args:
            pdf_path: Caminho do arquivo PDF

        Returns:
            CanonicalizationResult
        """
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        return self.canonicalize(pdf_bytes, filename=pdf_path)
