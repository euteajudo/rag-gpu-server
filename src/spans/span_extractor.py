"""
[DEPRECATED] Extrator de spans a partir do markdown canônico.

PR3 v2.1 - Este módulo foi DEPRECATED em favor de:
    - parsing/span_parser.py: Parser determinístico mais robusto
    - bridge/parsed_document_chunkparts.py: Converte ParsedDocument -> ChunkPart[]

MOTIVO DA DEPRECAÇÃO:
    O parsing/span_parser.py é significativamente mais robusto:
    - 4 camadas anti-alucinação (schema enum dinâmico, validação de cobertura, retry focado, validação de IDs)
    - Tratamento de quirks do Docling (prefixos numéricos como "12. Art. 5")
    - Índice O(1) para lookup de spans
    - Métodos úteis: get_span(), get_children(), reconstruct_text(), to_annotated_markdown()
    - Desambiguação de incisos (INC-005-I vs INC-005-I_2)

MIGRAÇÃO:
    # ANTES (deprecated):
    from spans import SpanExtractor
    extractor = SpanExtractor(document_id, document_type)
    result = extractor.extract(markdown)
    spans = result.spans

    # DEPOIS (recomendado):
    from parsing import SpanParser
    from bridge import ParsedDocumentChunkPartsBuilder

    parser = SpanParser()
    parsed_doc = parser.parse(markdown)

    builder = ParsedDocumentChunkPartsBuilder(document_id, document_type)
    chunks = builder.build(parsed_doc)

DATA: 2025-01-30
PR3 v2 - Hard Reset RAG Architecture (Original)
PR3 v2.1 - Rebase para usar parsing/ robusto
"""

import warnings

warnings.warn(
    "spans.span_extractor está deprecated. Use parsing.SpanParser + bridge.ParsedDocumentChunkPartsBuilder",
    DeprecationWarning,
    stacklevel=2,
)

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from .span_types import Span, DeviceType
from ..canonical import build_logical_node_id, get_prefix_for_document_type

logger = logging.getLogger(__name__)


@dataclass
class SpanExtractionResult:
    """Resultado da extração de spans."""

    spans: list[Span]
    total_chars: int
    article_count: int
    paragraph_count: int
    inciso_count: int
    alinea_count: int
    warnings: list[str] = field(default_factory=list)


class SpanExtractor:
    """
    Extrai spans hierárquicos do markdown canônico.

    Reconhece padrões de dispositivos legais brasileiros:
    - Art. 1º, Art. 2, Artigo 3
    - § 1º, § único, Parágrafo único
    - I -, II -, III - (incisos romanos)
    - a), b), c) (alíneas)
    """

    # Regex patterns para dispositivos legais
    ARTICLE_PATTERN = re.compile(
        r"^(?:Art\.?|Artigo)\s*(\d+)[°ºo]?\s*[-–—.]?\s*",
        re.IGNORECASE | re.MULTILINE,
    )

    PARAGRAPH_PATTERN = re.compile(
        r"^(?:§\s*(\d+)[°ºo]?|§\s*[Úú]nico|Par[áa]grafo\s+[Úú]nico)\s*[-–—.]?\s*",
        re.IGNORECASE | re.MULTILINE,
    )

    INCISO_PATTERN = re.compile(
        r"^([IVXLCDM]+)\s*[-–—.]\s*",
        re.MULTILINE,
    )

    ALINEA_PATTERN = re.compile(
        r"^([a-z])\s*\)\s*",
        re.MULTILINE,
    )

    def __init__(
        self,
        document_id: str,
        document_type: str = "LEI",
    ):
        """
        Inicializa o extrator.

        Args:
            document_id: ID do documento (ex: LEI-14133-2021)
            document_type: Tipo do documento (LEI, DECRETO, IN, etc.)
        """
        self.document_id = document_id
        self.document_type = document_type.upper()
        self.prefix = get_prefix_for_document_type(self.document_type)

    def extract(self, canonical_md: str) -> SpanExtractionResult:
        """
        Extrai spans do markdown canônico.

        Args:
            canonical_md: Markdown gerado pelo Docling

        Returns:
            SpanExtractionResult com lista de spans e métricas
        """
        spans: list[Span] = []
        warnings: list[str] = []

        # Divide por artigos primeiro
        article_blocks = self._split_by_articles(canonical_md)

        article_count = 0
        paragraph_count = 0
        inciso_count = 0
        alinea_count = 0

        for article_num, article_text in article_blocks:
            # Cria span do artigo
            article_span_id = f"ART-{article_num:03d}"
            article_logical_id = build_logical_node_id(
                self.prefix, self.document_id, article_span_id
            )

            article_span = Span(
                logical_node_id=article_logical_id,
                document_id=self.document_id,
                span_id=article_span_id,
                parent_span_id=None,  # Artigo é raiz
                device_type=DeviceType.ARTICLE,
                text=article_text.strip(),
                article_number=str(article_num),
                document_type=self.document_type,
            )
            spans.append(article_span)
            article_count += 1

            # Extrai parágrafos do artigo
            paragraph_spans = self._extract_paragraphs(
                article_text, article_num, article_span_id
            )
            spans.extend(paragraph_spans)
            paragraph_count += len(paragraph_spans)

            # Extrai incisos diretamente do artigo (não de parágrafos)
            inciso_spans = self._extract_incisos(
                article_text, article_num, article_span_id, parent_is_paragraph=False
            )
            spans.extend(inciso_spans)
            inciso_count += len(inciso_spans)

            # Para cada inciso, extrai alíneas
            for inciso_span in inciso_spans:
                alinea_spans = self._extract_alineas(
                    inciso_span.text, article_num, inciso_span.span_id
                )
                spans.extend(alinea_spans)
                alinea_count += len(alinea_spans)

            # Extrai incisos de cada parágrafo
            for para_span in paragraph_spans:
                para_incisos = self._extract_incisos(
                    para_span.text, article_num, para_span.span_id, parent_is_paragraph=True
                )
                spans.extend(para_incisos)
                inciso_count += len(para_incisos)

                # Para cada inciso do parágrafo, extrai alíneas
                for inc_span in para_incisos:
                    inc_alineas = self._extract_alineas(
                        inc_span.text, article_num, inc_span.span_id
                    )
                    spans.extend(inc_alineas)
                    alinea_count += len(inc_alineas)

        if article_count == 0:
            warnings.append("Nenhum artigo encontrado no documento")

        logger.info(
            f"Extração concluída: {article_count} artigos, "
            f"{paragraph_count} parágrafos, {inciso_count} incisos, "
            f"{alinea_count} alíneas"
        )

        return SpanExtractionResult(
            spans=spans,
            total_chars=len(canonical_md),
            article_count=article_count,
            paragraph_count=paragraph_count,
            inciso_count=inciso_count,
            alinea_count=alinea_count,
            warnings=warnings,
        )

    def _split_by_articles(self, text: str) -> list[tuple[int, str]]:
        """
        Divide o texto em blocos por artigo.

        Returns:
            Lista de tuplas (numero_artigo, texto_artigo)
        """
        articles = []

        # Encontra todas as posições de artigos
        matches = list(self.ARTICLE_PATTERN.finditer(text))

        if not matches:
            return articles

        for i, match in enumerate(matches):
            article_num = int(match.group(1))
            start = match.start()

            # Fim é o início do próximo artigo ou fim do texto
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            article_text = text[start:end]
            articles.append((article_num, article_text))

        return articles

    def _extract_paragraphs(
        self,
        article_text: str,
        article_num: int,
        article_span_id: str,
    ) -> list[Span]:
        """Extrai parágrafos de um artigo."""
        paragraphs = []

        matches = list(self.PARAGRAPH_PATTERN.finditer(article_text))

        for i, match in enumerate(matches):
            # Determina número do parágrafo
            if match.group(1):
                para_num = match.group(1)
            else:
                para_num = "UNICO"

            para_span_id = f"PAR-{article_num:03d}-{para_num}"
            para_logical_id = build_logical_node_id(
                self.prefix, self.document_id, para_span_id
            )

            # Texto do parágrafo vai até o próximo parágrafo ou fim
            start = match.start()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                # Vai até o fim do artigo, mas não inclui próximo dispositivo
                end = len(article_text)

            para_text = article_text[start:end].strip()

            para_span = Span(
                logical_node_id=para_logical_id,
                document_id=self.document_id,
                span_id=para_span_id,
                parent_span_id=article_span_id,
                device_type=DeviceType.PARAGRAPH,
                text=para_text,
                article_number=str(article_num),
                document_type=self.document_type,
            )
            paragraphs.append(para_span)

        return paragraphs

    def _extract_incisos(
        self,
        text: str,
        article_num: int,
        parent_span_id: str,
        parent_is_paragraph: bool = False,
    ) -> list[Span]:
        """Extrai incisos de um artigo ou parágrafo."""
        incisos = []

        matches = list(self.INCISO_PATTERN.finditer(text))

        for i, match in enumerate(matches):
            inciso_roman = match.group(1)

            # Constrói span_id baseado no parent
            if parent_is_paragraph:
                # INC-005-I_1 (inciso I do parágrafo 1 do artigo 5)
                # Extrai número do parágrafo do parent_span_id
                para_suffix = parent_span_id.split("-")[-1]
                inciso_span_id = f"INC-{article_num:03d}-{inciso_roman}_{para_suffix}"
            else:
                # INC-005-I (inciso I do artigo 5)
                inciso_span_id = f"INC-{article_num:03d}-{inciso_roman}"

            inciso_logical_id = build_logical_node_id(
                self.prefix, self.document_id, inciso_span_id
            )

            # Texto vai até o próximo inciso
            start = match.start()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            inciso_text = text[start:end].strip()

            inciso_span = Span(
                logical_node_id=inciso_logical_id,
                document_id=self.document_id,
                span_id=inciso_span_id,
                parent_span_id=parent_span_id,
                device_type=DeviceType.INCISO,
                text=inciso_text,
                article_number=str(article_num),
                document_type=self.document_type,
            )
            incisos.append(inciso_span)

        return incisos

    def _extract_alineas(
        self,
        text: str,
        article_num: int,
        parent_span_id: str,
    ) -> list[Span]:
        """Extrai alíneas de um inciso."""
        alineas = []

        matches = list(self.ALINEA_PATTERN.finditer(text))

        for i, match in enumerate(matches):
            alinea_letter = match.group(1)

            # ALI-005-I-a (alínea a do inciso I do artigo 5)
            alinea_span_id = f"ALI-{parent_span_id.replace('INC-', '')}-{alinea_letter}"
            alinea_logical_id = build_logical_node_id(
                self.prefix, self.document_id, alinea_span_id
            )

            # Texto vai até a próxima alínea
            start = match.start()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            alinea_text = text[start:end].strip()

            alinea_span = Span(
                logical_node_id=alinea_logical_id,
                document_id=self.document_id,
                span_id=alinea_span_id,
                parent_span_id=parent_span_id,
                device_type=DeviceType.ALINEA,
                text=alinea_text,
                article_number=str(article_num),
                document_type=self.document_type,
            )
            alineas.append(alinea_span)

        return alineas


def extract_spans(
    canonical_md: str,
    document_id: str,
    document_type: str = "LEI",
) -> SpanExtractionResult:
    """
    Função de conveniência para extrair spans.

    Args:
        canonical_md: Markdown canônico
        document_id: ID do documento
        document_type: Tipo do documento

    Returns:
        SpanExtractionResult
    """
    extractor = SpanExtractor(document_id, document_type)
    return extractor.extract(canonical_md)
