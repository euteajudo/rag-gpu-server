"""
Bridge: ParsedDocument (parsing/) -> ChunkPart[] (spans/)

PR3 v2.1 - Rebase

Este módulo faz a ponte entre o parser robusto (parsing/span_parser.py)
e os tipos físicos de chunk (spans/span_types.py).

Fluxo:
    Markdown -> SpanParser -> ParsedDocument -> build_chunk_parts() -> ChunkPart[]

Garantias:
    - O parser é determinístico (regex-based)
    - O LLM nunca gera texto, só seleciona span_ids
    - O split físico mantém overlap entre partes
    - Neo4j edges são criados no nível lógico (por Span), não físico (por ChunkPart)
"""

from typing import Optional
import logging

from ..parsing.span_models import SpanType, Span, ParsedDocument
from ..spans.span_types import DeviceType, ChunkPart
from ..spans.splitter import split_text_with_offsets, MAX_TEXT_CHARS, OVERLAP_CHARS
from ..canonical.id_conventions import (
    build_logical_node_id,
    build_node_id,
    build_chunk_id,
    build_parent_chunk_id,
    get_prefix_for_document_type,
)

logger = logging.getLogger(__name__)


def map_span_type_to_device_type(span_type: SpanType) -> DeviceType:
    """
    Mapeia SpanType (parsing/) para DeviceType (spans/).

    Args:
        span_type: Tipo de span do parsing/span_models.py

    Returns:
        DeviceType correspondente

    Examples:
        >>> map_span_type_to_device_type(SpanType.ARTIGO)
        <DeviceType.ARTICLE: 'article'>
        >>> map_span_type_to_device_type(SpanType.INCISO)
        <DeviceType.INCISO: 'inciso'>
    """
    mapping = {
        SpanType.ARTIGO: DeviceType.ARTICLE,
        SpanType.PARAGRAFO: DeviceType.PARAGRAPH,
        SpanType.INCISO: DeviceType.INCISO,
        SpanType.ALINEA: DeviceType.ALINEA,
        SpanType.HEADER: DeviceType.EMENTA,
        SpanType.CAPITULO: DeviceType.UNKNOWN,
        SpanType.SECAO: DeviceType.UNKNOWN,
        SpanType.SUBSECAO: DeviceType.UNKNOWN,
        SpanType.TITULO: DeviceType.UNKNOWN,
        SpanType.TEXTO: DeviceType.UNKNOWN,
        SpanType.ASSINATURA: DeviceType.UNKNOWN,
        SpanType.ITEM: DeviceType.ALINEA,  # Item é similar a alínea
    }
    return mapping.get(span_type, DeviceType.UNKNOWN)


def find_root_article_span_id(span: Span, parsed_doc: ParsedDocument) -> Optional[str]:
    """
    Encontra o span_id do artigo raiz de um span.

    Navega pela hierarquia até encontrar um artigo.
    Se o span já é um artigo, retorna seu próprio span_id.
    Se não encontrar artigo na hierarquia, retorna None.

    Args:
        span: Span cujo artigo raiz queremos encontrar
        parsed_doc: Documento parseado com índice de spans

    Returns:
        span_id do artigo raiz ou None

    Examples:
        >>> # Para um inciso INC-001-I com parent_id="ART-001"
        >>> find_root_article_span_id(inciso, doc)
        'ART-001'
        >>> # Para um artigo
        >>> find_root_article_span_id(artigo, doc)
        'ART-001'
    """
    # Se o span é um artigo, retorna ele mesmo
    if span.span_type == SpanType.ARTIGO:
        return span.span_id

    # Navega pela hierarquia
    current = span
    visited = set()  # Evita loops infinitos

    while current.parent_id and current.parent_id not in visited:
        visited.add(current.parent_id)
        parent = parsed_doc.get_span(current.parent_id)

        if parent is None:
            break

        if parent.span_type == SpanType.ARTIGO:
            return parent.span_id

        current = parent

    # Fallback: tenta extrair do span_id se segue o padrão
    # Ex: INC-005-I -> ART-005, PAR-003-1 -> ART-003
    span_id = span.span_id
    if "-" in span_id:
        parts = span_id.split("-")
        if len(parts) >= 2:
            article_num = parts[1]  # O segundo segmento é o número do artigo
            candidate = f"ART-{article_num}"
            if parsed_doc.get_span(candidate):
                return candidate

    return None


def build_chunk_parts(
    parsed_doc: ParsedDocument,
    document_id: str,
    document_type: str,
    prefix: Optional[str] = None,
) -> list[ChunkPart]:
    """
    Converte ParsedDocument em lista de ChunkParts.

    Para cada Span do ParsedDocument:
    1. Constrói o logical_node_id (ex: leis:LEI-14133-2021#ART-005)
    2. Se o texto for grande, divide em partes com overlap
    3. Cria ChunkParts com IDs físicos (node_id@P00, P01, etc.)

    Args:
        parsed_doc: Documento parseado pelo SpanParser
        document_id: ID do documento (ex: LEI-14133-2021)
        document_type: Tipo do documento (ex: LEI, DECRETO, IN)
        prefix: Prefixo para o namespace (ex: "leis"). Se None, usa get_prefix_for_document_type

    Returns:
        Lista de ChunkParts prontos para indexação

    Examples:
        >>> from parsing import SpanParser
        >>> parser = SpanParser()
        >>> doc = parser.parse(markdown_text)
        >>> chunks = build_chunk_parts(doc, "LEI-14133-2021", "LEI")
        >>> print(f"Total de chunks: {len(chunks)}")
    """
    if prefix is None:
        prefix = get_prefix_for_document_type(document_type)

    chunk_parts: list[ChunkPart] = []

    for span in parsed_doc.spans:
        # Só processa spans que são dispositivos legais
        device_type = map_span_type_to_device_type(span.span_type)
        if device_type == DeviceType.UNKNOWN:
            logger.debug(f"Ignorando span {span.span_id} (tipo {span.span_type})")
            continue

        # Constrói logical_node_id
        logical_node_id = build_logical_node_id(prefix, document_id, span.span_id)

        # Encontra artigo raiz e extrai número
        article_span_id = find_root_article_span_id(span, parsed_doc)
        article_number = None
        if article_span_id and article_span_id.startswith("ART-"):
            article_number = article_span_id.replace("ART-", "")

        # Divide texto se necessário
        text_parts = split_text_with_offsets(span.text, MAX_TEXT_CHARS, OVERLAP_CHARS)
        part_total = len(text_parts)

        for part_index, (part_text, char_start, char_end) in enumerate(text_parts):
            # Constrói IDs físicos
            node_id = build_node_id(logical_node_id, part_index)
            chunk_id = build_chunk_id(document_id, span.span_id, part_index)
            parent_chunk_id = build_parent_chunk_id(document_id, span.parent_id)

            # Cria ChunkPart
            chunk_part = ChunkPart(
                node_id=node_id,
                logical_node_id=logical_node_id,
                chunk_id=chunk_id,
                parent_chunk_id=parent_chunk_id,
                part_index=part_index,
                part_total=part_total,
                text=part_text,
                char_start=char_start,
                char_end=char_end,
                document_id=document_id,
                span_id=span.span_id,
                device_type=device_type,
                article_number=article_number,
                document_type=document_type,
            )
            chunk_parts.append(chunk_part)

    logger.info(
        f"Convertidos {len(parsed_doc.spans)} spans em {len(chunk_parts)} ChunkParts "
        f"para documento {document_id}"
    )

    return chunk_parts


class ParsedDocumentChunkPartsBuilder:
    """
    Builder que encapsula a conversão ParsedDocument -> ChunkPart[].

    Uso:
        builder = ParsedDocumentChunkPartsBuilder(
            document_id="LEI-14133-2021",
            document_type="LEI",
        )
        chunks = builder.build(parsed_doc)
    """

    def __init__(
        self,
        document_id: str,
        document_type: str,
        prefix: Optional[str] = None,
    ):
        """
        Inicializa o builder.

        Args:
            document_id: ID do documento (ex: LEI-14133-2021)
            document_type: Tipo do documento (ex: LEI, DECRETO, IN)
            prefix: Prefixo opcional para namespace
        """
        self.document_id = document_id
        self.document_type = document_type
        self.prefix = prefix or get_prefix_for_document_type(document_type)

    def build(self, parsed_doc: ParsedDocument) -> list[ChunkPart]:
        """
        Converte ParsedDocument em lista de ChunkParts.

        Args:
            parsed_doc: Documento parseado pelo SpanParser

        Returns:
            Lista de ChunkParts
        """
        return build_chunk_parts(
            parsed_doc=parsed_doc,
            document_id=self.document_id,
            document_type=self.document_type,
            prefix=self.prefix,
        )

    def build_from_spans(self, spans: list[Span]) -> list[ChunkPart]:
        """
        Converte lista de Spans em ChunkParts.

        Conveniência para quando já se tem os spans selecionados.

        Args:
            spans: Lista de spans a converter

        Returns:
            Lista de ChunkParts
        """
        # Cria ParsedDocument temporário
        doc = ParsedDocument(spans=spans)
        return self.build(doc)
