"""
Construtor de texto para retrieval (embeddings).

PR3 v2 - Hard Reset RAG Architecture
PR3 v2.1 - Rebase: Adicionado suporte para ParsedDocument

Constrói o campo `retrieval_text` que será usado para gerar embeddings.
Também resolve `parent_text` para spans filhos.

IMPORTANTE:
- `text`: Fonte de verdade, usado pelo LLM
- `retrieval_text`: Usado APENAS para embeddings e busca
- `parent_text`: Contexto do parent para melhorar retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from ..spans.span_types import Span, ChunkPart, DeviceType

# PR3 v2.1 - Imports para ParsedDocument (apenas para type checking)
if TYPE_CHECKING:
    from ..parsing.span_models import Span as ParsingSpan, ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Contexto para construção do retrieval_text."""

    retrieval_text: str
    parent_text: Optional[str]
    context_header: str


class ParentTextResolver:
    """
    Resolve o texto do parent para spans filhos.

    Para melhorar a qualidade do retrieval, spans filhos (parágrafos, incisos)
    incluem o texto do artigo pai como contexto.
    """

    def __init__(self, spans: list[Span]):
        """
        Inicializa o resolver com a lista de spans.

        Args:
            spans: Lista de todos os spans do documento
        """
        # Indexa spans por span_id para busca rápida
        self._span_index: dict[str, Span] = {}
        for span in spans:
            self._span_index[span.span_id] = span

        # Cache de parent_text já resolvidos
        self._parent_text_cache: dict[str, str] = {}

    def resolve_parent_text(self, span: Span) -> Optional[str]:
        """
        Resolve o texto do parent de um span.

        Args:
            span: Span para resolver o parent

        Returns:
            Texto do parent ou None se não houver parent
        """
        if span.parent_span_id is None:
            return None

        # Verifica cache
        if span.parent_span_id in self._parent_text_cache:
            return self._parent_text_cache[span.parent_span_id]

        # Busca parent no índice
        parent = self._span_index.get(span.parent_span_id)
        if parent is None:
            logger.warning(
                f"Parent {span.parent_span_id} não encontrado para span {span.span_id}"
            )
            return None

        # Limita o texto do parent para não ficar muito grande
        parent_text = self._truncate_parent_text(parent.text, max_chars=500)

        # Cache
        self._parent_text_cache[span.parent_span_id] = parent_text

        return parent_text

    def _truncate_parent_text(self, text: str, max_chars: int = 500) -> str:
        """
        Trunca o texto do parent se for muito grande.

        Args:
            text: Texto original
            max_chars: Máximo de caracteres

        Returns:
            Texto truncado
        """
        if len(text) <= max_chars:
            return text

        # Tenta truncar em um espaço
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")

        if last_space > max_chars // 2:
            truncated = truncated[:last_space]

        return truncated + "..."


class RetrievalTextBuilder:
    """
    Constrói o retrieval_text para embeddings.

    O retrieval_text é uma versão otimizada para busca semântica,
    incluindo contexto do parent e headers descritivos.
    """

    # Templates de context_header por tipo de dispositivo
    CONTEXT_TEMPLATES = {
        DeviceType.ARTICLE: "{doc_type} {doc_id}, Art. {article_num}",
        DeviceType.PARAGRAPH: "{doc_type} {doc_id}, Art. {article_num}, {para_id}",
        DeviceType.INCISO: "{doc_type} {doc_id}, Art. {article_num}, inciso {inciso_id}",
        DeviceType.ALINEA: "{doc_type} {doc_id}, Art. {article_num}, alínea {alinea_id}",
        DeviceType.CAPUT: "{doc_type} {doc_id}, Art. {article_num}, caput",
        DeviceType.EMENTA: "{doc_type} {doc_id}, Ementa",
        DeviceType.PREAMBULO: "{doc_type} {doc_id}, Preâmbulo",
        DeviceType.UNKNOWN: "{doc_type} {doc_id}",
    }

    def __init__(
        self,
        spans: list[Span],
        document_type: str = "LEI",
        document_id: str = "",
    ):
        """
        Inicializa o builder.

        Args:
            spans: Lista de spans do documento
            document_type: Tipo do documento
            document_id: ID do documento
        """
        self.spans = spans
        self.document_type = document_type
        self.document_id = document_id
        self.parent_resolver = ParentTextResolver(spans)

    def build(self, span: Span) -> RetrievalContext:
        """
        Constrói o RetrievalContext para um span.

        Args:
            span: Span para construir contexto

        Returns:
            RetrievalContext com retrieval_text, parent_text e context_header
        """
        # Resolve parent_text
        parent_text = self.parent_resolver.resolve_parent_text(span)

        # Constrói context_header
        context_header = self._build_context_header(span)

        # Constrói retrieval_text
        retrieval_text = self._build_retrieval_text(
            span=span,
            context_header=context_header,
            parent_text=parent_text,
        )

        return RetrievalContext(
            retrieval_text=retrieval_text,
            parent_text=parent_text,
            context_header=context_header,
        )

    def _build_context_header(self, span: Span) -> str:
        """
        Constrói o header de contexto para um span.

        Ex: "Lei 14133/2021, Art. 5, § 1º"
        """
        template = self.CONTEXT_TEMPLATES.get(
            span.device_type, self.CONTEXT_TEMPLATES[DeviceType.UNKNOWN]
        )

        # Extrai identificadores do span_id
        para_id = self._extract_paragraph_id(span.span_id)
        inciso_id = self._extract_inciso_id(span.span_id)
        alinea_id = self._extract_alinea_id(span.span_id)

        return template.format(
            doc_type=self.document_type,
            doc_id=self.document_id,
            article_num=span.article_number or "?",
            para_id=para_id,
            inciso_id=inciso_id,
            alinea_id=alinea_id,
        )

    def _build_retrieval_text(
        self,
        span: Span,
        context_header: str,
        parent_text: Optional[str],
    ) -> str:
        """
        Constrói o retrieval_text completo.

        Formato:
        [CONTEXTO: {context_header}]
        [PARENT: {parent_text}]  (se houver)
        {texto do span}
        """
        parts = []

        # Header de contexto
        parts.append(f"[CONTEXTO: {context_header}]")

        # Texto do parent (se houver e não for artigo)
        if parent_text and span.device_type != DeviceType.ARTICLE:
            parts.append(f"[PARENT: {parent_text}]")

        # Texto principal
        parts.append(span.text)

        return "\n\n".join(parts)

    def _extract_paragraph_id(self, span_id: str) -> str:
        """Extrai identificador do parágrafo do span_id."""
        # PAR-005-1 -> § 1º
        # PAR-005-UNICO -> Parágrafo único
        if "PAR-" not in span_id:
            return ""

        parts = span_id.split("-")
        if len(parts) >= 3:
            para_num = parts[-1]
            if para_num.upper() == "UNICO":
                return "Parágrafo único"
            return f"§ {para_num}º"

        return ""

    def _extract_inciso_id(self, span_id: str) -> str:
        """Extrai identificador do inciso do span_id."""
        # INC-005-I -> I
        # INC-005-II_1 -> II (do § 1º)
        if "INC-" not in span_id:
            return ""

        parts = span_id.split("-")
        if len(parts) >= 3:
            inciso_part = parts[-1]
            # Remove sufixo de parágrafo se houver
            if "_" in inciso_part:
                inciso_part = inciso_part.split("_")[0]
            return inciso_part

        return ""

    def _extract_alinea_id(self, span_id: str) -> str:
        """Extrai identificador da alínea do span_id."""
        # ALI-005-I-a -> a
        if "ALI-" not in span_id:
            return ""

        parts = span_id.split("-")
        if len(parts) >= 4:
            return parts[-1]

        return ""

    def build_all(self) -> dict[str, RetrievalContext]:
        """
        Constrói RetrievalContext para todos os spans.

        Returns:
            Dicionário span_id -> RetrievalContext
        """
        result = {}
        for span in self.spans:
            result[span.span_id] = self.build(span)
        return result


def build_retrieval_text(
    span: Span,
    parent_text: Optional[str] = None,
    document_type: str = "LEI",
    document_id: str = "",
) -> str:
    """
    Função de conveniência para construir retrieval_text de um span.

    Args:
        span: Span para construir
        parent_text: Texto do parent (opcional)
        document_type: Tipo do documento
        document_id: ID do documento

    Returns:
        String do retrieval_text
    """
    builder = RetrievalTextBuilder(
        spans=[span],
        document_type=document_type,
        document_id=document_id,
    )

    # Cria um resolver mock se parent_text foi fornecido
    if parent_text:
        builder.parent_resolver._parent_text_cache[span.parent_span_id or ""] = parent_text

    context = builder.build(span)
    return context.retrieval_text


# =============================================================================
# PR3 v2.1 - Suporte para ParsedDocument (parsing/span_models.py)
# =============================================================================


class ParentTextResolverFromParsedDocument:
    """
    Resolve o texto do parent usando ParsedDocument.

    PR3 v2.1: Versão que trabalha com parsing.ParsedDocument em vez de list[Span].
    """

    def __init__(self, parsed_doc: "ParsedDocument"):
        """
        Inicializa o resolver com ParsedDocument.

        Args:
            parsed_doc: Documento parseado com índice de spans
        """
        self._parsed_doc = parsed_doc
        self._parent_text_cache: dict[str, str] = {}

    def resolve_parent_text(self, span: "ParsingSpan") -> Optional[str]:
        """
        Resolve o texto do parent de um span.

        Args:
            span: Span do parsing module

        Returns:
            Texto do parent ou None se não houver parent
        """
        if span.parent_id is None:
            return None

        # Verifica cache
        if span.parent_id in self._parent_text_cache:
            return self._parent_text_cache[span.parent_id]

        # Busca parent no ParsedDocument (O(1) lookup)
        parent = self._parsed_doc.get_span(span.parent_id)
        if parent is None:
            logger.warning(
                f"Parent {span.parent_id} não encontrado para span {span.span_id}"
            )
            return None

        # Limita o texto do parent para não ficar muito grande
        parent_text = self._truncate_parent_text(parent.text, max_chars=500)

        # Cache
        self._parent_text_cache[span.parent_id] = parent_text

        return parent_text

    def _truncate_parent_text(self, text: str, max_chars: int = 500) -> str:
        """Trunca o texto do parent se for muito grande."""
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")

        if last_space > max_chars // 2:
            truncated = truncated[:last_space]

        return truncated + "..."


class RetrievalTextBuilderFromParsedDocument:
    """
    Constrói retrieval_text para embeddings usando ParsedDocument.

    PR3 v2.1: Versão que trabalha com parsing.ParsedDocument.

    O retrieval_text é uma versão otimizada para busca semântica,
    incluindo contexto do parent e headers descritivos.
    """

    # Templates de context_header por SpanType
    CONTEXT_TEMPLATES = {
        "artigo": "{doc_type} {doc_id}, Art. {article_num}",
        "paragrafo": "{doc_type} {doc_id}, Art. {article_num}, {para_id}",
        "inciso": "{doc_type} {doc_id}, Art. {article_num}, inciso {inciso_id}",
        "alinea": "{doc_type} {doc_id}, Art. {article_num}, alínea {alinea_id}",
        "capitulo": "{doc_type} {doc_id}, Capítulo {cap_id}",
        "secao": "{doc_type} {doc_id}, Seção {sec_id}",
        "header": "{doc_type} {doc_id}, Cabeçalho",
        "default": "{doc_type} {doc_id}",
    }

    def __init__(
        self,
        parsed_doc: "ParsedDocument",
        document_type: str = "LEI",
        document_id: str = "",
    ):
        """
        Inicializa o builder.

        Args:
            parsed_doc: Documento parseado
            document_type: Tipo do documento
            document_id: ID do documento
        """
        self.parsed_doc = parsed_doc
        self.document_type = document_type
        self.document_id = document_id
        self.parent_resolver = ParentTextResolverFromParsedDocument(parsed_doc)

    def build(self, span: "ParsingSpan") -> RetrievalContext:
        """
        Constrói o RetrievalContext para um span.

        Args:
            span: Span do parsing module

        Returns:
            RetrievalContext com retrieval_text, parent_text e context_header
        """
        # Resolve parent_text
        parent_text = self.parent_resolver.resolve_parent_text(span)

        # Constrói context_header
        context_header = self._build_context_header(span)

        # Constrói retrieval_text
        retrieval_text = self._build_retrieval_text(
            span=span,
            context_header=context_header,
            parent_text=parent_text,
        )

        return RetrievalContext(
            retrieval_text=retrieval_text,
            parent_text=parent_text,
            context_header=context_header,
        )

    def _build_context_header(self, span: "ParsingSpan") -> str:
        """Constrói o header de contexto para um span."""
        span_type = span.span_type.value if hasattr(span.span_type, 'value') else str(span.span_type)
        template = self.CONTEXT_TEMPLATES.get(
            span_type, self.CONTEXT_TEMPLATES["default"]
        )

        # Extrai identificadores do span_id
        para_id = self._extract_paragraph_id(span.span_id)
        inciso_id = self._extract_inciso_id(span.span_id)
        alinea_id = self._extract_alinea_id(span.span_id)
        cap_id = self._extract_capitulo_id(span.span_id)
        sec_id = self._extract_secao_id(span.span_id)

        return template.format(
            doc_type=self.document_type,
            doc_id=self.document_id,
            article_num=span.article_number or "?",
            para_id=para_id,
            inciso_id=inciso_id,
            alinea_id=alinea_id,
            cap_id=cap_id,
            sec_id=sec_id,
        )

    def _build_retrieval_text(
        self,
        span: "ParsingSpan",
        context_header: str,
        parent_text: Optional[str],
    ) -> str:
        """Constrói o retrieval_text completo."""
        parts = []

        # Header de contexto
        parts.append(f"[CONTEXTO: {context_header}]")

        # Texto do parent (se houver e não for artigo/capítulo)
        is_root_type = span.span_type.value in ("artigo", "capitulo", "secao", "header")
        if parent_text and not is_root_type:
            parts.append(f"[PARENT: {parent_text}]")

        # Texto principal
        parts.append(span.text)

        return "\n\n".join(parts)

    def _extract_paragraph_id(self, span_id: str) -> str:
        """Extrai identificador do parágrafo do span_id."""
        if "PAR-" not in span_id:
            return ""

        parts = span_id.split("-")
        if len(parts) >= 3:
            para_num = parts[-1]
            if para_num.upper() == "UNICO":
                return "Parágrafo único"
            return f"§ {para_num}º"

        return ""

    def _extract_inciso_id(self, span_id: str) -> str:
        """Extrai identificador do inciso do span_id."""
        if "INC-" not in span_id:
            return ""

        parts = span_id.split("-")
        if len(parts) >= 3:
            inciso_part = parts[-1]
            if "_" in inciso_part:
                inciso_part = inciso_part.split("_")[0]
            return inciso_part

        return ""

    def _extract_alinea_id(self, span_id: str) -> str:
        """Extrai identificador da alínea do span_id."""
        if "ALI-" not in span_id:
            return ""

        parts = span_id.split("-")
        if len(parts) >= 4:
            return parts[-1]

        return ""

    def _extract_capitulo_id(self, span_id: str) -> str:
        """Extrai identificador do capítulo do span_id."""
        if "CAP-" not in span_id:
            return ""
        # CAP-I, CAP-II, etc.
        return span_id.replace("CAP-", "")

    def _extract_secao_id(self, span_id: str) -> str:
        """Extrai identificador da seção do span_id."""
        if "SEC-" not in span_id:
            return ""
        return span_id.replace("SEC-", "")

    def build_all(self) -> dict[str, RetrievalContext]:
        """
        Constrói RetrievalContext para todos os spans.

        Returns:
            Dicionário span_id -> RetrievalContext
        """
        result = {}
        for span in self.parsed_doc.spans:
            result[span.span_id] = self.build(span)
        return result
