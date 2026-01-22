"""
Span Models - Estruturas de Dados para Spans de Documentos Legais.

Este módulo define as estruturas de dados fundamentais do sistema de parsing:
SpanType, Span e ParsedDocument. Cada span representa uma unidade atômica
do documento que pode ser referenciada por um ID único.

Princípio Anti-Alucinação:
=========================

    O LLM NUNCA gera texto - apenas seleciona span_ids.
    O texto é reconstruído de forma 100% determinística pelo código.

    ┌─────────────────────────────────────────────────────────────────────┐
    │                        FLUXO DE DADOS                               │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  SpanParser                  LLM                  ParsedDocument    │
    │  ┌─────────────┐       ┌──────────────┐       ┌─────────────────┐  │
    │  │  Regex      │──────▶│  Seleciona   │──────▶│  Reconstrói     │  │
    │  │  Patterns   │       │  span_ids    │       │  Texto          │  │
    │  └─────────────┘       └──────────────┘       └─────────────────┘  │
    │       │                      │                       │             │
    │       ▼                      ▼                       ▼             │
    │   Gera Spans           Lista de IDs            Texto Final         │
    │   com IDs únicos       ["ART-001",            (determinístico)     │
    │                         "INC-001-I"]                               │
    └─────────────────────────────────────────────────────────────────────┘

Hierarquia de SpanTypes:
=======================

    SpanType define os tipos de elementos em documentos legais brasileiros:

    ┌──────────────────────────────────────────────────────────────────┐
    │                    HIERARQUIA DE TIPOS                           │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  HEADER ─────────► Cabeçalho (ementa, órgão emissor)             │
    │      │                                                           │
    │      ▼                                                           │
    │  CAPITULO ───────► CAPÍTULO I, II, III...                        │
    │      │                                                           │
    │      ├── SECAO ──► Seção I, II, III...                           │
    │      │     │                                                     │
    │      │     └── SUBSECAO ──► Subseção (raro)                      │
    │      │                                                           │
    │      └── ARTIGO ─► Art. 1º, Art. 2º...                           │
    │            │                                                     │
    │            ├── PARAGRAFO ──► § 1º, § 2º, § único                 │
    │            │                                                     │
    │            └── INCISO ──────► I, II, III, IV...                  │
    │                  │                                               │
    │                  └── ALINEA ──► a), b), c)...                    │
    │                        │                                         │
    │                        └── ITEM ──► 1), 2), 3)... (raro)         │
    │                                                                  │
    │  ASSINATURA ────► Assinatura ao final do documento               │
    └──────────────────────────────────────────────────────────────────┘

Formato de Span IDs:
===================

    Cada tipo de span tem um formato de ID específico:

    | Tipo      | Formato              | Exemplos                       |
    |-----------|----------------------|--------------------------------|
    | Header    | HDR-{seq}            | HDR-001, HDR-002               |
    | Capítulo  | CAP-{romano}         | CAP-I, CAP-II, CAP-III         |
    | Artigo    | ART-{nnn}            | ART-001, ART-012, ART-123      |
    | Parágrafo | PAR-{art}-{n}        | PAR-001-1, PAR-005-UNICO       |
    | Inciso    | INC-{art}-{romano}   | INC-001-I, INC-005-IV          |
    | Alínea    | ALI-{art}-{inc}-{l}  | ALI-001-I-a, ALI-005-II-b      |

    Nota sobre Incisos em Parágrafos:
    --------------------------------
    Incisos dentro de parágrafos mantêm o formato INC-{art}-{romano}.
    O vínculo ao parágrafo fica em parent_id (ex: parent_id="PAR-001-2").

    Quando há conflito (mesmo romano em contextos diferentes):
    - INC-001-I    → Primeiro inciso I (do caput)
    - INC-001-I_2  → Segundo inciso I (do §2º)

Estrutura do Span:
=================

    @dataclass
    class Span:
        span_id: str           # ID único (ex: "ART-001")
        span_type: SpanType    # Tipo do span
        text: str              # Texto original
        identifier: str        # Identificador legal ("1º", "I", "a")
        parent_id: str         # ID do pai (ex: "ART-001" para incisos)
        start_pos: int         # Posição inicial no texto fonte
        end_pos: int           # Posição final no texto fonte
        order: int             # Ordem de inserção (para ordenação)
        metadata: dict         # Dados extras (título, contexto)

    Propriedades úteis:
    - is_article: True se for artigo
    - is_paragraph: True se for parágrafo
    - is_inciso: True se for inciso
    - is_alinea: True se for alínea
    - article_number: Extrai número do artigo do span_id

Estrutura do ParsedDocument:
===========================

    ParsedDocument é o container principal que armazena todos os spans
    de um documento parseado, com índice para lookup O(1).

    ┌─────────────────────────────────────────────────────────────────┐
    │                      ParsedDocument                             │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  spans: list[Span]                                              │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │ [Span(HDR-001), Span(CAP-I), Span(ART-001), ...]         │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                 │
    │  _span_index: dict[str, Span]  (índice interno)                │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │ {"ART-001": Span, "INC-001-I": Span, ...}                │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                 │
    │  Métodos principais:                                            │
    │  - add_span(span)          → Adiciona span e atualiza índice   │
    │  - get_span(span_id)       → Lookup O(1) por ID                │
    │  - get_children(parent_id) → Filhos de um span                 │
    │  - reconstruct_text(ids)   → Reconstrói texto de span_ids      │
    │  - to_annotated_markdown() → Markdown com [SPAN_ID] prefixos   │
    │                                                                 │
    │  Propriedades:                                                  │
    │  - articles: Lista de todos os artigos                         │
    │  - capitulos: Lista de todos os capítulos                      │
    └─────────────────────────────────────────────────────────────────┘

Exemplo de Uso:
==============

    ```python
    from parsing.span_models import SpanType, Span, ParsedDocument

    # Criar documento
    doc = ParsedDocument()

    # Adicionar spans
    doc.add_span(Span(
        span_id="ART-001",
        span_type=SpanType.ARTIGO,
        text="Art. 1º Esta Lei estabelece normas gerais...",
        identifier="1º",
    ))

    doc.add_span(Span(
        span_id="INC-001-I",
        span_type=SpanType.INCISO,
        text="I - princípio da legalidade;",
        identifier="I",
        parent_id="ART-001",
    ))

    # Lookup por ID
    artigo = doc.get_span("ART-001")
    print(artigo.text)

    # Filhos de um span
    incisos = doc.get_children("ART-001")
    for inc in incisos:
        print(f"  {inc.span_id}: {inc.text[:50]}...")

    # Reconstruir texto de IDs (chave do sistema!)
    ids = ["ART-001", "INC-001-I", "INC-001-II"]
    texto = doc.reconstruct_text(ids)

    # Gerar markdown anotado para o LLM
    annotated = doc.to_annotated_markdown()
    # "[ART-001] Art. 1º Esta Lei estabelece..."
    # "[INC-001-I] I - princípio da legalidade;"
    ```

Métodos de Validação:
====================

    O ParsedDocument oferece validação para garantir integridade:

    ```python
    # Validar se todos os IDs existem
    ids = ["ART-001", "ART-999"]  # ART-999 não existe
    valido, invalidos = doc.validate_span_ids(ids)

    if not valido:
        print(f"IDs inválidos: {invalidos}")  # ["ART-999"]
    ```

Módulos Relacionados:
====================

    - parsing/span_parser.py: Gera ParsedDocument via regex
    - parsing/article_orchestrator.py: Usa spans para extração LLM
    - chunking/chunk_materializer.py: Converte spans em chunks indexáveis
    - rag/answer_models.py: Referencia span_ids nas citações

@author: Equipe VectorGov
@version: 1.0.0
@since: 23/12/2024
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SpanType(str, Enum):
    """Tipos de spans em documentos legais brasileiros."""

    # Estrutura principal
    HEADER = "header"           # Cabeçalho do documento (ementa, órgão, etc.)
    CAPITULO = "capitulo"       # CAPÍTULO I, II, III...
    SECAO = "secao"             # Seção I, II, III...
    SUBSECAO = "subsecao"       # Subseção

    # Artigos e subdivisões
    ARTIGO = "artigo"           # Art. 1º, Art. 2º...
    PARAGRAFO = "paragrafo"     # § 1º, § 2º, Parágrafo único
    INCISO = "inciso"           # I, II, III, IV...
    ALINEA = "alinea"           # a), b), c)...
    ITEM = "item"               # 1), 2), 3)... (raro, mas existe)

    # Outros
    TITULO = "titulo"           # Título de artigo/seção
    TEXTO = "texto"             # Texto livre (entre estruturas)
    ASSINATURA = "assinatura"   # Assinatura do documento


@dataclass
class Span:
    """
    Representa um trecho atômico do documento.

    Attributes:
        span_id: Identificador único (ex: ART-001, INC-001-I)
        span_type: Tipo do span (artigo, inciso, etc.)
        text: Texto original do span
        identifier: Identificador legal (ex: "1º", "I", "a")
        parent_id: ID do span pai (ex: inciso aponta para artigo)
        start_pos: Posição inicial no texto original
        end_pos: Posição final no texto original
        metadata: Dados extras (título, contexto, etc.)
    """

    span_id: str
    span_type: SpanType
    text: str
    identifier: Optional[str] = None
    parent_id: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0
    order: int = 0  # Ordem de inserção (para ordenação estável)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Normaliza o texto."""
        self.text = self.text.strip()

    @property
    def is_article(self) -> bool:
        return self.span_type == SpanType.ARTIGO

    @property
    def is_paragraph(self) -> bool:
        return self.span_type == SpanType.PARAGRAFO

    @property
    def is_inciso(self) -> bool:
        return self.span_type == SpanType.INCISO

    @property
    def is_alinea(self) -> bool:
        return self.span_type == SpanType.ALINEA

    @property
    def article_number(self) -> Optional[str]:
        """Extrai número do artigo do span_id."""
        if self.span_id.startswith("ART-"):
            return self.span_id.split("-")[1]
        elif "-" in self.span_id:
            # PAR-001-1, INC-001-I, ALI-001-I-a
            parts = self.span_id.split("-")
            if len(parts) >= 2:
                return parts[1]
        return None

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "span_id": self.span_id,
            "span_type": self.span_type.value,
            "text": self.text,
            "identifier": self.identifier,
            "parent_id": self.parent_id,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "order": self.order,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Span({self.span_id}, {self.span_type.value}, '{text_preview}')"


@dataclass
class ParsedDocument:
    """
    Documento parseado com todos os spans identificados.

    Attributes:
        spans: Lista de todos os spans do documento
        span_index: Índice span_id -> Span para lookup rápido
        source_text: Texto original do documento
        metadata: Metadados do documento (tipo, número, data, etc.)
    """

    spans: list[Span] = field(default_factory=list)
    source_text: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Constrói índice de spans."""
        self._span_index: dict[str, Span] = {}
        self._rebuild_index()

    def _rebuild_index(self):
        """Reconstrói o índice de spans."""
        self._span_index = {span.span_id: span for span in self.spans}

    def add_span(self, span: Span):
        """Adiciona um span ao documento."""
        span.order = len(self.spans)  # Define ordem de inserção
        self.spans.append(span)
        self._span_index[span.span_id] = span

    def get_span(self, span_id: str) -> Optional[Span]:
        """Busca span por ID."""
        return self._span_index.get(span_id)

    def get_spans_by_type(self, span_type: SpanType) -> list[Span]:
        """Retorna todos os spans de um tipo, ordenados por ordem de inserção."""
        spans = [s for s in self.spans if s.span_type == span_type]
        return sorted(spans, key=lambda s: s.order)

    def get_children(self, parent_id: str) -> list[Span]:
        """Retorna filhos de um span, ordenados por ordem no documento."""
        children = [s for s in self.spans if s.parent_id == parent_id]
        return sorted(children, key=lambda s: s.order)

    def get_article_spans(self, article_number: str) -> list[Span]:
        """Retorna todos os spans de um artigo (incluindo filhos)."""
        art_id = f"ART-{article_number.zfill(3)}"
        result = []

        # Artigo principal
        if art_id in self._span_index:
            result.append(self._span_index[art_id])

        # Filhos diretos e indiretos
        for span in self.spans:
            if span.parent_id == art_id:
                result.append(span)
            elif span.parent_id and span.parent_id.startswith(f"INC-{article_number.zfill(3)}"):
                result.append(span)
            elif span.parent_id and span.parent_id.startswith(f"PAR-{article_number.zfill(3)}"):
                result.append(span)

        return result

    def reconstruct_text(self, span_ids: list[str]) -> str:
        """
        Reconstrói texto a partir de lista de span_ids.

        Esta é a função chave: o LLM retorna IDs, e o código
        reconstrói o texto de forma determinística.
        """
        texts = []
        for span_id in span_ids:
            span = self.get_span(span_id)
            if span:
                texts.append(span.text)
        return "\n".join(texts)

    def validate_span_ids(self, span_ids: list[str]) -> tuple[bool, list[str]]:
        """
        Valida se todos os span_ids existem.

        Returns:
            (válido, lista de IDs inválidos)
        """
        invalid = [sid for sid in span_ids if sid not in self._span_index]
        return len(invalid) == 0, invalid

    @property
    def articles(self) -> list[Span]:
        """Retorna todos os artigos."""
        return self.get_spans_by_type(SpanType.ARTIGO)

    @property
    def capitulos(self) -> list[Span]:
        """Retorna todos os capítulos."""
        return self.get_spans_by_type(SpanType.CAPITULO)

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "metadata": self.metadata,
            "span_count": len(self.spans),
            "article_count": len(self.articles),
            "spans": [s.to_dict() for s in self.spans],
        }

    def to_annotated_markdown(self) -> str:
        """
        Gera markdown anotado com span_ids.

        Este é o formato que será enviado ao LLM.
        """
        lines = []
        for span in self.spans:
            lines.append(f"[{span.span_id}] {span.text}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ParsedDocument("
            f"spans={len(self.spans)}, "
            f"articles={len(self.articles)}, "
            f"capitulos={len(self.capitulos)})"
        )
