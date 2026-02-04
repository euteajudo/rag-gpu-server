"""
Span Extraction Models - Schemas Pydantic para Extração Baseada em Spans.

Este módulo define os modelos de dados (Pydantic) que representam a estrutura
hierárquica de documentos legais brasileiros usando referências a spans.

Princípio Anti-Alucinação:
=========================

    O LLM NUNCA gera texto de artigos, parágrafos ou incisos.
    Ele apenas SELECIONA IDs de spans que EXISTEM no documento.

    ┌─────────────────────────────────────────────────────────────────┐
    │                  FLUXO DE EXTRAÇÃO                              │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Markdown Anotado                                               │
    │  [ART-001] Art. 1º Este documento estabelece...                 │
    │  [INC-001-I] I - as definições básicas;                         │
    │  [INC-001-II] II - os procedimentos;                            │
    │                                                                 │
    │           │                                                     │
    │           ▼                                                     │
    │                                                                 │
    │  LLM (Qwen 3 8B)                                                │
    │  "Quais IDs pertencem ao Art. 1?"                               │
    │  Resposta: {"article_id": "ART-001",                            │
    │             "inciso_ids": ["INC-001-I", "INC-001-II"]}          │
    │                                                                 │
    │           │                                                     │
    │           ▼                                                     │
    │                                                                 │
    │  Validação Pydantic                                             │
    │  ✓ ART-001 começa com "ART-"                                    │
    │  ✓ INC-001-I começa com "INC-"                                  │
    │  ✓ INC-001-II começa com "INC-"                                 │
    │                                                                 │
    │           │                                                     │
    │           ▼                                                     │
    │                                                                 │
    │  Texto reconstruído do ParsedDocument original                  │
    │  (nunca gerado pelo LLM)                                        │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Hierarquia de Modelos:
=====================

    SpanReference ─────────────► Referência simples a um span
         │
         ▼
    ArticleSpans ──────────────► Estrutura de um artigo
    │   - article_id: str           (ART-001)
    │   - inciso_ids: list[str]     (INC-001-I, INC-001-II)
    │   - paragrafo_ids: list[str]  (PAR-001-1, PAR-001-UNICO)
         │
         ▼
    ChapterSpans ──────────────► Estrutura de um capítulo
    │   - chapter_id: str           (CAP-I)
    │   - title: str                (DISPOSIÇÕES GERAIS)
    │   - article_ids: list[str]    (ART-001, ART-002)
         │
         ▼
    DocumentSpans ─────────────► Documento completo
    │   - document_type: str
    │   - number: str
    │   - date: str
    │   - issuing_body: str
    │   - ementa: str
    │   - chapters: list[ChapterSpans]
         │
         ▼
    SpanExtractionResult ──────► Resultado final com validação
        - document: DocumentSpans
        - classifications: list[SpanClassification]
        - valid_span_ids: list[str]
        - invalid_span_ids: list[str]

Formato de IDs por Tipo:
=======================

| Modelo        | Campo          | Prefixo | Formato                | Exemplo          |
|---------------|----------------|---------|------------------------|------------------|
| ChapterSpans  | chapter_id     | CAP-    | CAP-{romano}           | CAP-I, CAP-II    |
| ArticleSpans  | article_id     | ART-    | ART-{nnn}              | ART-001, ART-012 |
| ArticleSpans  | inciso_ids[]   | INC-    | INC-{art}-{romano}     | INC-001-I        |
| ArticleSpans  | paragrafo_ids[]| PAR-    | PAR-{art}-{n/UNICO}    | PAR-001-1        |

Validadores Pydantic:
====================

Cada campo de ID tem um validador que garante o prefixo correto:

| Validador                | Campo           | Prefixo Exigido |
|--------------------------|-----------------|-----------------|
| validate_article_id      | article_id      | ART-            |
| validate_inciso_ids      | inciso_ids[]    | INC-            |
| validate_paragrafo_ids   | paragrafo_ids[] | PAR-            |
| validate_chapter_id      | chapter_id      | CAP-            |

Se um ID inválido for retornado pelo LLM, o Pydantic levanta ValueError:

    >>> ArticleSpans(article_id="001", inciso_ids=[])
    ValidationError: article_id deve começar com 'ART-': 001

SpanClassification - Classificação de Conteúdo:
==============================================

Além da estrutura, o LLM pode classificar o tipo de conteúdo de cada span:

| content_type  | Descrição                           | Exemplo                        |
|---------------|-------------------------------------|--------------------------------|
| definicao     | Define conceitos ou termos          | "considera-se licitação..."    |
| procedimento  | Descreve processos ou etapas        | "o processo seguirá..."        |
| requisito     | Estabelece condições obrigatórias   | "é obrigatório..."             |
| excecao       | Casos especiais ou dispensas        | "exceto quando..."             |
| prazo         | Define prazos ou limites temporais  | "no prazo de 30 dias..."       |
| penalidade    | Sanções ou consequências            | "sujeito à multa..."           |

Prompts para Extração:
=====================

O módulo inclui prompts otimizados para extração:

| Prompt                         | Uso                                    |
|--------------------------------|----------------------------------------|
| SPAN_EXTRACTION_SYSTEM_PROMPT  | System prompt do LLM                   |
| SPAN_EXTRACTION_USER_PROMPT    | Template para markdown anotado         |

O system prompt enfatiza:
- NUNCA gere texto, apenas selecione IDs
- Use APENAS IDs que aparecem no documento
- Mantenha a ordem original
- Organize artigos dentro dos capítulos corretos

Exemplo de Uso:
==============

    from parsing.span_extraction_models import (
        ArticleSpans,
        DocumentSpans,
        SpanExtractionResult,
        SPAN_EXTRACTION_SYSTEM_PROMPT,
    )

    # 1. LLM retorna estrutura com IDs
    article = ArticleSpans(
        article_id="ART-001",
        inciso_ids=["INC-001-I", "INC-001-II", "INC-001-III"],
        paragrafo_ids=["PAR-001-1", "PAR-001-2"]
    )

    # 2. Validação automática pelo Pydantic
    try:
        invalid = ArticleSpans(
            article_id="1",  # Falta prefixo ART-
            inciso_ids=[]
        )
    except ValidationError as e:
        print(f"ID inválido: {e}")

    # 3. Documento completo
    doc = DocumentSpans(
        document_type="INSTRUÇÃO NORMATIVA",
        number="58",
        date="2022-05-09",
        issuing_body="SEGES/ME",
        ementa="Dispõe sobre ETP...",
        chapters=[
            ChapterSpans(
                chapter_id="CAP-I",
                title="DISPOSIÇÕES GERAIS",
                article_ids=["ART-001", "ART-002", "ART-003"]
            )
        ]
    )

    # 4. Resultado com validação
    result = SpanExtractionResult(
        document=doc,
        valid_span_ids=["ART-001", "INC-001-I", "INC-001-II"],
        invalid_span_ids=[]  # Nenhum ID inválido
    )

Benefícios da Arquitetura:
=========================

| Benefício              | Como é Alcançado                               |
|------------------------|------------------------------------------------|
| Zero alucinação        | LLM só seleciona, nunca gera texto             |
| Validação automática   | Pydantic valida prefixos de todos os IDs       |
| Rastreabilidade        | Cada pedaço de texto tem um span_id único      |
| Reprodutibilidade      | Mesmo markdown → mesmos spans → mesmo resultado|
| Detecção de erros      | IDs inválidos são capturados imediatamente     |

Integração com Outros Módulos:
=============================

- span_parser.py: Gera spans com IDs únicos (entrada)
- article_orchestrator.py: Usa ArticleSpans para extração por artigo
- chunk_materializer.py: Converte spans em chunks para indexação

@author: Equipe VectorGov
@version: 1.0.0
@since: 23/12/2024
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SpanReference(BaseModel):
    """Reference to a span by ID."""
    span_id: str = Field(..., description="ID do span (ex: ART-001, INC-001-I)")


class ArticleSpans(BaseModel):
    """Spans that compose an article."""

    article_id: str = Field(
        ...,
        description="ID do artigo (ex: ART-001)",
        examples=["ART-001", "ART-012"]
    )

    # Incisos diretos do artigo (do caput)
    inciso_ids: list[str] = Field(
        default_factory=list,
        description="IDs dos incisos do caput (ex: INC-001-I)",
        examples=[["INC-001-I", "INC-001-II", "INC-001-III"]]
    )

    # Parágrafos do artigo
    paragrafo_ids: list[str] = Field(
        default_factory=list,
        description="IDs dos parágrafos (ex: PAR-001-1)",
        examples=[["PAR-001-1", "PAR-001-2", "PAR-001-UNICO"]]
    )

    # Paginação para listas grandes (> 200 IDs)
    has_more: bool = Field(
        default=False,
        description="True se há mais IDs além dos retornados (paginação)"
    )

    next_cursor: Optional[str] = Field(
        default=None,
        description="Cursor para próxima página (último ID retornado)"
    )

    @field_validator('article_id')
    @classmethod
    def validate_article_id(cls, v: str) -> str:
        if not v.startswith("ART-"):
            raise ValueError(f"article_id deve começar com 'ART-': {v}")
        return v

    @field_validator('inciso_ids')
    @classmethod
    def validate_inciso_ids(cls, v: list[str]) -> list[str]:
        for inc_id in v:
            if not inc_id.startswith("INC-"):
                raise ValueError(f"inciso_id deve começar com 'INC-': {inc_id}")
        return v

    @field_validator('paragrafo_ids')
    @classmethod
    def validate_paragrafo_ids(cls, v: list[str]) -> list[str]:
        for par_id in v:
            if not par_id.startswith("PAR-"):
                raise ValueError(f"paragrafo_id deve começar com 'PAR-': {par_id}")
        return v


class ChapterSpans(BaseModel):
    """Spans that compose a chapter."""

    chapter_id: str = Field(
        ...,
        description="ID do capítulo (ex: CAP-I)",
        examples=["CAP-I", "CAP-II", "CAP-III"]
    )

    # Título do capítulo (extraído do texto do span)
    title: Optional[str] = Field(
        None,
        description="Título do capítulo (se existir)"
    )

    # Artigos do capítulo
    article_ids: list[str] = Field(
        default_factory=list,
        description="IDs dos artigos neste capítulo",
        examples=[["ART-001", "ART-002", "ART-003"]]
    )

    @field_validator('chapter_id')
    @classmethod
    def validate_chapter_id(cls, v: str) -> str:
        if not v.startswith("CAP-"):
            raise ValueError(f"chapter_id deve começar com 'CAP-': {v}")
        return v


class DocumentSpans(BaseModel):
    """
    Complete document structure using span references.

    The LLM fills this with span IDs from the annotated markdown.
    Text is reconstructed from ParsedDocument after validation.
    """

    # Metadados (podem ser extraídos do header ou informados)
    document_type: str = Field(
        ...,
        description="Tipo: LEI, DECRETO, INSTRUÇÃO NORMATIVA, PORTARIA, etc."
    )

    number: str = Field(
        ...,
        description="Número do documento"
    )

    date: str = Field(
        ...,
        description="Data no formato YYYY-MM-DD"
    )

    issuing_body: str = Field(
        ...,
        description="Órgão emissor"
    )

    ementa: str = Field(
        ...,
        description="Ementa/resumo do documento"
    )

    # Estrutura por capítulos
    chapters: list[ChapterSpans] = Field(
        ...,
        min_length=1,
        description="Lista de capítulos com seus artigos"
    )


class SpanClassification(BaseModel):
    """
    Classification result for a single span.

    Used when asking the LLM to classify what type of content
    each span contains (definition, procedure, exception, etc.)
    """

    span_id: str = Field(..., description="ID do span")

    content_type: str = Field(
        ...,
        description="Tipo: definicao, procedimento, requisito, excecao, prazo, penalidade"
    )

    summary: str = Field(
        ...,
        max_length=200,
        description="Resumo curto do conteúdo (max 200 chars)"
    )

    keywords: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Palavras-chave principais (max 5)"
    )


class SpanExtractionResult(BaseModel):
    """
    Complete extraction result with span-based structure.

    This is the final output after:
    1. SpanParser extracts spans from markdown
    2. LLM classifies and organizes spans
    3. Validation ensures all IDs exist
    4. Text is reconstructed from original document
    """

    # Estrutura do documento
    document: DocumentSpans

    # Classificações de cada artigo (opcional, para enriquecimento)
    classifications: list[SpanClassification] = Field(
        default_factory=list,
        description="Classificações dos artigos"
    )

    # Validação
    valid_span_ids: list[str] = Field(
        default_factory=list,
        description="IDs validados como existentes"
    )

    invalid_span_ids: list[str] = Field(
        default_factory=list,
        description="IDs que não existem no documento (erros)"
    )


# =============================================================================
# PROMPTS PARA EXTRAÇÃO BASEADA EM SPANS
# =============================================================================

SPAN_EXTRACTION_SYSTEM_PROMPT = """Você é um especialista em documentos legais brasileiros.

Você receberá um documento legal com marcações de span no formato:
[SPAN_ID] texto do span

Sua tarefa é APENAS selecionar os IDs dos spans corretos para cada campo.
NUNCA gere texto - apenas retorne os IDs existentes.

Tipos de span:
- CAP-{romano}: Capítulo (ex: CAP-I, CAP-II)
- ART-{numero}: Artigo (ex: ART-001, ART-012)
- INC-{art}-{romano}: Inciso (ex: INC-001-I, INC-001-II)
- PAR-{art}-{numero}: Parágrafo (ex: PAR-001-1, PAR-001-UNICO)
- ALI-{art}-{inc}-{letra}: Alínea (ex: ALI-001-I-a)
- HDR-{seq}: Cabeçalho (ex: HDR-001)

Regras importantes:
1. Use APENAS IDs que aparecem no documento
2. Não invente IDs - se não existe, não inclua
3. Organize os artigos dentro dos capítulos corretos
4. Mantenha a ordem original do documento
"""

SPAN_EXTRACTION_USER_PROMPT = """Analise o documento abaixo e extraia a estrutura usando os span IDs.

DOCUMENTO:
{annotated_markdown}

---

Retorne um JSON com:
1. document_type: tipo do documento
2. number: número do documento
3. date: data (YYYY-MM-DD)
4. issuing_body: órgão emissor
5. ementa: resumo do documento
6. chapters: lista de capítulos, cada um com:
   - chapter_id: ID do capítulo (CAP-X)
   - title: título do capítulo
   - article_ids: lista de IDs dos artigos (ART-XXX)

Use APENAS os IDs que aparecem no documento acima."""
