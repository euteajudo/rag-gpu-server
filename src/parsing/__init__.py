"""
Módulo de Parsing para Documentos Legais Brasileiros.

Este módulo implementa extração determinística da estrutura hierárquica
de documentos legais usando regex, eliminando alucinações de LLM por design.
O LLM nunca "descobre" estrutura - apenas classifica spans já identificados.

Arquitetura do Parsing:
======================

    PDF (Docling)
         │
         ▼
    Markdown Estruturado
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                          SpanParser                                 │
    │                    (Extração Determinística)                        │
    │                                                                     │
    │  Markdown ──► [Regex Patterns] ──► Spans com IDs únicos             │
    │                                                                     │
    │  Padrões detectados:                                                │
    │  - CAPÍTULO I, II, III...    → CAP-I, CAP-II                        │
    │  - Art. 1º, Art. 2º...       → ART-001, ART-002                     │
    │  - § 1º, § 2º, § único       → PAR-001-1, PAR-001-UNICO             │
    │  - I -, II -, III -          → INC-001-I, INC-001-II                │
    │  - a), b), c)                → ALI-001-I-a, ALI-001-I-b             │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      ArticleOrchestrator                            │
    │                    (Extração LLM por Artigo)                        │
    │                                                                     │
    │  Para cada artigo:                                                  │
    │  1. Gera documento anotado: [SPAN_ID] texto                         │
    │  2. LLM extrai hierarquia (PAR-xxx, INC-xxx, ALI-xxx)               │
    │  3. Valida cobertura (parser vs LLM)                                │
    │  4. Retry focado se cobertura < 100%                                │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                       PageSpanExtractor                             │
    │                    (Coordenadas do PDF)                             │
    │                                                                     │
    │  Extrai bounding boxes do Docling para citações visuais:            │
    │  - page: número da página                                           │
    │  - bbox: {left, top, right, bottom}                                 │
    │  - coord_origin: TOPLEFT                                            │
    └─────────────────────────────────────────────────────────────────────┘

Hierarquia de Documentos Legais Brasileiros:
============================================

    CAPÍTULO (CAP-I, CAP-II)
        │
        └── Seção (opcional)
              │
              └── Subseção (opcional)
                    │
                    └── Artigo (ART-001, ART-002)
                          │
                          ├── Caput (texto principal do artigo)
                          │
                          ├── Parágrafo (PAR-001-1, PAR-001-UNICO)
                          │     │
                          │     └── Inciso (INC-001-I_1, sufixo para § específico)
                          │
                          └── Inciso (INC-001-I, INC-001-II)
                                │
                                └── Alínea (ALI-001-I-a, ALI-001-I-b)
                                      │
                                      └── Item (raro, dentro de alíneas)

Formato de Span IDs:
===================

| Tipo      | Formato              | Exemplo                      |
|-----------|----------------------|------------------------------|
| Capítulo  | CAP-{romano}         | CAP-I, CAP-II                |
| Artigo    | ART-{nnn}            | ART-001, ART-012             |
| Parágrafo | PAR-{art}-{n}        | PAR-001-1, PAR-001-UNICO     |
| Inciso    | INC-{art}-{romano}   | INC-001-I, INC-001-IV        |
| Alínea    | ALI-{art}-{inc}-{l}  | ALI-001-I-a, ALI-001-II-b    |
| Inc de §  | INC-{art}-{r}_{par}  | INC-001-I_2 (inciso I do §2) |

Componentes Principais:
======================

1. SpanParser
   Parser determinístico usando regex. Identifica estrutura sem LLM.
   - Entrada: Markdown do Docling
   - Saída: ParsedDocument com spans indexados

2. ArticleOrchestrator
   Orquestra extração LLM por artigo com validação de cobertura.
   - Schema enum dinâmico: IDs permitidos passados ao LLM
   - Retry focado: se faltou PAR, retry com janela de PAR
   - Curto-circuito: artigos sem filhos não chamam LLM

3. PageSpanExtractor
   Extrai coordenadas PDF para navegação visual.
   - Usa provenance do Docling
   - Mapeia spans para páginas e bounding boxes

4. SpanExtractor
   Extrator simplificado para uso standalone.

Exemplo de Uso:
==============

    ```python
    from parsing import SpanParser, ArticleOrchestrator, SpanType

    # 1. Parse markdown para spans (determinístico)
    parser = SpanParser()
    doc = parser.parse(markdown_text)

    print(f"Artigos encontrados: {len(doc.articles)}")
    for span in doc.articles:
        print(f"  {span.span_id}: {span.text[:50]}...")

    # 2. Extração LLM por artigo
    from llm import VLLMClient

    llm = VLLMClient(...)
    orchestrator = ArticleOrchestrator(llm)
    result = orchestrator.extract_all_articles(doc)

    print(f"Taxa de sucesso: {result.success_rate:.0%}")
    for chunk in result.chunks:
        print(f"  {chunk.article_id}: {len(chunk.citations)} citações")

    # 3. Coordenadas PDF
    from parsing import PageSpanExtractor

    extractor = PageSpanExtractor()
    span_locations = extractor.map_spans_to_locations(doc, text_locations)
    for span_id, loc in span_locations.items():
        print(f"  {span_id}: página {loc.page}")
    ```

Anti-Alucinação por Design:
==========================

O módulo foi desenhado para eliminar alucinações de LLM:

| Problema           | Solução                                    |
|--------------------|--------------------------------------------|
| LLM inventa IDs    | Schema enum dinâmico com IDs permitidos    |
| LLM pula spans     | Validação de cobertura parser vs LLM       |
| LLM duplica IDs    | Detecção de duplicatas no orchestrator     |
| LLM mistura artigos| Extração isolada por artigo                |

Módulos Relacionados:
====================

- chunking/chunk_materializer.py: Converte spans em chunks indexáveis
- enrichment/chunk_enricher.py: Enriquece chunks com contexto LLM
- llm/vllm_client.py: Cliente LLM para extração
- docling: Conversão PDF → Markdown

@author: Equipe VectorGov
@version: 1.0.0
@since: 23/12/2024
"""

from .span_models import (
    SpanType,
    Span,
    ParsedDocument,
)
from .span_parser import SpanParser, ParserConfig
from .span_extraction_models import (
    DocumentSpans,
    ChapterSpans,
    ArticleSpans,
    SpanClassification,
    SpanExtractionResult,
)
from .span_extractor import (
    SpanExtractor,
    SpanExtractorConfig,
    ExtractionResult,
    extract_with_spans,
)
from .article_orchestrator import (
    ArticleOrchestrator,
    OrchestratorConfig,
    ArticleChunk,
    ArticleExtractionResult,
    ValidationStatus,
    extract_articles_with_hierarchy,
)
from .page_spans import (
    PageSpanExtractor,
    BoundingBox,
    TextLocation,
    SpanLocation,
    extract_page_spans_from_pdf,
)

# Acórdãos TCU
from .acordao_models import (
    AcordaoSpanType,
    AcordaoSpan,
    AcordaoMetadata,
    ParsedAcordao,
    normalize_acordao_id,
    parse_colegiado,
)
from .acordao_span_parser import AcordaoSpanParser

# Address Validator (ADDRESS_MISMATCH detection)
from .address_validator import AddressValidator, ValidationResult

__all__ = [
    # Core models
    "SpanType",
    "Span",
    "ParsedDocument",
    # Parser
    "SpanParser",
    "ParserConfig",
    # Extraction models
    "DocumentSpans",
    "ChapterSpans",
    "ArticleSpans",
    "SpanClassification",
    "SpanExtractionResult",
    # Extractor (Fase 2)
    "SpanExtractor",
    "SpanExtractorConfig",
    "ExtractionResult",
    "extract_with_spans",
    # Orchestrator (Fase 3)
    "ArticleOrchestrator",
    "OrchestratorConfig",
    "ArticleChunk",
    "ArticleExtractionResult",
    "ValidationStatus",
    "extract_articles_with_hierarchy",
    # Page spans (Fase 4)
    "PageSpanExtractor",
    "BoundingBox",
    "TextLocation",
    "SpanLocation",
    "extract_page_spans_from_pdf",
    # Acórdãos TCU
    "AcordaoSpanType",
    "AcordaoSpan",
    "AcordaoMetadata",
    "ParsedAcordao",
    "normalize_acordao_id",
    "parse_colegiado",
    "AcordaoSpanParser",
    # Address Validator
    "AddressValidator",
    "ValidationResult",
]
