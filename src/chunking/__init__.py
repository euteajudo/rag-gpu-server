"""
Módulo de Chunking para Documentos Legais Brasileiros.

Este módulo transforma documentos legais estruturados (LegalDocument) em chunks
prontos para indexação no Milvus, com suporte a parent-child retrieval e
enriquecimento via LLM (Contextual Retrieval da Anthropic).

Arquitetura do Módulo:
=====================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        PIPELINE DE CHUNKING                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  LegalDocument (JSON)                                                   │
    │         │                                                               │
    │         ├───────────────────┬──────────────────────┐                    │
    │         │                   │                      │                    │
    │         ▼                   ▼                      ▼                    │
    │  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
    │  │ LawChunker  │    │ ArticleChunk    │    │ ChunkMaterializer│         │
    │  │ (legado)    │    │ (via Parsing)   │    │ (novo)          │         │
    │  └─────────────┘    └─────────────────┘    └─────────────────┘         │
    │         │                   │                      │                    │
    │         ▼                   ▼                      ▼                    │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │                    MaterializedChunk                            │   │
    │  │  - chunk_id: "IN-65-2021#ART-005"                               │   │
    │  │  - parent_chunk_id: "" (artigo) ou "IN-65-2021#ART-005" (filho) │   │
    │  │  - device_type: ARTICLE, PARAGRAPH, INCISO                      │   │
    │  │  - text, enriched_text, context_header, thesis_text             │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │                    ┌──────────────────┐                                 │
    │                    │    BGE-M3        │                                 │
    │                    │ (dense + sparse) │                                 │
    │                    └──────────────────┘                                 │
    │                              │                                          │
    │                              ▼                                          │
    │                    ┌──────────────────┐                                 │
    │                    │  Milvus leis_v3  │                                 │
    │                    └──────────────────┘                                 │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Dois Caminhos de Chunking:
=========================

| Caminho         | Entrada          | Quando Usar                         |
|-----------------|------------------|-------------------------------------|
| LawChunker      | LegalDocument    | Pipeline legado, documentos simples |
| ChunkMaterializer| ArticleChunk    | Pipeline v3, parent-child retrieval |

Estrutura Parent-Child:
======================

    Chunk Pai (ARTICLE)           Chunks Filhos
    ┌─────────────────────┐       ┌─────────────────────┐
    │ IN-65-2021#ART-005  │──────▶│ IN-65-2021#PAR-005-1│
    │ parent_chunk_id: "" │       │ parent: ART-005     │
    │ type: ARTICLE       │       │ type: PARAGRAPH     │
    │ text: "Art. 5..."   │       │ text: "§1 ..."      │
    └─────────────────────┘       ├─────────────────────┤
                                  │ IN-65-2021#INC-005-I│
                                  │ parent: ART-005     │
                                  │ type: INCISO        │
                                  │ text: "I - ..."     │
                                  └─────────────────────┘

Estratégia de Busca:
===================

    Query → Busca chunks filhos (INC/PAR) → Agrega chunks pai → LLM

    1. Busca semântica retorna chunk filho (ex: INC-005-II)
    2. Sistema recupera chunk pai via parent_chunk_id (ex: ART-005)
    3. Contexto expandido passa para LLM (pai + filho + irmãos relevantes)

Enriquecimento LLM (Contextual Retrieval):
=========================================

Cada chunk é enriquecido com:

| Campo               | Gerado por | Descrição                              |
|---------------------|------------|----------------------------------------|
| context_header      | LLM        | Frase contextualizando o dispositivo   |
| thesis_text         | LLM        | Resumo do que o dispositivo determina  |
| thesis_type         | LLM        | Classificação (definicao, procedimento)|
| synthetic_questions | LLM        | Perguntas que o chunk responde         |
| enriched_text       | Código     | context + text + questions (embedding) |

Componentes Principais:
======================

| Componente           | Descrição                                     |
|----------------------|-----------------------------------------------|
| LegalChunk           | Dataclass para chunks prontos para Milvus     |
| ChunkLevel           | Enum: DOCUMENT, CHAPTER, ARTICLE, DEVICE      |
| ThesisType           | Enum: definicao, procedimento, prazo, etc     |
| ChunkingResult       | Resultado do chunking com estatísticas        |
| LawChunker           | Pipeline legado (LegalDocument → LegalChunk)  |
| ChunkMaterializer    | Pipeline v3 (ArticleChunk → MaterializedChunk)|
| MaterializedChunk    | Chunk com suporte parent-child                |
| MaterializationResult| Resultado com breakdown por tipo              |
| DeviceType           | Enum: ARTICLE, PARAGRAPH, INCISO, ALINEA      |
| ChunkMetadata        | Metadados de proveniência e versão            |

Exemplo de Uso (Pipeline v3):
============================

    from parsing import SpanParser, ArticleOrchestrator
    from chunking import ChunkMaterializer, DeviceType

    # 1. Parseia markdown
    parser = SpanParser()
    parsed_doc = parser.parse(markdown_text)

    # 2. Extrai hierarquia por artigo (via LLM)
    orchestrator = ArticleOrchestrator(llm_client)
    extraction = orchestrator.extract_all_articles(parsed_doc)

    # 3. Materializa em chunks
    materializer = ChunkMaterializer(
        document_id="IN-65-2021",
        tipo_documento="IN",
        numero="65",
        ano=2021
    )

    all_chunks = []
    for article_chunk in extraction.chunks:
        chunks = materializer.materialize_article(article_chunk, parsed_doc)
        all_chunks.extend(chunks)

    # 4. Estatísticas
    articles = sum(1 for c in all_chunks if c.device_type == DeviceType.ARTICLE)
    paragraphs = sum(1 for c in all_chunks if c.device_type == DeviceType.PARAGRAPH)
    incisos = sum(1 for c in all_chunks if c.device_type == DeviceType.INCISO)

    print(f"Total: {len(all_chunks)} chunks")
    print(f"  ARTICLE: {articles}")
    print(f"  PARAGRAPH: {paragraphs}")
    print(f"  INCISO: {incisos}")

Exemplo de Uso (Pipeline Legado):
================================

    from chunking import LawChunker, ChunkerConfig
    from models.legal_document import LegalDocument

    # 1. Carrega documento
    doc = LegalDocument.model_validate(json_data)

    # 2. Configura chunker
    config = ChunkerConfig(
        enrich_with_llm=True,
        generate_embeddings=True,
        batch_size=5
    )

    # 3. Chunking completo
    chunker = LawChunker(llm_client=llm, embedding_model=bge_m3, config=config)
    result = chunker.chunk_document(doc)

    # 4. Resultado
    print(result.summary())
    for chunk in result.chunks:
        print(f"{chunk.chunk_id}: {chunk.thesis_type}")

Módulos Relacionados:
====================

- parsing/span_parser.py: Gera ParsedDocument com spans
- parsing/article_orchestrator.py: Extrai ArticleChunks
- enrichment/chunk_enricher.py: Enriquece chunks via LLM
- embeddings/bge_m3.py: Gera embeddings dense + sparse
- milvus/schema_v3.py: Schema do Milvus com parent-child

@author: Equipe VectorGov
@version: 2.0.0
@since: 21/12/2024
"""

from .chunk_models import LegalChunk, ChunkLevel, ThesisType, ChunkingResult
from .law_chunker import LawChunker
from .chunk_materializer import (
    ChunkMaterializer,
    MaterializedChunk,
    MaterializationResult,
    ChunkMetadata,
    DeviceType,
    NodeIdValidationError,
)
from .node_id_metrics import (
    NodeIdMetricsCollector,
    NodeIdStats,
    validate_and_collect_metrics,
)

# Acórdãos TCU
from .acordao_chunker import (
    AcordaoChunker,
    AcordaoChunkMetadata,
    MaterializedAcordaoChunk,
    materialize_acordao,
)

# Extração de citações normativas
from .citation_extractor import (
    CitationExtractor,
    extract_citations_from_chunk,
    NormativeReference,
)

__all__ = [
    # Modelos
    "LegalChunk",
    "ChunkLevel",
    "ThesisType",
    "ChunkingResult",
    # Chunker original
    "LawChunker",
    # Parent-child materializer
    "ChunkMaterializer",
    "MaterializedChunk",
    "MaterializationResult",
    "ChunkMetadata",
    "DeviceType",
    # Validação de node_id
    "NodeIdValidationError",
    "NodeIdMetricsCollector",
    "NodeIdStats",
    "validate_and_collect_metrics",
    # Acórdãos TCU
    "AcordaoChunker",
    "AcordaoChunkMetadata",
    "MaterializedAcordaoChunk",
    "materialize_acordao",
    # Extração de citações
    "CitationExtractor",
    "extract_citations_from_chunk",
    "NormativeReference",
]
