"""
Chunk Materializer - Transforma ArticleChunk em Chunks Indexáveis com Parent-Child.

Este módulo materializa a hierarquia extraída pelo ArticleOrchestrator em chunks
prontos para indexação no Milvus, com suporte completo a parent-child retrieval.
É o componente central do Pipeline v3 (Span-Based) que converte estruturas
hierárquicas em chunks indexáveis.

Arquitetura do Materializer:
===========================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    FLUXO DE MATERIALIZAÇÃO                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  ArticleChunk                                                           │
    │  (do ArticleOrchestrator)                                               │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │ article_id: ART-005                                               │  │
    │  │ text: "Art. 5º O estudo técnico..."                               │  │
    │  │ paragrafo_ids: [PAR-005-1, PAR-005-2]                             │  │
    │  │ inciso_ids: [INC-005-I, INC-005-II, INC-005-III]                  │  │
    │  │ citations: [ART-005, PAR-005-1, ..., INC-005-III]                 │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    │                              │                                          │
    │                              ▼                                          │
    │                   ┌─────────────────────┐                               │
    │                   │  ChunkMaterializer  │                               │
    │                   │  materialize_article│                               │
    │                   └─────────────────────┘                               │
    │                              │                                          │
    │              ┌───────────────┼───────────────┐                          │
    │              ▼               ▼               ▼                          │
    │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
    │    │ ARTICLE     │  │ PARAGRAPH   │  │ INCISO      │                    │
    │    │ (chunk pai) │  │ (chunk      │  │ (chunk      │                    │
    │    │             │  │  filho)     │  │  filho)     │                    │
    │    │ parent: ""  │  │ parent:     │  │ parent:     │                    │
    │    └─────────────┘  │ ART-005     │  │ ART-005     │                    │
    │                     └─────────────┘  └─────────────┘                    │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Estrutura Parent-Child:
======================

A estratégia parent-child permite recuperação contextual expandida:

    ┌───────────────────────────────────────────────────────────────────────┐
    │                    HIERARQUIA DE CHUNKS                               │
    ├───────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │  Chunk PAI (ARTICLE)                                                  │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │ chunk_id: "IN-65-2021#ART-005"                                  │  │
    │  │ parent_node_id: ""  ← Artigo não tem pai                        │  │
    │  │ span_id: "ART-005"                                              │  │
    │  │ device_type: ARTICLE                                            │  │
    │  │ text: "Art. 5º O estudo técnico preliminar..."                  │  │
    │  │ citations: ["ART-005", "PAR-005-1", "INC-005-I", ...]           │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                    │                                                  │
    │          ┌─────────┴─────────┐                                        │
    │          ▼                   ▼                                        │
    │  ┌─────────────────┐  ┌─────────────────┐                             │
    │  │ Chunk FILHO     │  │ Chunk FILHO     │                             │
    │  │ (PARAGRAPH)     │  │ (INCISO)        │                             │
    │  ├─────────────────┤  ├─────────────────┤                             │
    │  │ chunk_id:       │  │ chunk_id:       │                             │
    │  │ IN-65-2021#     │  │ IN-65-2021#     │                             │
    │  │ PAR-005-1       │  │ INC-005-I       │                             │
    │  │                 │  │                 │                             │
    │  │ parent_node_id: │  │ parent_node_id: │                             │
    │  │ leis:IN-65-2021#│  │ leis:IN-65-2021#│                             │
    │  │ ART-005         │  │ ART-005         │                             │
    │  │                 │  │                 │                             │
    │  │ span_id:        │  │ span_id:        │                             │
    │  │ PAR-005-1       │  │ INC-005-I       │                             │
    │  │                 │  │                 │                             │
    │  │ device_type:    │  │ device_type:    │                             │
    │  │ PARAGRAPH       │  │ INCISO          │                             │
    │  └─────────────────┘  └─────────────────┘                             │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘

Fluxo de Retrieval com Parent-Child:
===================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   ESTRATÉGIA DE BUSCA PARENT-CHILD                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. Query "Quando o ETP pode ser dispensado?"                           │
    │                    │                                                    │
    │                    ▼                                                    │
    │  2. Busca semântica retorna chunk FILHO mais relevante                  │
    │     → INC-005-II (score: 0.95)                                          │
    │                    │                                                    │
    │                    ▼                                                    │
    │  3. Sistema recupera chunk PAI via parent_node_id                       │
    │     → Busca: node_id == "leis:IN-65-2021#ART-005" (parent_node_id)      │
    │     → Obtém: ART-005 (texto completo do artigo)                         │
    │                    │                                                    │
    │                    ▼                                                    │
    │  4. Monta contexto expandido:                                           │
    │     ┌────────────────────────────────────────────────────────┐          │
    │     │ CONTEXTO PARA O LLM:                                   │          │
    │     │                                                        │          │
    │     │ [PAI] Art. 5º O estudo técnico preliminar...           │          │
    │     │                                                        │          │
    │     │ [FILHOS RELEVANTES]                                    │          │
    │     │ § 1º O ETP será dispensado quando...                   │          │
    │     │ I - contratação direta por...                          │          │
    │     │ II - prorrogação de contratos... ← MATCH               │          │
    │     └────────────────────────────────────────────────────────┘          │
    │                    │                                                    │
    │                    ▼                                                    │
    │  5. LLM gera resposta com contexto completo                             │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

DeviceType (Tipos de Dispositivo):
=================================

| Enum       | Valor       | Descrição                          | Chunk Level |
|------------|-------------|---------------------------------------|-------------|
| ARTICLE    | "article"   | Artigo (chunk pai)                    | ARTICLE     |
| PARAGRAPH  | "paragraph" | Parágrafo (§1º, §2º, § único)        | DEVICE      |
| INCISO     | "inciso"    | Inciso (I, II, III...)                | DEVICE      |
| ALINEA     | "alinea"    | Alínea (a, b, c...) - dentro de INC   | DEVICE      |

ChunkMetadata (Metadados de Proveniência):
=========================================

| Campo               | Tipo   | Descrição                              |
|---------------------|--------|----------------------------------------|
| schema_version      | str    | Versão do schema (ex: "1.0.0")         |
| extractor_version   | str    | Versão do extrator (ex: "1.0.0")       |
| ingestion_timestamp | str    | Timestamp ISO da ingestão              |
| document_hash       | str    | SHA-256 do PDF original                |
| pdf_path            | str    | Caminho do arquivo PDF (opcional)      |
| valid_from          | str    | Data início vigência (YYYY-MM-DD)      |
| valid_to            | str    | Data fim vigência (None = vigente)     |
| page_spans          | dict   | Coordenadas no PDF por span_id         |

MaterializedChunk (Campos Principais):
=====================================

| Campo              | Tipo          | Descrição                              |
|--------------------|---------------|----------------------------------------|
| chunk_id           | str           | ID único: "{doc}#{span}"               |
| parent_node_id     | str           | node_id do pai ("" para artigos)       |
| span_id            | str           | ID do span: ART-005, PAR-005-1         |
| device_type        | DeviceType    | ARTICLE, PARAGRAPH, INCISO, ALINEA     |
| chunk_level        | ChunkLevel    | ARTICLE ou DEVICE                      |
| text               | str           | Texto original do dispositivo          |
| enriched_text      | str           | Contexto + texto + perguntas           |
| context_header     | str           | Frase contextualizando o dispositivo   |
| thesis_text        | str           | Resumo do que o dispositivo determina  |
| thesis_type        | str           | Tipo: definicao, procedimento, etc     |
| synthetic_questions| str           | Perguntas que o chunk responde         |
| aliases            | list[str]     | Termos alternativos para recall        |
| sparse_source      | str           | enriched_text + aliases (para sparse)  |
| document_id        | str           | ID do documento: "IN-65-2021"          |
| tipo_documento     | str           | LEI, DECRETO, IN, PORTARIA             |
| numero             | str           | Número do documento                    |
| ano                | int           | Ano do documento                       |
| article_number     | str           | Número do artigo: "5", "10"            |
| citations          | list[str]     | Spans que compõem este chunk           |
| dense_vector       | list[float]   | Embedding BGE-M3 (1024 dims)           |
| sparse_vector      | dict          | Sparse embedding BGE-M3                |

Formato de IDs:
==============

| Componente       | Formato                  | Exemplo                    |
|------------------|--------------------------|----------------------------|
| chunk_id (pai)   | {doc_id}#{ART-nnn}       | IN-65-2021#ART-005         |
| chunk_id (filho) | {doc_id}#{span_id}       | IN-65-2021#PAR-005-1       |
| span_id artigo   | ART-{nnn}                | ART-005, ART-012           |
| span_id parágrafo| PAR-{art}-{n}            | PAR-005-1, PAR-005-UNICO   |
| span_id inciso   | INC-{art}-{romano}       | INC-005-I, INC-005-II      |
| span_id alínea   | ALI-{art}-{romano}-{let} | ALI-005-I-a, ALI-005-II-b  |

Exemplo de Uso:
==============

    from parsing import SpanParser, ArticleOrchestrator
    from chunking import ChunkMaterializer, DeviceType

    # 1. Parseia markdown para spans
    parser = SpanParser()
    parsed_doc = parser.parse(markdown_text)

    # 2. Extrai hierarquia por artigo (via LLM)
    orchestrator = ArticleOrchestrator(llm_client)
    extraction = orchestrator.extract_all_articles(parsed_doc)

    # 3. Configura materializer
    materializer = ChunkMaterializer(
        document_id="IN-65-2021",
        tipo_documento="IN",
        numero="65",
        ano=2021
    )

    # 4. Materializa todos os artigos
    all_chunks = materializer.materialize_all(
        extraction.chunks,
        parsed_doc,
        include_children=True
    )

    # 5. Estatísticas por tipo
    articles = [c for c in all_chunks if c.device_type == DeviceType.ARTICLE]
    paragraphs = [c for c in all_chunks if c.device_type == DeviceType.PARAGRAPH]
    incisos = [c for c in all_chunks if c.device_type == DeviceType.INCISO]

    print(f"Total: {len(all_chunks)} chunks")
    print(f"  ARTICLE: {len(articles)}")
    print(f"  PARAGRAPH: {len(paragraphs)}")
    print(f"  INCISO: {len(incisos)}")

    # 6. Preparar para Milvus
    for chunk in all_chunks:
        milvus_row = chunk.to_milvus_dict()
        # Insere no Milvus...

Métodos do ChunkMaterializer:
============================

| Método               | Parâmetros                            | Retorno                    |
|----------------------|---------------------------------------|----------------------------|
| materialize_article  | article_chunk, parsed_doc, children   | list[MaterializedChunk]    |
| materialize_all      | article_chunks, parsed_doc, children  | list[MaterializedChunk]    |
| _reconstruct_inciso  | inc_id, parsed_doc                    | str (texto com alíneas)    |
| _get_inciso_citations| inc_id, parsed_doc                    | list[str] (span_ids)       |

MaterializationResult (Estatísticas):
====================================

    result = MaterializationResult(
        chunks=all_chunks,
        total_chunks=len(all_chunks),
        article_chunks=len(articles),
        paragraph_chunks=len(paragraphs),
        inciso_chunks=len(incisos),
        document_id="IN-65-2021",
        ingestion_timestamp=datetime.utcnow().isoformat()
    )

    print(result.summary())
    # {
    #     "document_id": "IN-65-2021",
    #     "total_chunks": 47,
    #     "breakdown": {
    #         "articles": 11,
    #         "paragraphs": 19,
    #         "incisos": 17
    #     },
    #     "ingestion_timestamp": "2024-12-23T14:30:00Z"
    # }

Integração com Outros Módulos:
=============================

- **parsing/span_parser.py**: Gera ParsedDocument com spans identificados
- **parsing/article_orchestrator.py**: Extrai ArticleChunks por artigo
- **enrichment/chunk_enricher.py**: Enriquece chunks (context, thesis, questions)
- **embeddings/bge_m3.py**: Gera embeddings dense + sparse
- **milvus/schema_v3.py**: Define schema da collection com parent-child

Pipeline Completo (v3):
======================

    PDF
     │
     ▼
    [Docling] → Markdown
     │
     ▼
    [SpanParser] → ParsedDocument (spans com IDs)
     │
     ▼
    [ArticleOrchestrator] → ArticleChunks (hierarquia por artigo)
     │
     ▼
    [ChunkMaterializer] → MaterializedChunks (parent-child) ← VOCÊ ESTÁ AQUI
     │
     ▼
    [ChunkEnricher] → Chunks enriquecidos (LLM)
     │
     ▼
    [BGE-M3] → Embeddings (dense + sparse)
     │
     ▼
    [Milvus leis_v3] → Indexação com parent-child

@author: Equipe VectorGov
@version: 1.0.0
@since: 23/12/2024
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import logging
import re

from .chunk_models import LegalChunk, ChunkLevel
from .canonical_offsets import resolve_child_offsets, OffsetResolutionError

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURAÇÃO DE SPLITTING PARA ARTIGOS GRANDES
# ==============================================================================
# Quando um artigo ultrapassa SPLIT_THRESHOLD_CHARS, ele é dividido em subchunks
# menores (PART) que são indexados como filhos do artigo pai canônico.
#
# Hierarquia resultante:
#   leis:DOC#ART-005           (pai canônico - texto completo, NÃO indexado no Milvus)
#   leis:DOC#ART-005-P1        (parte 1 - indexada)
#   leis:DOC#ART-005-P2        (parte 2 - indexada)
#   leis:DOC#ART-005-P3        (parte 3 - indexada)
#
# O artigo pai mantém integridade para navegação e citações, mas os subchunks
# são o que efetivamente vai para o Milvus para retrieval.
# ==============================================================================

SPLIT_THRESHOLD_CHARS: int = 8000    # Artigo > 8k chars será splitado
TARGET_CHUNK_CHARS: int = 4000       # Tamanho alvo de cada subchunk
OVERLAP_CHARS: int = 300             # Overlap entre subchunks para contexto

# Marcadores jurídicos para split (ordem de preferência)
LEGAL_MARKERS = [
    r'(?:^|\n)\s*§\s*\d+[º°]?',           # Parágrafos: § 1º, § 2º
    r'(?:^|\n)\s*Parágrafo\s+único',      # Parágrafo único
    r'(?:^|\n)\s*[IVXLCDM]+\s*[-–—]',     # Incisos romanos: I -, II -, III -
    r'(?:^|\n)\s*[a-z]\)\s',              # Alíneas: a), b), c)
    r'(?:^|\n)\s*\d+\)\s',                # Itens numerados: 1), 2), 3)
    r'(?:^|\n)\s*Art\.\s*\d+',            # Referência a outros artigos
]

# Regex para encontrar quebras de parágrafo
PARAGRAPH_BREAK = re.compile(r'\n\s*\n')

# Regex para encontrar fim de sentença
SENTENCE_END = re.compile(r'[.;:]\s+')


class NodeIdValidationError(Exception):
    """Erro de validação de node_id."""

    def __init__(self, message: str, chunk_id: str = "", node_id: str = ""):
        self.chunk_id = chunk_id
        self.node_id = node_id
        super().__init__(message)


class DeviceType(str, Enum):
    """Tipo de dispositivo legal."""
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    INCISO = "inciso"
    ALINEA = "alinea"
    PART = "part"  # Subchunk de artigo grande (ART-005-P1, ART-005-P2, etc.)


@dataclass
class ChunkMetadata:
    """Metadados de proveniência e versão."""

    # Versões
    schema_version: str = "1.0.0"
    extractor_version: str = "1.0.0"

    # Timestamps
    ingestion_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Proveniência do documento
    document_hash: str = ""  # SHA-256 do PDF
    pdf_path: Optional[str] = None

    # Versão do documento (vigência)
    valid_from: Optional[str] = None  # Data início vigência
    valid_to: Optional[str] = None    # Data fim vigência (None = vigente)

    # Citações visuais (coordenadas no PDF)
    page_spans: dict = field(default_factory=dict)  # {span_id: {page, x, y, w, h}}

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "extractor_version": self.extractor_version,
            "ingestion_timestamp": self.ingestion_timestamp,
            "document_hash": self.document_hash,
            "pdf_path": self.pdf_path,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "page_spans": self.page_spans,
        }


@dataclass
class MaterializedChunk:
    """Chunk materializado com suporte a parent-child."""

    # IDs
    node_id: str            # Ex: "leis:IN-65-2021#ART-005" (PK canônica para Milvus/Neo4j)
    chunk_id: str           # Ex: "IN-65-2021#ART-005" ou "IN-65-2021#PAR-005-1"
    parent_node_id: str     # Ex: "" para artigos, "leis:IN-65-2021#ART-005" para filhos
    span_id: str            # Ex: "ART-005", "PAR-005-1", "INC-005-I"

    # Tipo
    device_type: DeviceType
    chunk_level: ChunkLevel

    # Conteúdo
    text: str
    enriched_text: str = ""
    parent_text: str = ""  # Texto do artigo pai (caput) para dispositivos filhos
    retrieval_text: str = ""  # Texto determinístico para embedding (sem LLM)

    # Contexto
    context_header: str = ""
    thesis_text: str = ""
    thesis_type: str = "disposicao"
    synthetic_questions: str = ""

    # Aliases (para melhoria de recall na busca esparsa)
    aliases: list[str] = field(default_factory=list)
    sparse_source: str = ""  # enriched_text + aliases

    # Hierarquia legal
    document_id: str = ""
    tipo_documento: str = ""
    numero: str = ""
    ano: int = 0
    article_number: str = ""

    # Citations (lista de span_ids que compõem este chunk)
    citations: list[str] = field(default_factory=list)

    # Metadados
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Embeddings (preenchidos depois)
    dense_vector: Optional[list[float]] = None
    sparse_vector: Optional[dict[int, float]] = None

    # PR13: Offsets verdadeiros no canonical_text (zero fallback find)
    # Quando canonical_hash == hash_atual E start/end >= 0:
    #   → usa slicing puro: canonical_text[start:end]
    # Caso contrário:
    #   → fallback best-effort via find()
    canonical_start: int = -1  # Offset início no canonical_text (-1 se desconhecido)
    canonical_end: int = -1    # Offset fim no canonical_text (-1 se desconhecido)
    canonical_hash: str = ""   # SHA256 do canonical_text para anti-mismatch

    # OriginClassifier: Classificação de origem material
    # Distingue entre material da própria lei vs material de outras leis citadas
    # Ex: Art. 337-E está no PDF da Lei 14.133 mas É do Código Penal
    origin_type: str = "self"           # "self" | "external"
    origin_reference: Optional[str] = None      # "DL-2848-1940" | None
    origin_reference_name: Optional[str] = None # "Código Penal" | None
    is_external_material: bool = False
    origin_confidence: str = "high"     # "high" | "medium" | "low"
    origin_reason: Optional[str] = None # "rule:codigo_penal_art337" | None

    def validate(self) -> None:
        """
        Valida consistência do node_id.

        Regras:
        - chunk_id deve existir e ser não-vazio
        - node_id deve existir e ser não-vazio
        - node_id deve ser igual a f"leis:{chunk_id}"

        Raises:
            NodeIdValidationError: Se qualquer validação falhar
        """
        # Validação 1: chunk_id existe e é não-vazio
        if not self.chunk_id or not self.chunk_id.strip():
            raise NodeIdValidationError(
                message=f"chunk_id vazio ou ausente para span_id={self.span_id}",
                chunk_id=self.chunk_id,
                node_id=self.node_id,
            )

        # Validação 2: node_id existe e é não-vazio
        if not self.node_id or not self.node_id.strip():
            raise NodeIdValidationError(
                message=f"node_id vazio ou ausente para chunk_id={self.chunk_id}",
                chunk_id=self.chunk_id,
                node_id=self.node_id,
            )

        # Validação 3: node_id == f"leis:{chunk_id}"
        expected_node_id = f"leis:{self.chunk_id}"
        if self.node_id != expected_node_id:
            raise NodeIdValidationError(
                message=f"node_id inconsistente: esperado '{expected_node_id}', obtido '{self.node_id}'",
                chunk_id=self.chunk_id,
                node_id=self.node_id,
            )

    def to_milvus_dict(self) -> dict:
        """Converte para formato Milvus com campos dinâmicos."""
        return {
            # Campos principais
            "node_id": self.node_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "enriched_text": self.enriched_text,

            # Embeddings
            "dense_vector": self.dense_vector or [],
            "sparse_vector": self.sparse_vector or {},

            # Campos para filtro
            "document_id": self.document_id,
            "tipo_documento": self.tipo_documento,
            "numero": self.numero,
            "ano": self.ano,
            "article_number": self.article_number,

            # Campos dinâmicos (parent-child)
            "parent_node_id": self.parent_node_id,
            "span_id": self.span_id,
            "device_type": self.device_type.value,
            "chunk_level": self.chunk_level.value,
            "citations": self.citations,

            # Enriquecimento
            "context_header": self.context_header,
            "thesis_text": self.thesis_text,
            "thesis_type": self.thesis_type,
            "synthetic_questions": self.synthetic_questions,

            # Aliases
            "aliases": self.aliases,
            "sparse_source": self.sparse_source,

            # PR13: Offsets verdadeiros (zero fallback find)
            "canonical_start": self.canonical_start,
            "canonical_end": self.canonical_end,
            "canonical_hash": self.canonical_hash,

            # Proveniência
            **self.metadata.to_dict(),
        }


# ==============================================================================
# RETRIEVAL TEXT DETERMINÍSTICO
# ==============================================================================
# Função que constrói texto para embedding de forma 100% determinística,
# sem depender de LLM para enriquecimento. Usa apenas:
# - Metadados confiáveis (document_id, article_number, device_type)
# - Texto original do dispositivo (chunk.text)
# - Texto do artigo pai - caput (parent_text)
#
# Formato:
#   {document_id} | Art. {article_number} | {device_type_label}
#   [CAPUT] {parent_text truncado}
#   [DISPOSITIVO] {chunk.text}
# ==============================================================================

# Constantes para truncamento do caput (evita contexto muito longo)
CAPUT_HEAD_CHARS: int = 600   # Primeiros 600 chars do caput (tema principal)
CAPUT_TAIL_CHARS: int = 200   # Últimos 200 chars do caput (condições finais)

# Mapeamento de device_type para label legível
DEVICE_TYPE_LABELS = {
    DeviceType.ARTICLE: "Artigo",
    DeviceType.PARAGRAPH: "Parágrafo",
    DeviceType.INCISO: "Inciso",
    DeviceType.ALINEA: "Alínea",
    DeviceType.PART: "Parte",
}


def build_retrieval_text(
    document_id: str,
    article_number: str,
    device_type: DeviceType,
    text: str,
    parent_text: str = "",
    span_id: str = "",
) -> str:
    """
    Constrói texto determinístico para embedding sem dependência de LLM.

    Esta função gera um texto estruturado para o campo dense_vector
    que captura semântica de forma confiável usando apenas dados conhecidos.

    Args:
        document_id: ID do documento (ex: "IN-58-2022", "LEI-14133-2021")
        article_number: Número do artigo (ex: "14", "5")
        device_type: Tipo do dispositivo (ARTICLE, PARAGRAPH, INCISO, etc)
        text: Texto original do chunk
        parent_text: Texto do artigo pai (caput) - para dispositivos filhos
        span_id: ID do span (ex: "INC-014-I") - para label mais específico

    Returns:
        Texto estruturado para embedding:
        ```
        IN-58-2022 | Art. 14 | Inciso I
        I - contratação direta por inexigibilidade...
        [CONTEXTO] Art. 14. A elaboração do ETP é facultada...
        ```

        IMPORTANTE: O dispositivo vem PRIMEIRO para ter mais peso no embedding.
        O caput do pai vem depois como [CONTEXTO] para evitar que vetores
        de dispositivos do mesmo artigo fiquem muito similares.

    Example:
        >>> text = build_retrieval_text(
        ...     document_id="IN-58-2022",
        ...     article_number="14",
        ...     device_type=DeviceType.INCISO,
        ...     text="I - contratação direta por inexigibilidade...",
        ...     parent_text="Art. 14. A elaboração do ETP é facultada...",
        ...     span_id="INC-014-I"
        ... )
        >>> print(text[:50])
        'IN-58-2022 | Art. 14 | Inciso I'
    """
    lines = []

    # 1. Cabeçalho com metadados
    device_label = DEVICE_TYPE_LABELS.get(device_type, device_type.value.capitalize())

    # Extrai identificador específico do span_id para dispositivos filhos
    specific_id = ""
    if device_type == DeviceType.INCISO and span_id:
        # INC-014-I -> I
        parts = span_id.split("-")
        if len(parts) >= 3:
            specific_id = f" {parts[-1]}"  # " I", " II", etc.
    elif device_type == DeviceType.PARAGRAPH and span_id:
        # PAR-014-1 -> § 1º, PAR-014-UNICO -> § único
        parts = span_id.split("-")
        if len(parts) >= 3:
            par_num = parts[-1]
            if par_num.upper() == "UNICO":
                specific_id = " único"
            else:
                specific_id = f" {par_num}º"
    elif device_type == DeviceType.ALINEA and span_id:
        # ALI-014-I-a -> a)
        parts = span_id.split("-")
        if len(parts) >= 4:
            specific_id = f" {parts[-1]})"

    header = f"{document_id} | Art. {article_number} | {device_label}{specific_id}"
    lines.append(header)

    # 2. Texto do dispositivo PRIMEIRO (destaque - mais peso no embedding)
    lines.append(text)

    # 3. Caput do artigo pai como CONTEXTO (se disponível e não for o próprio artigo)
    # Vem DEPOIS para não "diluir" a especificidade do dispositivo
    if parent_text and device_type != DeviceType.ARTICLE:
        truncated_caput = _truncate_head_tail(
            parent_text,
            head_chars=CAPUT_HEAD_CHARS,
            tail_chars=CAPUT_TAIL_CHARS
        )
        lines.append(f"[CONTEXTO] {truncated_caput}")

    return "\n".join(lines)


def _truncate_head_tail(text: str, head_chars: int, tail_chars: int) -> str:
    """
    Trunca texto mantendo início e fim, similar à estratégia do GraphRetriever.

    Args:
        text: Texto a truncar
        head_chars: Caracteres a manter do início
        tail_chars: Caracteres a manter do fim

    Returns:
        Texto truncado com "..." no meio se necessário
    """
    if len(text) <= head_chars + tail_chars:
        return text

    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip()

    # Tenta não cortar no meio de palavra
    if head and not head[-1].isspace() and not head.endswith(('.', ',', ';', ':')):
        last_space = head.rfind(' ')
        if last_space > head_chars - 100:  # Não recua muito
            head = head[:last_space]

    return f"{head} ... {tail}"


class ChunkMaterializer:
    """
    Materializa ArticleChunk em chunks indexáveis com parent-child.

    Gera:
    - 1 chunk ARTICLE (pai) com texto completo
    - N chunks PARAGRAPH (filhos) para cada parágrafo
    - M chunks INCISO (filhos) para cada inciso

    PR13: Suporta offsets canônicos para slicing direto (zero fallback find).
    Se offsets_map e canonical_hash forem fornecidos, os chunks materializados
    terão canonical_start, canonical_end e canonical_hash preenchidos.
    """

    def __init__(
        self,
        document_id: str,
        tipo_documento: str = "",
        numero: str = "",
        ano: int = 0,
        metadata: Optional[ChunkMetadata] = None,
        offsets_map: Optional[dict[str, tuple[int, int]]] = None,
        canonical_hash: str = "",
        canonical_text: str = "",
    ):
        """
        Inicializa o ChunkMaterializer.

        Args:
            document_id: ID do documento (ex: "IN-65-2021")
            tipo_documento: Tipo do documento (ex: "IN", "LEI")
            numero: Número do documento
            ano: Ano do documento
            metadata: Metadados de proveniência
            offsets_map: PR13 - Mapa span_id → (start, end) para offsets canônicos (artigos)
            canonical_hash: PR13 - SHA256 do canonical_text para anti-mismatch
            canonical_text: PR13 STRICT - Texto canônico completo para resolução de offsets de filhos
        """
        self.document_id = document_id
        self.tipo_documento = tipo_documento
        self.numero = numero
        self.ano = ano
        self.metadata = metadata or ChunkMetadata()
        # PR13: Offsets canônicos
        self.offsets_map = offsets_map or {}
        self.canonical_hash = canonical_hash
        # PR13 STRICT: Texto canônico para resolução de offsets de filhos
        self.canonical_text = canonical_text
        # Cache de offsets resolvidos (para hierarquia PAR -> INC)
        self._resolved_child_offsets: dict[str, tuple[int, int]] = {}

    def materialize_article(
        self,
        article_chunk,  # ArticleChunk do article_orchestrator
        parsed_doc,     # ParsedDocument para reconstruir texto
        include_children: bool = True
    ) -> list[MaterializedChunk]:
        """
        Materializa um ArticleChunk em chunks pai e filhos.

        Se o artigo exceder SPLIT_THRESHOLD_CHARS (8000), usa _materialize_large_article
        para criar subchunks (PART) em vez de indexar o artigo inteiro.

        Args:
            article_chunk: ArticleChunk extraído
            parsed_doc: Documento parseado (para reconstruir texto dos filhos)
            include_children: Se True, gera chunks para PAR/INC também

        Returns:
            Lista de MaterializedChunk (1 pai + N filhos)
        """
        # Verifica se precisa splittar artigo grande
        if self._should_split_article(article_chunk.text):
            logger.info(
                f"Artigo {article_chunk.article_id} com {len(article_chunk.text)} chars "
                f"excede threshold de {SPLIT_THRESHOLD_CHARS}, splitando..."
            )
            return self._materialize_large_article(article_chunk, parsed_doc, include_children)

        chunks = []

        # 1. Chunk pai (ARTICLE)
        parent_chunk_id = f"{self.document_id}#{article_chunk.article_id}"
        parent_node_id = f"leis:{parent_chunk_id}"

        # Texto do artigo (caput) - será usado como parent_text para filhos
        article_caput = article_chunk.text

        # Constrói retrieval_text determinístico para o artigo
        article_retrieval_text = build_retrieval_text(
            document_id=self.document_id,
            article_number=article_chunk.article_number,
            device_type=DeviceType.ARTICLE,
            text=article_caput,
            parent_text="",  # Artigo não tem pai
            span_id=article_chunk.article_id,
        )

        # PR13: Obtém offsets canônicos para o artigo
        art_start, art_end, art_hash = self._get_canonical_offsets(article_chunk.article_id)

        parent = MaterializedChunk(
            node_id=parent_node_id,
            chunk_id=parent_chunk_id,
            parent_node_id="",  # Artigo não tem pai
            span_id=article_chunk.article_id,
            device_type=DeviceType.ARTICLE,
            chunk_level=ChunkLevel.ARTICLE,
            text=article_chunk.text,
            retrieval_text=article_retrieval_text,
            document_id=self.document_id,
            tipo_documento=self.tipo_documento,
            numero=self.numero,
            ano=self.ano,
            article_number=article_chunk.article_number,
            citations=article_chunk.citations,
            metadata=self.metadata,
            canonical_start=art_start,
            canonical_end=art_end,
            canonical_hash=art_hash,
        )
        # Validação obrigatória - aborta se node_id inconsistente
        parent.validate()
        chunks.append(parent)

        if not include_children:
            return chunks

        # 2. Chunks filhos (PARAGRAPH)
        for par_id in article_chunk.paragrafo_ids:
            par_span = parsed_doc.get_span(par_id)
            if not par_span:
                continue

            child_chunk_id = f"{self.document_id}#{par_id}"
            child_node_id = f"leis:{child_chunk_id}"

            # Constrói retrieval_text determinístico para o parágrafo
            par_retrieval_text = build_retrieval_text(
                document_id=self.document_id,
                article_number=article_chunk.article_number,
                device_type=DeviceType.PARAGRAPH,
                text=par_span.text,
                parent_text=article_caput,  # Caput do artigo pai
                span_id=par_id,
            )

            # PR13 STRICT: Resolve offsets para o parágrafo
            par_start, par_end, par_hash = self._resolve_child_offset_strict(
                span_id=par_id,
                device_type="paragraph",
                chunk_text=par_span.text,
                parent_start=art_start,
                parent_end=art_end,
            )

            child = MaterializedChunk(
                node_id=child_node_id,
                chunk_id=child_chunk_id,
                parent_node_id=parent_node_id,  # node_id do artigo pai
                span_id=par_id,
                device_type=DeviceType.PARAGRAPH,
                chunk_level=ChunkLevel.DEVICE,
                text=par_span.text,
                parent_text=article_caput,  # Caput do artigo pai
                retrieval_text=par_retrieval_text,
                document_id=self.document_id,
                tipo_documento=self.tipo_documento,
                numero=self.numero,
                ano=self.ano,
                article_number=article_chunk.article_number,
                citations=[par_id],
                metadata=self.metadata,
                canonical_start=par_start,
                canonical_end=par_end,
                canonical_hash=par_hash,
            )
            # Validação obrigatória - aborta se node_id inconsistente
            child.validate()
            chunks.append(child)

        # 3. Chunks filhos (INCISO)
        for inc_id in article_chunk.inciso_ids:
            inc_span = parsed_doc.get_span(inc_id)
            if not inc_span:
                continue

            child_chunk_id = f"{self.document_id}#{inc_id}"
            child_node_id = f"leis:{child_chunk_id}"

            # Reconstrói texto do inciso com alíneas
            inc_text = self._reconstruct_inciso_text(inc_id, parsed_doc)
            inc_citations = self._get_inciso_citations(inc_id, parsed_doc)

            # Determina parent_node_id correto: parágrafo (se existir) ou artigo
            if inc_span.parent_id and inc_span.parent_id.startswith("PAR-"):
                correct_parent_node_id = f"leis:{self.document_id}#{inc_span.parent_id}"
            else:
                correct_parent_node_id = parent_node_id  # Fallback para artigo

            # Constrói retrieval_text determinístico para o inciso
            inc_retrieval_text = build_retrieval_text(
                document_id=self.document_id,
                article_number=article_chunk.article_number,
                device_type=DeviceType.INCISO,
                text=inc_text,
                parent_text=article_caput,  # Caput do artigo pai
                span_id=inc_id,
            )

            # PR13 STRICT: Resolve offsets para o inciso
            # IMPORTANTE: Sempre busca no range do ARTIGO, não do parágrafo!
            # Motivo: O texto do parágrafo (do SpanParser) contém apenas o caput,
            # não os incisos. Os incisos estão no texto do artigo, após o parágrafo.
            # O parent_node_id continua apontando para o parágrafo (hierarquia lógica),
            # mas o offset é resolvido dentro do artigo (posição física no texto).

            # NOTA: Usa inc_span.text (original) para busca, não inc_text (reconstruído com alíneas)
            # O texto reconstruído inclui formatação que não existe no canonical_text
            inc_start, inc_end, inc_hash = self._resolve_child_offset_strict(
                span_id=inc_id,
                device_type="inciso",
                chunk_text=inc_span.text,  # Texto original do span, não reconstruído
                parent_start=art_start,    # Sempre busca no artigo
                parent_end=art_end,
            )

            child = MaterializedChunk(
                node_id=child_node_id,
                chunk_id=child_chunk_id,
                parent_node_id=correct_parent_node_id,  # node_id do pai (PAR ou ART)
                span_id=inc_id,
                device_type=DeviceType.INCISO,
                chunk_level=ChunkLevel.DEVICE,
                text=inc_text,
                parent_text=article_caput,  # Caput do artigo pai
                retrieval_text=inc_retrieval_text,
                document_id=self.document_id,
                tipo_documento=self.tipo_documento,
                numero=self.numero,
                ano=self.ano,
                article_number=article_chunk.article_number,
                citations=inc_citations,
                metadata=self.metadata,
                canonical_start=inc_start,
                canonical_end=inc_end,
                canonical_hash=inc_hash,
            )
            # Validação obrigatória - aborta se node_id inconsistente
            child.validate()
            chunks.append(child)

        return chunks

    def _reconstruct_inciso_text(self, inc_id: str, parsed_doc) -> str:
        """Reconstrói texto do inciso incluindo alíneas."""
        inc_span = parsed_doc.get_span(inc_id)
        if not inc_span:
            return ""

        lines = [inc_span.text]

        # Adiciona alíneas
        for child in parsed_doc.get_children(inc_id):
            lines.append(f"  {child.text}")

        return "\n".join(lines)

    def _get_inciso_citations(self, inc_id: str, parsed_doc) -> list[str]:
        """Obtém lista de citations para o inciso (inclui alíneas)."""
        citations = [inc_id]

        for child in parsed_doc.get_children(inc_id):
            citations.append(child.span_id)

        return citations

    def _get_canonical_offsets(self, span_id: str) -> tuple[int, int, str]:
        """
        Obtém offsets canônicos para um span_id (PR13).

        Busca no offsets_map e retorna:
        - (start, end, hash) se encontrado
        - (-1, -1, "") se não encontrado ou offsets_map vazio

        Args:
            span_id: ID do span (ex: "ART-005", "PAR-005-1")

        Returns:
            Tupla (canonical_start, canonical_end, canonical_hash)
        """
        if not self.offsets_map or span_id not in self.offsets_map:
            return (-1, -1, "")

        start, end = self.offsets_map[span_id]
        return (start, end, self.canonical_hash)

    def _resolve_child_offset_strict(
        self,
        span_id: str,
        device_type: str,
        chunk_text: str,
        parent_start: int,
        parent_end: int,
    ) -> tuple[int, int, str]:
        """
        Resolve offsets de um chunk filho de forma determinística (PR13 STRICT).

        Busca chunk_text dentro do range do pai no canonical_text.
        - Se não encontrado → OffsetResolutionError (ABORT)
        - Se múltiplas ocorrências → OffsetResolutionError (ABORT)
        - Se encontrado exatamente uma vez → retorna (start, end, hash)

        Args:
            span_id: ID do span (ex: "PAR-005-1")
            device_type: Tipo do dispositivo (ex: "paragraph", "inciso")
            chunk_text: Texto do chunk a localizar
            parent_start: Offset início do pai no canonical_text
            parent_end: Offset fim do pai no canonical_text

        Returns:
            Tupla (canonical_start, canonical_end, canonical_hash)

        Raises:
            OffsetResolutionError: Se chunk não encontrado ou ambíguo
        """
        # Primeiro, verifica se já existe no offsets_map (do SpanParser)
        if self.offsets_map and span_id in self.offsets_map:
            start, end = self.offsets_map[span_id]
            # Armazena no cache para uso por filhos
            self._resolved_child_offsets[span_id] = (start, end)
            return (start, end, self.canonical_hash)

        # Verifica se temos canonical_text para resolução
        if not self.canonical_text:
            raise OffsetResolutionError(
                f"canonical_text não fornecido para resolver offsets de '{span_id}'",
                document_id=self.document_id,
                span_id=span_id,
                device_type=device_type,
                reason="NO_CANONICAL_TEXT",
            )

        # Resolve usando busca determinística no canonical_text
        child_start, child_end = resolve_child_offsets(
            canonical_text=self.canonical_text,
            parent_start=parent_start,
            parent_end=parent_end,
            chunk_text=chunk_text,
            document_id=self.document_id,
            span_id=span_id,
            device_type=device_type,
        )

        # Armazena no cache para uso por filhos (ex: inciso busca offset do parágrafo pai)
        self._resolved_child_offsets[span_id] = (child_start, child_end)

        logger.debug(
            f"PR13 STRICT: Offset resolvido para {span_id} ({device_type}): "
            f"[{child_start}:{child_end}]"
        )

        return (child_start, child_end, self.canonical_hash)

    def _get_resolved_child_offset(self, span_id: str) -> Optional[tuple[int, int]]:
        """
        Obtém offset já resolvido de um filho (do cache).

        Usado para hierarquia: quando um inciso está sob um parágrafo,
        precisamos do range do parágrafo para buscar o inciso.

        Args:
            span_id: ID do span (ex: "PAR-005-1")

        Returns:
            Tupla (start, end) se encontrado no cache, None caso contrário
        """
        # Primeiro verifica no cache de resolvidos
        if span_id in self._resolved_child_offsets:
            return self._resolved_child_offsets[span_id]

        # Fallback para offsets_map do SpanParser
        if self.offsets_map and span_id in self.offsets_map:
            return self.offsets_map[span_id]

        return None

    # ==========================================================================
    # SPLITTING DE ARTIGOS GRANDES
    # ==========================================================================

    def _should_split_article(self, text: str) -> bool:
        """Verifica se o artigo precisa ser splitado."""
        return len(text) > SPLIT_THRESHOLD_CHARS

    def _find_best_split_point(self, text: str, target_pos: int, window: int = 500) -> int:
        """
        Encontra o melhor ponto de split próximo à posição alvo.

        Ordem de preferência:
        1. Marcadores jurídicos (§, inciso, alínea)
        2. Quebras de parágrafo
        3. Fim de sentença

        Args:
            text: Texto a ser splitado
            target_pos: Posição alvo para o split
            window: Janela de busca ao redor da posição alvo

        Returns:
            Posição do melhor ponto de split
        """
        start = max(0, target_pos - window)
        end = min(len(text), target_pos + window)
        search_region = text[start:end]

        # 1. Tentar marcadores jurídicos
        for pattern in LEGAL_MARKERS:
            matches = list(re.finditer(pattern, search_region))
            if matches:
                # Pega o match mais próximo do target
                best_match = min(matches, key=lambda m: abs((start + m.start()) - target_pos))
                return start + best_match.start()

        # 2. Tentar quebra de parágrafo
        para_matches = list(PARAGRAPH_BREAK.finditer(search_region))
        if para_matches:
            best_match = min(para_matches, key=lambda m: abs((start + m.start()) - target_pos))
            return start + best_match.end()

        # 3. Tentar fim de sentença
        sent_matches = list(SENTENCE_END.finditer(search_region))
        if sent_matches:
            best_match = min(sent_matches, key=lambda m: abs((start + m.start()) - target_pos))
            return start + best_match.end()

        # 4. Fallback: posição alvo exata (evita cortar no meio de palavra)
        space_pos = text.rfind(' ', target_pos - 50, target_pos + 50)
        if space_pos != -1:
            return space_pos + 1

        return target_pos

    def _split_large_article(self, text: str, article_id: str) -> list[dict]:
        """
        Divide artigo grande em subchunks com overlap.

        Args:
            text: Texto completo do artigo
            article_id: ID do artigo (ex: ART-005)

        Returns:
            Lista de dicts com:
            - part_num: Número da parte (1, 2, 3...)
            - span_id: ID do subchunk (ex: ART-005-P1)
            - text: Texto do subchunk
            - start_pos: Posição inicial no texto original
            - end_pos: Posição final no texto original
        """
        parts = []
        pos = 0
        part_num = 1

        while pos < len(text):
            # Calcula posição final alvo
            end_target = pos + TARGET_CHUNK_CHARS

            if end_target >= len(text):
                # Último chunk - pega o resto
                chunk_text = text[pos:]
            else:
                # Encontra melhor ponto de split
                split_pos = self._find_best_split_point(text, end_target)
                chunk_text = text[pos:split_pos]

            # Evita chunks muito pequenos no final
            if len(chunk_text.strip()) < 100 and parts:
                # Anexa ao chunk anterior
                parts[-1]["text"] += "\n" + chunk_text
                parts[-1]["end_pos"] = len(text)
                break

            part_span_id = f"{article_id}-P{part_num}"

            parts.append({
                "part_num": part_num,
                "span_id": part_span_id,
                "text": chunk_text.strip(),
                "start_pos": pos,
                "end_pos": pos + len(chunk_text),
            })

            # Avança posição com overlap
            if end_target >= len(text):
                break

            pos = self._find_best_split_point(text, end_target) - OVERLAP_CHARS
            pos = max(pos, parts[-1]["end_pos"] - OVERLAP_CHARS)  # Garante progresso
            part_num += 1

            # Segurança: limite de 50 partes
            if part_num > 50:
                logger.warning(
                    f"Artigo {article_id} excedeu limite de 50 partes, "
                    f"truncando em {len(text)} chars"
                )
                break

        logger.info(
            f"Artigo {article_id} splitado em {len(parts)} partes "
            f"(original: {len(text)} chars)"
        )

        return parts

    def _materialize_large_article(
        self,
        article_chunk,
        parsed_doc,
        include_children: bool = True
    ) -> list[MaterializedChunk]:
        """
        Materializa artigo grande em pai canônico + subchunks (PART).

        O pai canônico é criado com `_skip_milvus_index=True` para indicar
        que não deve ser indexado diretamente (apenas os PARTs são indexados).

        Args:
            article_chunk: ArticleChunk com texto grande
            parsed_doc: Documento parseado
            include_children: Se True, também gera PAR/INC como antes

        Returns:
            Lista com: [pai_canônico, part1, part2, ..., paragraphs, incisos]
        """
        chunks = []

        # Texto do artigo (caput) - será usado como parent_text para filhos
        article_caput = article_chunk.text

        # 1. Pai canônico (NÃO indexado no Milvus, apenas para navegação)
        parent_chunk_id = f"{self.document_id}#{article_chunk.article_id}"
        parent_node_id = f"leis:{parent_chunk_id}"

        # Constrói retrieval_text determinístico para o artigo pai
        article_retrieval_text = build_retrieval_text(
            document_id=self.document_id,
            article_number=article_chunk.article_number,
            device_type=DeviceType.ARTICLE,
            text=article_caput,
            parent_text="",  # Artigo não tem pai
            span_id=article_chunk.article_id,
        )

        # PR13: Obtém offsets canônicos para o artigo
        art_start, art_end, art_hash = self._get_canonical_offsets(article_chunk.article_id)

        parent = MaterializedChunk(
            node_id=parent_node_id,
            chunk_id=parent_chunk_id,
            parent_node_id="",  # Artigo não tem pai
            span_id=article_chunk.article_id,
            device_type=DeviceType.ARTICLE,
            chunk_level=ChunkLevel.ARTICLE,
            text=article_chunk.text,
            retrieval_text=article_retrieval_text,
            document_id=self.document_id,
            tipo_documento=self.tipo_documento,
            numero=self.numero,
            ano=self.ano,
            article_number=article_chunk.article_number,
            citations=article_chunk.citations,
            metadata=self.metadata,
            canonical_start=art_start,
            canonical_end=art_end,
            canonical_hash=art_hash,
        )
        parent.validate()
        # Marca que este chunk não deve ser indexado no Milvus
        parent._skip_milvus_index = True
        chunks.append(parent)

        # 2. Subchunks (PART) - estes SÃO indexados
        parts = self._split_large_article(article_chunk.text, article_chunk.article_id)

        for part in parts:
            part_chunk_id = f"{self.document_id}#{part['span_id']}"
            part_node_id = f"leis:{part_chunk_id}"

            # Constrói retrieval_text determinístico para a parte
            part_retrieval_text = build_retrieval_text(
                document_id=self.document_id,
                article_number=article_chunk.article_number,
                device_type=DeviceType.PART,
                text=part["text"],
                parent_text=article_caput,  # Caput do artigo pai
                span_id=part["span_id"],
            )

            part_chunk = MaterializedChunk(
                node_id=part_node_id,
                chunk_id=part_chunk_id,
                parent_node_id=parent_node_id,  # node_id do pai canônico
                span_id=part["span_id"],
                device_type=DeviceType.PART,
                chunk_level=ChunkLevel.DEVICE,  # Filhos são DEVICE level
                text=part["text"],
                parent_text=article_caput,  # Caput do artigo pai
                retrieval_text=part_retrieval_text,
                document_id=self.document_id,
                tipo_documento=self.tipo_documento,
                numero=self.numero,
                ano=self.ano,
                article_number=article_chunk.article_number,
                # Citations inclui o artigo pai + o span da parte
                citations=[article_chunk.article_id, part["span_id"]],
                metadata=self.metadata,
            )
            part_chunk.validate()
            chunks.append(part_chunk)

        # 3. Filhos estruturais (PARAGRAPH, INCISO) - se existirem
        if include_children:
            # Parágrafos
            for par_id in article_chunk.paragrafo_ids:
                par_span = parsed_doc.get_span(par_id)
                if not par_span:
                    continue

                child_chunk_id = f"{self.document_id}#{par_id}"
                child_node_id = f"leis:{child_chunk_id}"

                # Constrói retrieval_text determinístico para o parágrafo
                par_retrieval_text = build_retrieval_text(
                    document_id=self.document_id,
                    article_number=article_chunk.article_number,
                    device_type=DeviceType.PARAGRAPH,
                    text=par_span.text,
                    parent_text=article_caput,
                    span_id=par_id,
                )

                # PR13 STRICT: Resolve offsets para o parágrafo
                par_start, par_end, par_hash = self._resolve_child_offset_strict(
                    span_id=par_id,
                    device_type="paragraph",
                    chunk_text=par_span.text,
                    parent_start=art_start,
                    parent_end=art_end,
                )

                child = MaterializedChunk(
                    node_id=child_node_id,
                    chunk_id=child_chunk_id,
                    parent_node_id=parent_node_id,  # node_id do artigo pai
                    span_id=par_id,
                    device_type=DeviceType.PARAGRAPH,
                    chunk_level=ChunkLevel.DEVICE,
                    text=par_span.text,
                    parent_text=article_caput,
                    retrieval_text=par_retrieval_text,
                    document_id=self.document_id,
                    tipo_documento=self.tipo_documento,
                    numero=self.numero,
                    ano=self.ano,
                    article_number=article_chunk.article_number,
                    citations=[par_id],
                    metadata=self.metadata,
                    canonical_start=par_start,
                    canonical_end=par_end,
                    canonical_hash=par_hash,
                )
                child.validate()
                chunks.append(child)

            # Incisos
            for inc_id in article_chunk.inciso_ids:
                inc_span = parsed_doc.get_span(inc_id)
                if not inc_span:
                    continue

                child_chunk_id = f"{self.document_id}#{inc_id}"
                child_node_id = f"leis:{child_chunk_id}"

                inc_text = self._reconstruct_inciso_text(inc_id, parsed_doc)
                inc_citations = self._get_inciso_citations(inc_id, parsed_doc)

                # Determina parent_node_id correto: parágrafo (se existir) ou artigo
                if inc_span.parent_id and inc_span.parent_id.startswith("PAR-"):
                    correct_parent_node_id = f"leis:{self.document_id}#{inc_span.parent_id}"
                else:
                    correct_parent_node_id = parent_node_id  # Fallback para artigo

                # Constrói retrieval_text determinístico para o inciso
                inc_retrieval_text = build_retrieval_text(
                    document_id=self.document_id,
                    article_number=article_chunk.article_number,
                    device_type=DeviceType.INCISO,
                    text=inc_text,
                    parent_text=article_caput,
                    span_id=inc_id,
                )

                # PR13 STRICT: Resolve offsets para o inciso
                # IMPORTANTE: Sempre busca no range do ARTIGO, não do parágrafo!
                # O texto do parágrafo contém apenas o caput, não os incisos.
                # NOTA: Usa inc_span.text (original) para busca, não inc_text (reconstruído)
                inc_start, inc_end, inc_hash = self._resolve_child_offset_strict(
                    span_id=inc_id,
                    device_type="inciso",
                    chunk_text=inc_span.text,  # Texto original, não reconstruído
                    parent_start=art_start,    # Sempre busca no artigo
                    parent_end=art_end,
                )

                child = MaterializedChunk(
                    node_id=child_node_id,
                    chunk_id=child_chunk_id,
                    parent_node_id=correct_parent_node_id,  # node_id do pai (PAR ou ART)
                    span_id=inc_id,
                    device_type=DeviceType.INCISO,
                    chunk_level=ChunkLevel.DEVICE,
                    text=inc_text,
                    parent_text=article_caput,
                    retrieval_text=inc_retrieval_text,
                    document_id=self.document_id,
                    tipo_documento=self.tipo_documento,
                    numero=self.numero,
                    ano=self.ano,
                    article_number=article_chunk.article_number,
                    citations=inc_citations,
                    metadata=self.metadata,
                    canonical_start=inc_start,
                    canonical_end=inc_end,
                    canonical_hash=inc_hash,
                )
                child.validate()
                chunks.append(child)

        logger.info(
            f"Artigo grande {article_chunk.article_id} materializado: "
            f"1 pai + {len(parts)} partes + "
            f"{len(article_chunk.paragrafo_ids)} PAR + "
            f"{len(article_chunk.inciso_ids)} INC"
        )

        return chunks

    def materialize_all(
        self,
        article_chunks: list,  # list[ArticleChunk]
        parsed_doc,
        include_children: bool = True
    ) -> list[MaterializedChunk]:
        """
        Materializa todos os ArticleChunks.

        Returns:
            Lista completa de MaterializedChunk
        """
        all_chunks = []

        for article_chunk in article_chunks:
            chunks = self.materialize_article(
                article_chunk, parsed_doc, include_children
            )
            all_chunks.extend(chunks)

        # Log estatísticas
        result = MaterializationResult.from_chunks(all_chunks, self.document_id)
        logger.info(
            f"Materialização completa para {self.document_id}: "
            f"{result.total_chunks} chunks total "
            f"({result.article_chunks} artigos, {result.part_chunks} partes, "
            f"{result.paragraph_chunks} parágrafos, {result.inciso_chunks} incisos)"
        )
        if result.split_articles > 0:
            logger.info(
                f"  → {result.split_articles} artigos grandes foram splitados em partes"
            )

        return all_chunks

    def materialize_all_with_result(
        self,
        article_chunks: list,  # list[ArticleChunk]
        parsed_doc,
        include_children: bool = True
    ) -> "MaterializationResult":
        """
        Materializa todos os ArticleChunks e retorna resultado com estatísticas.

        Returns:
            MaterializationResult com chunks e estatísticas
        """
        chunks = self.materialize_all(article_chunks, parsed_doc, include_children)
        return MaterializationResult.from_chunks(chunks, self.document_id)


@dataclass
class MaterializationResult:
    """Resultado da materialização."""

    chunks: list[MaterializedChunk] = field(default_factory=list)

    # Estatísticas
    total_chunks: int = 0
    article_chunks: int = 0
    paragraph_chunks: int = 0
    inciso_chunks: int = 0
    part_chunks: int = 0  # Subchunks de artigos grandes
    split_articles: int = 0  # Quantidade de artigos que foram splitados

    # Metadados
    document_id: str = ""
    ingestion_timestamp: str = ""

    def summary(self) -> dict:
        return {
            "document_id": self.document_id,
            "total_chunks": self.total_chunks,
            "breakdown": {
                "articles": self.article_chunks,
                "paragraphs": self.paragraph_chunks,
                "incisos": self.inciso_chunks,
                "parts": self.part_chunks,
            },
            "split_articles": self.split_articles,
            "ingestion_timestamp": self.ingestion_timestamp,
        }

    @classmethod
    def from_chunks(
        cls,
        chunks: list[MaterializedChunk],
        document_id: str = ""
    ) -> "MaterializationResult":
        """Cria MaterializationResult a partir de lista de chunks."""
        article_chunks = 0
        paragraph_chunks = 0
        inciso_chunks = 0
        part_chunks = 0
        split_articles = set()

        for chunk in chunks:
            if chunk.device_type == DeviceType.ARTICLE:
                article_chunks += 1
                # Verifica se é artigo splitado (tem _skip_milvus_index)
                if getattr(chunk, '_skip_milvus_index', False):
                    split_articles.add(chunk.chunk_id)
            elif chunk.device_type == DeviceType.PARAGRAPH:
                paragraph_chunks += 1
            elif chunk.device_type == DeviceType.INCISO:
                inciso_chunks += 1
            elif chunk.device_type == DeviceType.PART:
                part_chunks += 1

        return cls(
            chunks=chunks,
            total_chunks=len(chunks),
            article_chunks=article_chunks,
            paragraph_chunks=paragraph_chunks,
            inciso_chunks=inciso_chunks,
            part_chunks=part_chunks,
            split_articles=len(split_articles),
            document_id=document_id,
            ingestion_timestamp=datetime.utcnow().isoformat(),
        )
