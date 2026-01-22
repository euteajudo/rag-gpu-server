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
    │  │ parent_chunk_id: ""  ← Artigo não tem pai                       │  │
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
    │  │ parent_chunk_id:│  │ parent_chunk_id:│                             │
    │  │ IN-65-2021#     │  │ IN-65-2021#     │                             │
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
    │  3. Sistema recupera chunk PAI via parent_chunk_id                      │
    │     → Busca: parent_chunk_id == "IN-65-2021#ART-005"                    │
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
| parent_chunk_id    | str           | ID do pai ("" para artigos)            |
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

from .chunk_models import LegalChunk, ChunkLevel

logger = logging.getLogger(__name__)


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
    parent_chunk_id: str    # Ex: "" para artigos, "IN-65-2021#ART-005" para filhos
    span_id: str            # Ex: "ART-005", "PAR-005-1", "INC-005-I"

    # Tipo
    device_type: DeviceType
    chunk_level: ChunkLevel

    # Conteúdo
    text: str
    enriched_text: str = ""

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
            "parent_chunk_id": self.parent_chunk_id,
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

            # Proveniência
            **self.metadata.to_dict(),
        }


class ChunkMaterializer:
    """
    Materializa ArticleChunk em chunks indexáveis com parent-child.

    Gera:
    - 1 chunk ARTICLE (pai) com texto completo
    - N chunks PARAGRAPH (filhos) para cada parágrafo
    - M chunks INCISO (filhos) para cada inciso
    """

    def __init__(
        self,
        document_id: str,
        tipo_documento: str = "",
        numero: str = "",
        ano: int = 0,
        metadata: Optional[ChunkMetadata] = None
    ):
        self.document_id = document_id
        self.tipo_documento = tipo_documento
        self.numero = numero
        self.ano = ano
        self.metadata = metadata or ChunkMetadata()

    def materialize_article(
        self,
        article_chunk,  # ArticleChunk do article_orchestrator
        parsed_doc,     # ParsedDocument para reconstruir texto
        include_children: bool = True
    ) -> list[MaterializedChunk]:
        """
        Materializa um ArticleChunk em chunks pai e filhos.

        Args:
            article_chunk: ArticleChunk extraído
            parsed_doc: Documento parseado (para reconstruir texto dos filhos)
            include_children: Se True, gera chunks para PAR/INC também

        Returns:
            Lista de MaterializedChunk (1 pai + N filhos)
        """
        chunks = []

        # 1. Chunk pai (ARTICLE)
        parent_chunk_id = f"{self.document_id}#{article_chunk.article_id}"
        parent_node_id = f"leis:{parent_chunk_id}"

        parent = MaterializedChunk(
            node_id=parent_node_id,
            chunk_id=parent_chunk_id,
            parent_chunk_id="",  # Artigo não tem pai
            span_id=article_chunk.article_id,
            device_type=DeviceType.ARTICLE,
            chunk_level=ChunkLevel.ARTICLE,
            text=article_chunk.text,
            document_id=self.document_id,
            tipo_documento=self.tipo_documento,
            numero=self.numero,
            ano=self.ano,
            article_number=article_chunk.article_number,
            citations=article_chunk.citations,
            metadata=self.metadata,
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

            child = MaterializedChunk(
                node_id=child_node_id,
                chunk_id=child_chunk_id,
                parent_chunk_id=parent_chunk_id,
                span_id=par_id,
                device_type=DeviceType.PARAGRAPH,
                chunk_level=ChunkLevel.DEVICE,
                text=par_span.text,
                document_id=self.document_id,
                tipo_documento=self.tipo_documento,
                numero=self.numero,
                ano=self.ano,
                article_number=article_chunk.article_number,
                citations=[par_id],
                metadata=self.metadata,
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

            child = MaterializedChunk(
                node_id=child_node_id,
                chunk_id=child_chunk_id,
                parent_chunk_id=parent_chunk_id,
                span_id=inc_id,
                device_type=DeviceType.INCISO,
                chunk_level=ChunkLevel.DEVICE,
                text=inc_text,
                document_id=self.document_id,
                tipo_documento=self.tipo_documento,
                numero=self.numero,
                ano=self.ano,
                article_number=article_chunk.article_number,
                citations=inc_citations,
                metadata=self.metadata,
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

        return all_chunks


@dataclass
class MaterializationResult:
    """Resultado da materialização."""

    chunks: list[MaterializedChunk] = field(default_factory=list)

    # Estatísticas
    total_chunks: int = 0
    article_chunks: int = 0
    paragraph_chunks: int = 0
    inciso_chunks: int = 0

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
            },
            "ingestion_timestamp": self.ingestion_timestamp,
        }
