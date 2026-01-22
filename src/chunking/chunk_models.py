"""
Modelos de Dados para Chunks de Documentos Legais.

Este módulo define as estruturas de dados fundamentais para representação
de chunks de documentos legais brasileiros, prontos para indexação no Milvus
com suporte a busca híbrida usando embeddings BGE-M3.

Arquitetura de Dados:
====================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                        LEGALCHUNK DATACLASS                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                    IDs Hierárquicos                          │   │
    │  │  chunk_id ──────► "IN-SEGES-58-2022#CAP-I#ART-3"             │   │
    │  │  parent_id ─────► "IN-SEGES-58-2022#CAP-I"                   │   │
    │  │  chunk_level ───► ARTICLE (ou CHAPTER, DEVICE)               │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                                                                     │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                      Conteúdo                                │   │
    │  │  text ──────────► "Art. 3º Para fins do disposto nesta..."   │   │
    │  │  enriched_text ─► "[CONTEXTO: Este artigo...] Art. 3º..."   │   │
    │  │  context_header ► "Este artigo da IN 58/2022 define..."      │   │
    │  │  thesis_text ───► "Estabelece definições de termos..."       │   │
    │  │  thesis_type ───► "definicao"                                │   │
    │  │  synthetic_questions ► "O que é ETP?\nQuem é requisitante?" │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                                                                     │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                 Embeddings BGE-M3                            │   │
    │  │  dense_vector ──► [0.023, -0.156, ...] (1024 dimensões)      │   │
    │  │  thesis_vector ─► [0.045, 0.089, ...]  (1024 dimensões)      │   │
    │  │  sparse_vector ─► {12345: 0.89, 67890: 0.45, ...}            │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Hierarquia de Níveis (ChunkLevel):
=================================

    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │  DOCUMENT (0)      Documento inteiro (raro, apenas contexto)       │
    │       │                                                            │
    │       └── CHAPTER (1)      Capítulo completo                       │
    │              │                                                     │
    │              └── ARTICLE (2)      Artigo (principal, padrão)       │
    │                     │                                              │
    │                     └── DEVICE (3)      § ou inciso isolado        │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

Tipos de Tese (ThesisType):
==========================

    | Tipo        | Descrição                      | Exemplo                     |
    |-------------|--------------------------------|-----------------------------|
    | definicao   | Define conceitos ou termos     | "considera-se licitação..." |
    | procedimento| Descreve processos ou etapas   | "o processo seguirá..."     |
    | prazo       | Define prazos ou limites       | "no prazo de 30 dias..."    |
    | requisito   | Estabelece condições           | "são requisitos: I -..."    |
    | competencia | Define competências            | "compete ao órgão..."       |
    | vedacao     | Proíbe algo                    | "é vedado..."               |
    | excecao     | Casos especiais ou dispensas   | "exceto quando..."          |
    | sancao      | Sanções ou penalidades         | "sujeito à multa..."        |
    | disposicao  | Disposição geral (default)     | Qualquer outro              |

Embeddings BGE-M3 (Busca Híbrida):
=================================

    O BGE-M3 gera dois tipos de vetores complementares:

    ┌─────────────────────────────────────────────────────────────────┐
    │  DENSE VECTOR (1024 dimensões)                                  │
    │  - Captura significado semântico                                │
    │  - Encontra documentos conceptualmente similares                │
    │  - "ETP" ≈ "Estudo Técnico Preliminar" ≈ "planejamento"         │
    ├─────────────────────────────────────────────────────────────────┤
    │  THESIS VECTOR (1024 dimensões)                                 │
    │  - Embedding do resumo/tese do artigo                           │
    │  - Foco na essência do dispositivo                              │
    │  - Usado em busca focada (20% do peso)                          │
    ├─────────────────────────────────────────────────────────────────┤
    │  SPARSE VECTOR (learned, variável)                              │
    │  - Modelo aprendido (superior ao BM25)                          │
    │  - Captura sinônimos e variações linguísticas                   │
    │  - "requisitante" ≈ "demandante" ≈ "solicitante"                │
    │  - Formato: {token_id: peso, ...}                               │
    └─────────────────────────────────────────────────────────────────┘

    Busca Híbrida no Milvus:
    ┌─────────────────────────────────────────────────────────────────┐
    │  dense_vector  ───► 50% do score (semântica geral)              │
    │  sparse_vector ───► 30% do score (termos específicos)           │
    │  thesis_vector ───► 20% do score (essência do artigo)           │
    └─────────────────────────────────────────────────────────────────┘

Campos do LegalChunk:
====================

    | Categoria       | Campo              | Tipo          | Descrição                  |
    |-----------------|--------------------|---------------|----------------------------|
    | **IDs**         | chunk_id           | str           | ID hierárquico único       |
    |                 | parent_id          | str           | ID do chunk pai            |
    |                 | chunk_index        | int           | Índice sequencial          |
    |                 | chunk_level        | ChunkLevel    | Nível hierárquico          |
    | **Conteúdo**    | text               | str           | Texto original             |
    |                 | enriched_text      | str           | Contexto + texto + perguntas|
    | **Enriquecimento** | context_header  | str           | Frase de contexto (LLM)    |
    |                 | thesis_text        | str           | Resumo do dispositivo (LLM)|
    |                 | thesis_type        | str           | Tipo de conteúdo (LLM)     |
    |                 | synthetic_questions| str           | Perguntas relacionadas (LLM)|
    | **Hierarquia**  | document_id        | str           | ID do documento            |
    |                 | tipo_documento     | str           | LEI, DECRETO, IN, etc      |
    |                 | numero             | str           | Número do documento        |
    |                 | ano                | int           | Ano do documento           |
    |                 | chapter_number     | str           | Número do capítulo         |
    |                 | chapter_title      | str           | Título do capítulo         |
    |                 | article_number     | str           | Número do artigo           |
    |                 | article_title      | str           | Título do artigo           |
    | **Estrutura**   | has_items          | bool          | Tem incisos?               |
    |                 | has_paragraphs     | bool          | Tem parágrafos?            |
    |                 | item_count         | int           | Qtd de incisos             |
    |                 | paragraph_count    | int           | Qtd de parágrafos          |
    | **Metadados**   | token_count        | int           | Contagem de tokens         |
    |                 | char_start         | int           | Posição inicial            |
    |                 | char_end           | int           | Posição final              |
    | **Embeddings**  | dense_vector       | list[float]   | BGE-M3 denso (1024d)       |
    |                 | thesis_vector      | list[float]   | Embedding da tese (1024d)  |
    |                 | sparse_vector      | dict[int,float]| Sparse aprendido          |

Exemplo de Uso:
==============

    ```python
    from chunking import LegalChunk, ChunkLevel, ThesisType, ChunkingResult

    # 1. Criar chunk manualmente
    chunk = LegalChunk(
        chunk_id="IN-58-2022#CAP-I#ART-3",
        parent_id="IN-58-2022#CAP-I",
        chunk_index=3,
        chunk_level=ChunkLevel.ARTICLE,
        text="Art. 3º Para fins do disposto nesta Instrução Normativa...",
        enriched_text="[CONTEXTO: Este artigo define conceitos...] Art. 3º...",
        context_header="Este artigo da IN 58/2022 define os conceitos básicos",
        thesis_text="Estabelece definições de termos técnicos",
        thesis_type="definicao",
        synthetic_questions="O que é ETP?\\nO que é contratação direta?",
        document_id="IN-58-2022",
        tipo_documento="INSTRUÇÃO NORMATIVA",
        numero="58",
        ano=2022,
        chapter_number="I",
        chapter_title="DISPOSIÇÕES GERAIS",
        article_number="3",
        has_items=True,
        item_count=15,
        token_count=450,
    )

    # 2. Converter para dicionário (Milvus)
    milvus_data = chunk.to_dict()

    # 3. Serializar para JSON (sem embeddings)
    json_str = chunk.to_json()

    # 4. Recriar de dicionário
    chunk_restored = LegalChunk.from_dict(milvus_data)

    # 5. Trabalhar com resultado de chunking
    result = ChunkingResult(
        chunks=[chunk],
        document_id="IN-58-2022",
        total_chunks=25,
    )
    print(result.summary())
    # {
    #     "document_id": "IN-58-2022",
    #     "total_chunks": 1,
    #     "chunks_by_level": {"ARTICLE": 1},
    #     "chunks_by_type": {"definicao": 1},
    #     ...
    # }
    ```

ChunkingResult - Resultado do Processo:
======================================

    ┌────────────────────────────────────────────────────────────────────┐
    │  ChunkingResult                                                    │
    │  ├── chunks: list[LegalChunk]  ──► Lista de chunks gerados         │
    │  ├── document_id: str          ──► ID do documento processado      │
    │  ├── total_chunks: int         ──► Total de chunks criados         │
    │  ├── total_tokens: int         ──► Soma de tokens                  │
    │  ├── processing_time_seconds   ──► Tempo de processamento          │
    │  ├── errors: list[str]         ──► Erros encontrados               │
    │  └── warnings: list[str]       ──► Avisos                          │
    │                                                                    │
    │  Método summary():                                                 │
    │  ├── chunks_by_level ──► {"ARTICLE": 15, "DEVICE": 20}             │
    │  └── chunks_by_type  ──► {"definicao": 5, "procedimento": 10}      │
    └────────────────────────────────────────────────────────────────────┘

Integração com Outros Módulos:
=============================

    - law_chunker.py: LawChunker gera LegalChunks a partir de LegalDocument
    - chunk_materializer.py: ChunkMaterializer gera MaterializedChunks (v3)
    - enrichment/chunk_enricher.py: ChunkEnricher preenche campos de enriquecimento
    - embeddings/bge_m3.py: BGEM3Embedder preenche vetores
    - milvus/schema_v3.py: Schema que recebe LegalChunk.to_dict()

@author: Equipe VectorGov
@version: 1.0.0
@since: 21/12/2024
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class ChunkLevel(Enum):
    """Nível hierárquico do chunk."""
    DOCUMENT = 0    # Documento inteiro (raro, para contexto)
    CHAPTER = 1     # Capítulo inteiro (para contexto amplo)
    ARTICLE = 2     # Artigo completo (principal, retrieval padrão)
    DEVICE = 3      # Dispositivo isolado (§, inciso com alíneas)


class ThesisType(Enum):
    """Tipo de tese/conteúdo do dispositivo legal."""
    DEFINICAO = "definicao"         # Define conceitos
    PROCEDIMENTO = "procedimento"   # Estabelece procedimentos
    PRAZO = "prazo"                 # Define prazos
    REQUISITO = "requisito"         # Estabelece requisitos
    COMPETENCIA = "competencia"     # Define competências
    VEDACAO = "vedacao"             # Proíbe algo
    EXCECAO = "excecao"             # Estabelece exceções
    SANCAO = "sancao"               # Define sanções/penalidades
    DISPOSICAO = "disposicao"       # Disposição geral


@dataclass
class LegalChunk:
    """
    Chunk de documento legal pronto para Milvus.

    Contem todos os campos necessarios para busca hibrida com BGE-M3:
    - dense_vector: embedding denso do texto (1024d)
    - thesis_vector: embedding denso da tese/resumo (1024d)
    - sparse_vector: learned sparse BGE-M3 {token_id: weight}

    O sparse do BGE-M3 e superior ao BM25 porque:
    1. Treinado junto com dense - se complementam
    2. Aprende pesos semanticos, nao apenas frequencia
    3. Captura sinonimos: "requisitante" ~ "demandante" ~ "solicitante"

    Attributes:
        chunk_id: ID hierarquico unico (ex: "IN-SEGES-58-2022#CAP-I#ART-3")
        parent_id: ID do chunk pai (ex: "IN-SEGES-58-2022#CAP-I")
        chunk_index: Indice sequencial global no documento
        chunk_level: Nivel hierarquico (CHAPTER, ARTICLE, DEVICE)

        text: Texto original do dispositivo
        enriched_text: Contexto + texto + perguntas (para LLM)

        context_header: Frase contextualizando o dispositivo
        thesis_text: Resumo objetivo do que determina/define
        thesis_type: Classificacao do tipo de conteudo
        synthetic_questions: Perguntas que o chunk responde (separadas por \\n)

        document_id: ID unico do documento
        tipo_documento: LEI, DECRETO, INSTRUCAO NORMATIVA, etc
        numero: Numero do documento
        ano: Ano do documento
        chapter_number: Numero do capitulo (I, II, III)
        chapter_title: Titulo do capitulo
        article_number: Numero do artigo
        article_title: Titulo do artigo (se houver)

        has_items: Se o artigo tem incisos
        has_paragraphs: Se o artigo tem paragrafos
        item_count: Quantidade de incisos
        paragraph_count: Quantidade de paragrafos

        token_count: Contagem de tokens do texto
        char_start: Posicao inicial no documento original
        char_end: Posicao final no documento original

        dense_vector: Embedding denso BGE-M3 (1024d)
        thesis_vector: Embedding denso da tese (1024d)
        sparse_vector: Learned sparse BGE-M3 {token_id: weight}
    """

    # === IDs Hierárquicos ===
    chunk_id: str
    parent_id: str
    chunk_index: int
    chunk_level: ChunkLevel

    # === Conteúdo ===
    text: str
    enriched_text: str = ""

    # === Enriquecimento (gerado por LLM) ===
    context_header: str = ""
    thesis_text: str = ""
    thesis_type: str = "disposicao"
    synthetic_questions: str = ""

    # === Hierarquia Legal ===
    document_id: str = ""
    tipo_documento: str = ""
    numero: str = ""
    ano: int = 0
    chapter_number: str = ""
    chapter_title: str = ""
    article_number: str = ""
    article_title: str = ""

    # === Flags Estruturais ===
    has_items: bool = False
    has_paragraphs: bool = False
    item_count: int = 0
    paragraph_count: int = 0

    # === Metadados ===
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0

    # === Embeddings BGE-M3 (preenchidos pelo pipeline) ===
    dense_vector: Optional[list[float]] = None       # 1024d
    thesis_vector: Optional[list[float]] = None      # 1024d
    sparse_vector: Optional[dict[int, float]] = None # {token_id: weight}

    def to_dict(self) -> dict:
        """
        Converte para dicionário compatível com Milvus.

        Nota: O campo 'id' é auto-gerado pelo Milvus (autoID=true).
        """
        return {
            # Campos de texto
            "text": self.text,
            "enriched_text": self.enriched_text,
            "context_header": self.context_header,
            "thesis_text": self.thesis_text,
            "thesis_type": self.thesis_type,
            "synthetic_questions": self.synthetic_questions,

            # Hierarquia
            "document_id": self.document_id,
            "tipo_documento": self.tipo_documento,
            "numero": self.numero,
            "ano": self.ano,
            "section": f"Capítulo {self.chapter_number}" if self.chapter_number else "",
            "section_type": "capitulo",
            "section_title": self.chapter_title,
            "chunk_index": self.chunk_index,

            # Embeddings BGE-M3
            "dense_vector": self.dense_vector or [],
            "thesis_vector": self.thesis_vector or [],
            "sparse_vector": self.sparse_vector or {},

            # Campos adicionais (dynamic fields no Milvus)
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "chunk_level": self.chunk_level.value,
            "article_number": self.article_number,
            "article_title": self.article_title or "",
            "has_items": self.has_items,
            "has_paragraphs": self.has_paragraphs,
            "item_count": self.item_count,
            "paragraph_count": self.paragraph_count,
            "token_count": self.token_count,
        }

    def to_json(self) -> str:
        """Serializa para JSON."""
        d = self.to_dict()
        # Remove embeddings do JSON (muito grande)
        d.pop("dense_vector", None)
        d.pop("thesis_vector", None)
        d.pop("sparse_vector", None)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "LegalChunk":
        """Cria LegalChunk a partir de dicionário."""
        return cls(
            chunk_id=data.get("chunk_id", ""),
            parent_id=data.get("parent_id", ""),
            chunk_index=data.get("chunk_index", 0),
            chunk_level=ChunkLevel(data.get("chunk_level", 2)),
            text=data.get("text", ""),
            enriched_text=data.get("enriched_text", ""),
            context_header=data.get("context_header", ""),
            thesis_text=data.get("thesis_text", ""),
            thesis_type=data.get("thesis_type", "disposicao"),
            synthetic_questions=data.get("synthetic_questions", ""),
            document_id=data.get("document_id", ""),
            tipo_documento=data.get("tipo_documento", ""),
            numero=data.get("numero", ""),
            ano=data.get("ano", 0),
            chapter_number=data.get("chapter_number", ""),
            chapter_title=data.get("chapter_title", ""),
            article_number=data.get("article_number", ""),
            article_title=data.get("article_title", ""),
            has_items=data.get("has_items", False),
            has_paragraphs=data.get("has_paragraphs", False),
            item_count=data.get("item_count", 0),
            paragraph_count=data.get("paragraph_count", 0),
            token_count=data.get("token_count", 0),
            char_start=data.get("char_start", 0),
            char_end=data.get("char_end", 0),
            dense_vector=data.get("dense_vector"),
            thesis_vector=data.get("thesis_vector"),
            sparse_vector=data.get("sparse_vector"),
        )

    def __repr__(self) -> str:
        return (
            f"LegalChunk(id={self.chunk_id!r}, "
            f"type={self.thesis_type}, "
            f"items={self.item_count}, "
            f"paragraphs={self.paragraph_count}, "
            f"tokens={self.token_count})"
        )


@dataclass
class ChunkingResult:
    """Resultado do processo de chunking de um documento."""

    chunks: list[LegalChunk] = field(default_factory=list)
    document_id: str = ""
    total_chunks: int = 0
    total_tokens: int = 0
    processing_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """Retorna resumo do resultado."""
        level_counts = {}
        type_counts = {}

        for chunk in self.chunks:
            level = chunk.chunk_level.name
            level_counts[level] = level_counts.get(level, 0) + 1

            t = chunk.thesis_type
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "document_id": self.document_id,
            "total_chunks": len(self.chunks),
            "total_tokens": sum(c.token_count for c in self.chunks),
            "chunks_by_level": level_counts,
            "chunks_by_type": type_counts,
            "processing_time": f"{self.processing_time_seconds:.2f}s",
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }
