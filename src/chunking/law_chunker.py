"""
LawChunker - Pipeline Legado de Chunking para Documentos Legais.

Este módulo transforma documentos legais estruturados (LegalDocument) em chunks
prontos para indexação no Milvus, com suporte a enriquecimento via LLM e
geração de embeddings BGE-M3.

NOTA: Este é o pipeline LEGADO. Para novos projetos, prefira o pipeline v3
com SpanParser + ArticleOrchestrator + ChunkMaterializer, que oferece
suporte a parent-child retrieval e melhor controle de alucinação.

Arquitetura do Pipeline:
=======================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                     PIPELINE LAWCHUNKER (LEGADO)                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  LegalDocument (JSON)                                               │
    │        │                                                            │
    │        ▼                                                            │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │  FASE 1: Criação de Chunks Brutos                             │  │
    │  │  _create_raw_chunks()                                         │  │
    │  │  ├── Itera capítulos e artigos                                │  │
    │  │  ├── Artigo < max_tokens → chunk único                        │  │
    │  │  └── Artigo > max_tokens → divide em sub-chunks               │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │        │                                                            │
    │        ▼                                                            │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │  FASE 2: Enriquecimento LLM (se habilitado)                   │  │
    │  │  _enrich_chunks()                                             │  │
    │  │  ├── Processa em batches (config.batch_size)                  │  │
    │  │  ├── Gera context_header, thesis_text, thesis_type            │  │
    │  │  ├── Gera synthetic_questions                                 │  │
    │  │  └── Monta enriched_text                                      │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │        │                                                            │
    │        ▼                                                            │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │  FASE 3: Geração de Embeddings (se habilitado)                │  │
    │  │  _generate_embeddings()                                       │  │
    │  │  ├── BGE-M3 encode_hybrid()                                   │  │
    │  │  ├── dense_vector (1024d) do enriched_text                    │  │
    │  │  ├── thesis_vector (1024d) do thesis_text                     │  │
    │  │  └── sparse_vector do enriched_text                           │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │        │                                                            │
    │        ▼                                                            │
    │  list[LegalChunk] → Milvus                                         │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Hierarquia de Chunking:
======================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  LegalDocument                                                      │
    │       │                                                             │
    │       └── Chapter (Capítulo)                                        │
    │              │                                                      │
    │              └── Article (Artigo)                                   │
    │                     │                                               │
    │                     ├── Se < max_tokens → 1 chunk ARTICLE           │
    │                     │                                               │
    │                     └── Se > max_tokens → divide:                   │
    │                            ├── CAPUT + Incisos → chunk DEVICE       │
    │                            ├── Incisos em grupos → chunks DEVICE    │
    │                            └── Parágrafos → chunk DEVICE            │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Estratégias de Divisão:
======================

    1. ARTIGO PEQUENO (< max_tokens)
       ┌────────────────────────────────────────────┐
       │  Art. 3º + Incisos + Parágrafos            │
       │  = 1 chunk (ARTICLE level)                 │
       └────────────────────────────────────────────┘

    2. ARTIGO GRANDE (> max_tokens)
       ┌────────────────────────────────────────────┐
       │  Chunk 1: Art. 3º + Incisos I-X            │
       │  (DEVICE level, ID: ART-3#CAPUT)           │
       ├────────────────────────────────────────────┤
       │  Chunk 2: Art. 3º (cont.) + Incisos XI-XX  │
       │  (DEVICE level, ID: ART-3#INC-XI-XX)       │
       ├────────────────────────────────────────────┤
       │  Chunk 3: Art. 3º (parágrafos)             │
       │  (DEVICE level, ID: ART-3#PARAS)           │
       └────────────────────────────────────────────┘

Configurações (ChunkerConfig):
=============================

    | Config                | Default | Descrição                         |
    |-----------------------|---------|-----------------------------------|
    | max_chunk_tokens      | 1024    | Máximo de tokens por chunk        |
    | min_chunk_tokens      | 50      | Mínimo de tokens por chunk        |
    | enrich_with_llm       | True    | Ativar enriquecimento LLM         |
    | batch_size            | 5       | Chunks por batch no LLM           |
    | llm_temperature       | 0.0     | Temperatura do LLM                |
    | generate_embeddings   | True    | Gerar embeddings BGE-M3           |
    | split_large_articles  | True    | Dividir artigos grandes           |
    | split_strategy        | items_paragraphs | Estratégia de divisão   |
    | id_separator          | #       | Separador em IDs hierárquicos     |

    Presets disponíveis:
    - ChunkerConfig.default(): Configuração padrão
    - ChunkerConfig.fast(): Sem LLM, sem embeddings (apenas estrutura)
    - ChunkerConfig.full(): Tudo habilitado, batch_size=10

Protocol Interfaces:
===================

    O LawChunker usa Protocol (duck typing) para flexibilidade:

    | Interface       | Método Requerido                       | Implementação        |
    |-----------------|----------------------------------------|----------------------|
    | LLMClient       | chat(messages, temperature, max_tokens)| VLLMClient           |
    | EmbeddingModel  | encode(texts) → list[list[float]]      | BGEM3Embedder        |
    |                 | encode_hybrid(texts) → dict            | BGEM3Embedder        |
    | Tokenizer       | encode(text) → list[int]               | tiktoken             |

Formato de IDs:
==============

    | Nível   | Formato                              | Exemplo                       |
    |---------|--------------------------------------|-------------------------------|
    | Documento| {TIPO}-{SIGLA}-{NUM}-{ANO}          | IN-SEGES-58-2022              |
    | Capítulo | {DOC}#CAP-{NUM}                     | IN-SEGES-58-2022#CAP-I        |
    | Artigo   | {CAP}#ART-{NUM}                     | IN-SEGES-58-2022#CAP-I#ART-3  |
    | Dispositivo| {ART}#CAPUT ou #INC-I-X ou #PARAS | ...#ART-3#CAPUT               |

Exemplo de Uso:
==============

    ```python
    from chunking import LawChunker, ChunkerConfig
    from models.legal_document import LegalDocument
    from llm import VLLMClient
    from embeddings import BGEM3Embedder

    # 1. Configuração completa
    config = ChunkerConfig(
        max_chunk_tokens=1024,
        enrich_with_llm=True,
        generate_embeddings=True,
        batch_size=5,
    )

    # 2. Inicializar componentes
    llm = VLLMClient()
    embedder = BGEM3Embedder()

    # 3. Criar chunker
    chunker = LawChunker(
        llm_client=llm,
        embedding_model=embedder,
        config=config,
    )

    # 4. Carregar documento
    with open("documento.json") as f:
        doc = LegalDocument.model_validate(json.load(f))

    # 5. Executar pipeline
    result = chunker.chunk_document(doc)

    # 6. Processar resultado
    print(result.summary())
    # {
    #     "document_id": "IN-SEGES-58-2022",
    #     "total_chunks": 25,
    #     "chunks_by_level": {"ARTICLE": 15, "DEVICE": 10},
    #     "chunks_by_type": {"definicao": 5, "procedimento": 10, ...},
    #     "processing_time": "12.34s"
    # }

    # 7. Inserir no Milvus
    for chunk in result.chunks:
        milvus_data = chunk.to_dict()
        collection.insert([milvus_data])
    ```

    ```python
    # Modo rápido (sem LLM, sem embeddings)
    chunker = LawChunker(config=ChunkerConfig.fast())
    result = chunker.chunk_document(doc)
    # Retorna estrutura de chunks sem enriquecimento
    ```

Métodos Principais:
==================

    | Método                      | Descrição                                  |
    |-----------------------------|---------------------------------------------|
    | chunk_document(doc)         | Pipeline completo: chunk + enrich + embed   |
    | _create_raw_chunks(doc)     | Cria chunks brutos (sem enriquecimento)     |
    | _split_large_article(...)   | Divide artigo grande em sub-chunks          |
    | _enrich_chunks(chunks, doc) | Enriquece com LLM                           |
    | _generate_embeddings(chunks)| Gera embeddings BGE-M3                      |
    | _build_document_id(doc)     | Gera ID único do documento                  |
    | _build_article_text(article)| Monta texto completo do artigo              |

Tratamento de Erros:
===================

    - Se LLM falhar: Usa _default_enrichment() (fallback simples)
    - Se batch LLM falhar: Tenta individualmente cada chunk
    - Se embedder falhar: Retorna chunks sem vetores
    - Se tiktoken não instalado: Usa _SimpleTokenizer (estimativa)

Integração com Outros Módulos:
=============================

    - models/legal_document.py: LegalDocument (entrada)
    - enrichment_prompts.py: Prompts de enriquecimento
    - chunk_models.py: LegalChunk, ChunkingResult (saída)
    - llm/vllm_client.py: VLLMClient (implementa LLMClient)
    - embeddings/bge_m3.py: BGEM3Embedder (implementa EmbeddingModel)
    - milvus/schema.py: Schema que recebe LegalChunk.to_dict()

Comparação com Pipeline v3:
==========================

    | Aspecto           | LawChunker (legado)       | ChunkMaterializer (v3)      |
    |-------------------|---------------------------|------------------------------|
    | Entrada           | LegalDocument (JSON)      | ArticleChunk (de SpanParser) |
    | Parent-Child      | Não                       | Sim                          |
    | Anti-alucinação   | Básico                    | Schema enum dinâmico         |
    | Citações          | Não                       | Sim (citations list)         |
    | Page spans        | Não                       | Sim (coordenadas PDF)        |
    | Caso de uso       | Migração, docs simples    | Produção, novos projetos     |

@author: Equipe VectorGov
@version: 1.0.0
@since: 21/12/2024
"""

import json
import logging
import re
import time
from typing import Optional, Protocol, Any
from dataclasses import dataclass

from .chunk_models import LegalChunk, ChunkLevel, ChunkingResult
from .enrichment_prompts import (
    build_enrichment_prompt,
    build_batch_enrichment_prompt,
    build_enriched_text,
    parse_enrichment_response,
    parse_batch_enrichment_response,
)

# Importa modelos do documento legal
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.legal_document import LegalDocument, Chapter, Article, Item, Paragraph


logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLOS (Interfaces)
# =============================================================================

class LLMClient(Protocol):
    """Interface para cliente LLM (vLLM, Ollama, etc)."""

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Envia mensagens e retorna resposta."""
        ...


class EmbeddingModel(Protocol):
    """Interface para modelo de embedding BGE-M3."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings densos para lista de textos."""
        ...

    def encode_hybrid(self, texts: list[str]) -> dict:
        """Gera embeddings hibridos (dense + sparse)."""
        ...


class Tokenizer(Protocol):
    """Interface para tokenizer."""

    def encode(self, text: str) -> list[int]:
        """Tokeniza texto."""
        ...


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

@dataclass
class ChunkerConfig:
    """Configuração do LawChunker."""

    # Limites de tamanho
    max_chunk_tokens: int = 1024
    min_chunk_tokens: int = 50

    # Enriquecimento
    enrich_with_llm: bool = True
    batch_size: int = 5  # Chunks por batch para LLM
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # Embeddings
    generate_embeddings: bool = True

    # Divisão de artigos grandes
    split_large_articles: bool = True
    split_strategy: str = "items_paragraphs"  # items_paragraphs | groups

    # IDs
    id_separator: str = "#"
    normalize_ids: bool = True

    @classmethod
    def default(cls) -> "ChunkerConfig":
        """Configuração padrão."""
        return cls()

    @classmethod
    def fast(cls) -> "ChunkerConfig":
        """Configuração rápida (sem LLM, sem embeddings)."""
        return cls(
            enrich_with_llm=False,
            generate_embeddings=False,
        )

    @classmethod
    def full(cls) -> "ChunkerConfig":
        """Configuração completa."""
        return cls(
            enrich_with_llm=True,
            generate_embeddings=True,
            batch_size=10,
        )


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class LawChunker:
    """
    Transforma LegalDocument em chunks prontos para Milvus.

    Implementa chunking hierárquico para documentos legais brasileiros,
    respeitando a estrutura: Capítulo > Artigo > Inciso/Parágrafo.

    Attributes:
        llm: Cliente LLM para enriquecimento
        embedder: Modelo de embedding (BGE-M3)
        config: Configurações do chunker
        tokenizer: Tokenizer para contagem de tokens
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        config: Optional[ChunkerConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        Inicializa o LawChunker.

        Args:
            llm_client: Cliente para LLM (vLLM, Ollama). Opcional se enrich_with_llm=False
            embedding_model: Modelo BGE-M3. Opcional se generate_embeddings=False
            config: Configurações. Usa default se não fornecido
            tokenizer: Tokenizer para contagem. Usa tiktoken se não fornecido
        """
        self.llm = llm_client
        self.embedder = embedding_model
        self.config = config or ChunkerConfig.default()
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        """Lazy loading do tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken não instalado, usando estimativa de tokens")
                self._tokenizer = _SimpleTokenizer()
        return self._tokenizer

    # =========================================================================
    # PIPELINE PRINCIPAL
    # =========================================================================

    def chunk_document(self, doc: LegalDocument) -> ChunkingResult:
        """
        Pipeline completo: chunking + enriquecimento + embedding.

        Args:
            doc: Documento legal estruturado (LegalDocument)

        Returns:
            ChunkingResult com lista de LegalChunk prontos para Milvus
        """
        start_time = time.time()
        result = ChunkingResult(document_id=self._build_document_id(doc))

        try:
            # 1. Criar chunks brutos (sem enriquecimento)
            logger.info(f"Criando chunks para {result.document_id}")
            raw_chunks = self._create_raw_chunks(doc)
            logger.info(f"Criados {len(raw_chunks)} chunks brutos")

            # 2. Enriquecer com LLM (se habilitado)
            if self.config.enrich_with_llm and self.llm:
                logger.info("Enriquecendo chunks com LLM...")
                enriched_chunks = self._enrich_chunks(raw_chunks, doc)
            else:
                enriched_chunks = raw_chunks
                logger.info("Enriquecimento LLM desabilitado")

            # 3. Gerar embeddings (se habilitado)
            if self.config.generate_embeddings and self.embedder:
                logger.info("Gerando embeddings...")
                final_chunks = self._generate_embeddings(enriched_chunks)
            else:
                final_chunks = [LegalChunk(**c) for c in enriched_chunks]
                logger.info("Geração de embeddings desabilitada")

            result.chunks = final_chunks
            result.total_chunks = len(final_chunks)
            result.total_tokens = sum(c.token_count for c in final_chunks)

        except Exception as e:
            logger.exception(f"Erro no chunking: {e}")
            result.errors.append(str(e))

        result.processing_time_seconds = time.time() - start_time
        return result

    # =========================================================================
    # CRIAÇÃO DE CHUNKS
    # =========================================================================

    def _create_raw_chunks(self, doc: LegalDocument) -> list[dict]:
        """
        Cria chunks brutos respeitando hierarquia legal.

        Regras:
        1. Unidade base é o Artigo completo
        2. Artigo > max_tokens é dividido em sub-chunks
        3. Nunca divide inciso ou parágrafo no meio
        """
        chunks = []
        chunk_index = 0

        doc_id = self._build_document_id(doc)
        year = self._extract_year(doc.date)

        for chapter in doc.chapters:
            chapter_id = self._build_chapter_id(doc_id, chapter)

            for article in chapter.articles:
                article_id = self._build_article_id(chapter_id, article)

                # Monta texto completo do artigo
                article_text = self._build_article_text(article)
                token_count = self._count_tokens(article_text)

                # Verifica se precisa dividir
                if token_count > self.config.max_chunk_tokens and self.config.split_large_articles:
                    # Divide artigo em sub-chunks
                    sub_chunks = self._split_large_article(
                        article, chapter, doc, article_id, chunk_index
                    )
                    for sub in sub_chunks:
                        sub["document_id"] = doc_id
                        sub["tipo_documento"] = doc.document_type
                        sub["numero"] = doc.number
                        sub["ano"] = year
                        sub["chapter_number"] = chapter.chapter_number or ""
                        sub["chapter_title"] = chapter.title
                        chunks.append(sub)
                        chunk_index += 1
                else:
                    # Artigo inteiro como um chunk
                    chunk = {
                        "chunk_id": article_id,
                        "parent_id": chapter_id,
                        "chunk_index": chunk_index,
                        "chunk_level": ChunkLevel.ARTICLE,
                        "text": article_text,
                        "document_id": doc_id,
                        "tipo_documento": doc.document_type,
                        "numero": doc.number,
                        "ano": year,
                        "chapter_number": chapter.chapter_number or "",
                        "chapter_title": chapter.title,
                        "article_number": article.article_number,
                        "article_title": article.title or "",
                        "has_items": len(article.items) > 0,
                        "has_paragraphs": len(article.paragraphs) > 0,
                        "item_count": len(article.items),
                        "paragraph_count": len(article.paragraphs),
                        "token_count": token_count,
                    }
                    chunks.append(chunk)
                    chunk_index += 1

        return chunks

    def _split_large_article(
        self,
        article: Article,
        chapter: Chapter,
        doc: LegalDocument,
        article_id: str,
        start_index: int,
    ) -> list[dict]:
        """
        Divide artigo grande em sub-chunks.

        Estratégia: Separa caput+incisos de parágrafos.
        Se ainda grande, agrupa incisos.
        """
        chunks = []
        sep = self.config.id_separator

        # Parte 1: Caput + Incisos
        caput_text = self._build_caput_with_items(article)
        caput_tokens = self._count_tokens(caput_text)

        if caput_tokens <= self.config.max_chunk_tokens:
            # Caput + incisos cabem em um chunk
            chunks.append({
                "chunk_id": f"{article_id}{sep}CAPUT",
                "parent_id": article_id,
                "chunk_index": start_index,
                "chunk_level": ChunkLevel.DEVICE,
                "text": caput_text,
                "article_number": article.article_number,
                "article_title": article.title or "",
                "has_items": len(article.items) > 0,
                "has_paragraphs": False,
                "item_count": len(article.items),
                "paragraph_count": 0,
                "token_count": caput_tokens,
            })
        else:
            # Precisa dividir incisos em grupos
            item_chunks = self._split_items_into_groups(article, article_id, start_index)
            chunks.extend(item_chunks)

        # Parte 2: Parágrafos (se houver)
        if article.paragraphs:
            para_text = self._build_paragraphs_text(article)
            para_tokens = self._count_tokens(para_text)

            # Adiciona contexto do caput
            context_prefix = f"Art. {article.article_number}º (parágrafos):\n\n"

            chunks.append({
                "chunk_id": f"{article_id}{sep}PARAS",
                "parent_id": article_id,
                "chunk_index": start_index + len(chunks),
                "chunk_level": ChunkLevel.DEVICE,
                "text": context_prefix + para_text,
                "article_number": article.article_number,
                "article_title": article.title or "",
                "has_items": False,
                "has_paragraphs": True,
                "item_count": 0,
                "paragraph_count": len(article.paragraphs),
                "token_count": para_tokens + self._count_tokens(context_prefix),
            })

        return chunks

    def _split_items_into_groups(
        self,
        article: Article,
        article_id: str,
        start_index: int,
    ) -> list[dict]:
        """
        Divide incisos em grupos quando artigo é muito grande.
        """
        chunks = []
        sep = self.config.id_separator
        max_tokens = self.config.max_chunk_tokens

        # Caput sempre no primeiro chunk
        caput = f"Art. {article.article_number}º {article.content}\n\n"
        current_text = caput
        current_items = []
        group_num = 1

        for item in article.items:
            item_text = self._build_item_text(item)
            combined = current_text + item_text + "\n"

            if self._count_tokens(combined) > max_tokens and current_items:
                # Salva grupo atual
                first_item = current_items[0].item_identifier
                last_item = current_items[-1].item_identifier
                chunk_id = f"{article_id}{sep}INC-{first_item}-{last_item}"

                chunks.append({
                    "chunk_id": chunk_id,
                    "parent_id": article_id,
                    "chunk_index": start_index + len(chunks),
                    "chunk_level": ChunkLevel.DEVICE,
                    "text": current_text.strip(),
                    "article_number": article.article_number,
                    "article_title": article.title or "",
                    "has_items": True,
                    "has_paragraphs": False,
                    "item_count": len(current_items),
                    "paragraph_count": 0,
                    "token_count": self._count_tokens(current_text),
                })

                # Inicia novo grupo com contexto
                current_text = f"Art. {article.article_number}º (continuação):\n\n{item_text}\n"
                current_items = [item]
                group_num += 1
            else:
                current_text = combined
                current_items.append(item)

        # Último grupo
        if current_items:
            first_item = current_items[0].item_identifier
            last_item = current_items[-1].item_identifier
            chunk_id = f"{article_id}{sep}INC-{first_item}-{last_item}"

            chunks.append({
                "chunk_id": chunk_id,
                "parent_id": article_id,
                "chunk_index": start_index + len(chunks),
                "chunk_level": ChunkLevel.DEVICE,
                "text": current_text.strip(),
                "article_number": article.article_number,
                "article_title": article.title or "",
                "has_items": True,
                "has_paragraphs": False,
                "item_count": len(current_items),
                "paragraph_count": 0,
                "token_count": self._count_tokens(current_text),
            })

        return chunks

    # =========================================================================
    # CONSTRUÇÃO DE TEXTO
    # =========================================================================

    def _build_article_text(self, article: Article) -> str:
        """Monta texto completo do artigo para chunking."""
        parts = []

        # Caput
        parts.append(f"Art. {article.article_number}º {article.content}")

        # Incisos
        for item in article.items:
            parts.append(self._build_item_text(item))

        # Parágrafos
        for para in article.paragraphs:
            parts.append(self._build_paragraph_text(para))

        return "\n".join(parts)

    def _build_caput_with_items(self, article: Article) -> str:
        """Monta caput + incisos (sem parágrafos)."""
        parts = [f"Art. {article.article_number}º {article.content}"]

        for item in article.items:
            parts.append(self._build_item_text(item))

        return "\n".join(parts)

    def _build_paragraphs_text(self, article: Article) -> str:
        """Monta texto dos parágrafos."""
        parts = []
        for para in article.paragraphs:
            parts.append(self._build_paragraph_text(para))
        return "\n".join(parts)

    def _build_item_text(self, item: Item) -> str:
        """Monta texto de um inciso com suas alíneas."""
        text = f"{item.item_identifier} - {item.description}"

        for sub in item.sub_items:
            text += f"\n  {sub.item_identifier}) {sub.description}"

        return text

    def _build_paragraph_text(self, para: Paragraph) -> str:
        """Monta texto de um parágrafo."""
        if para.paragraph_identifier == "unico":
            return f"Parágrafo único. {para.content}"
        return f"§ {para.paragraph_identifier}º {para.content}"

    # =========================================================================
    # ENRIQUECIMENTO COM LLM
    # =========================================================================

    def _enrich_chunks(self, chunks: list[dict], doc: LegalDocument) -> list[dict]:
        """
        Enriquece chunks com LLM (context_header, thesis_text, synthetic_questions).
        """
        if not self.llm:
            logger.warning("LLM não configurado, pulando enriquecimento")
            return chunks

        year = self._extract_year(doc.date)
        enriched = []

        # Processa em batches
        batch_size = self.config.batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            if len(batch) == 1:
                # Chunk único - usa prompt individual
                chunk = batch[0]
                enrichment = self._enrich_single_chunk(chunk, doc, year)
                chunk.update(enrichment)
                enriched.append(chunk)
            else:
                # Múltiplos chunks - usa prompt batch
                enrichments = self._enrich_batch(batch, doc, year)
                for chunk, enrich in zip(batch, enrichments):
                    chunk.update(enrich)
                    enriched.append(chunk)

        return enriched

    def _enrich_single_chunk(self, chunk: dict, doc: LegalDocument, year: int) -> dict:
        """Enriquece um único chunk."""
        try:
            system_prompt, user_prompt = build_enrichment_prompt(
                text=chunk["text"],
                document_type=doc.document_type,
                number=doc.number,
                year=year,
                issuing_body=doc.issuing_body,
                chapter_number=chunk.get("chapter_number", ""),
                chapter_title=chunk.get("chapter_title", ""),
                article_number=chunk.get("article_number", ""),
                article_title=chunk.get("article_title"),
            )

            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )

            enrichment = parse_enrichment_response(response)

            # Monta enriched_text
            questions = enrichment["synthetic_questions"].split("\n")
            enrichment["enriched_text"] = build_enriched_text(
                text=chunk["text"],
                context_header=enrichment["context_header"],
                synthetic_questions=questions,
            )

            return enrichment

        except Exception as e:
            logger.error(f"Erro ao enriquecer chunk {chunk.get('chunk_id')}: {e}")
            return self._default_enrichment(chunk)

    def _enrich_batch(self, chunks: list[dict], doc: LegalDocument, year: int) -> list[dict]:
        """Enriquece batch de chunks."""
        try:
            batch_data = [
                {
                    "text": c["text"],
                    "chapter_number": c.get("chapter_number", ""),
                    "chapter_title": c.get("chapter_title", ""),
                    "article_number": c.get("article_number", ""),
                    "article_title": c.get("article_title"),
                }
                for c in chunks
            ]

            system_prompt, user_prompt = build_batch_enrichment_prompt(
                chunks=batch_data,
                document_type=doc.document_type,
                number=doc.number,
                year=year,
                issuing_body=doc.issuing_body,
            )

            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens * len(chunks),
            )

            enrichments = parse_batch_enrichment_response(response, len(chunks))

            # Monta enriched_text para cada
            for i, enrich in enumerate(enrichments):
                questions = enrich["synthetic_questions"].split("\n")
                enrich["enriched_text"] = build_enriched_text(
                    text=chunks[i]["text"],
                    context_header=enrich["context_header"],
                    synthetic_questions=questions,
                )

            return enrichments

        except Exception as e:
            logger.error(f"Erro no enriquecimento batch: {e}")
            # Fallback: enriquece individualmente
            return [self._enrich_single_chunk(c, doc, year) for c in chunks]

    def _default_enrichment(self, chunk: dict) -> dict:
        """Enriquecimento padrão quando LLM falha."""
        return {
            "context_header": f"Art. {chunk.get('article_number', '')} do documento",
            "thesis_text": chunk["text"][:500] if len(chunk["text"]) > 500 else chunk["text"],
            "thesis_type": "disposicao",
            "synthetic_questions": "",
            "enriched_text": chunk["text"],
        }

    # =========================================================================
    # GERAÇÃO DE EMBEDDINGS
    # =========================================================================

    def _generate_embeddings(self, chunks: list[dict]) -> list[LegalChunk]:
        """
        Gera embeddings hibridos (dense + sparse) para cada chunk.

        Usa BGE-M3 para gerar:
        - dense_vector: embedding denso 1024d
        - thesis_vector: embedding denso da tese 1024d
        - sparse_vector: learned sparse {token_id: weight}
        """
        if not self.embedder:
            logger.warning("Embedder nao configurado, retornando sem embeddings")
            return [LegalChunk(**c) for c in chunks]

        # Prepara textos para embedding
        # Usa enriched_text (texto + contexto + perguntas) se disponivel
        texts = [c.get("enriched_text") or c["text"] for c in chunks]
        thesis_texts = [c.get("thesis_text") or c["text"] for c in chunks]

        # Gera embeddings hibridos em batch
        logger.info(f"Gerando embeddings hibridos para {len(texts)} chunks...")

        try:
            # Verifica se embedder suporta encode_hybrid
            if hasattr(self.embedder, 'encode_hybrid'):
                # Embeddings hibridos (dense + sparse)
                text_result = self.embedder.encode_hybrid(texts)
                thesis_result = self.embedder.encode_hybrid(thesis_texts)

                dense_vectors = text_result["dense"]
                sparse_vectors = text_result["sparse"]
                thesis_vectors = thesis_result["dense"]
            else:
                # Fallback para encode simples (apenas dense)
                logger.warning("Embedder nao suporta encode_hybrid, usando apenas dense")
                dense_vectors = self.embedder.encode(texts)
                thesis_vectors = self.embedder.encode(thesis_texts)
                sparse_vectors = [None] * len(chunks)

        except Exception as e:
            logger.error(f"Erro ao gerar embeddings: {e}")
            dense_vectors = [None] * len(chunks)
            thesis_vectors = [None] * len(chunks)
            sparse_vectors = [None] * len(chunks)

        # Monta LegalChunks finais
        final_chunks = []
        for i, chunk_data in enumerate(chunks):
            chunk_data["dense_vector"] = dense_vectors[i]
            chunk_data["thesis_vector"] = thesis_vectors[i]
            chunk_data["sparse_vector"] = sparse_vectors[i]
            final_chunks.append(LegalChunk(**chunk_data))

        return final_chunks

    # =========================================================================
    # CONSTRUÇÃO DE IDs
    # =========================================================================

    def _build_document_id(self, doc: LegalDocument) -> str:
        """Gera ID único do documento."""
        tipo = self._normalize_id(doc.document_type)
        numero = doc.number
        ano = self._extract_year(doc.date)

        if doc.issuing_body_acronym:
            sigla = self._normalize_id(doc.issuing_body_acronym)
            return f"{tipo}-{sigla}-{numero}-{ano}"

        return f"{tipo}-{numero}-{ano}"

    def _build_chapter_id(self, doc_id: str, chapter: Chapter) -> str:
        """Gera ID do capítulo."""
        sep = self.config.id_separator
        cap_num = chapter.chapter_number or "UNICO"
        return f"{doc_id}{sep}CAP-{cap_num}"

    def _build_article_id(self, chapter_id: str, article: Article) -> str:
        """Gera ID do artigo."""
        sep = self.config.id_separator
        return f"{chapter_id}{sep}ART-{article.article_number}"

    def _normalize_id(self, text: str) -> str:
        """Normaliza texto para uso em ID."""
        if not self.config.normalize_ids:
            return text

        # Remove acentos
        import unicodedata
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ASCII", "ignore").decode("ASCII")

        # Substitui espaços por hífen
        text = re.sub(r"\s+", "-", text)

        # Remove caracteres especiais
        text = re.sub(r"[^A-Za-z0-9\-]", "", text)

        return text.upper()

    def _extract_year(self, date_str: str) -> int:
        """Extrai ano da data."""
        try:
            return int(date_str.split("-")[0])
        except (ValueError, IndexError):
            return 0

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    def _count_tokens(self, text: str) -> int:
        """Conta tokens no texto."""
        return len(self.tokenizer.encode(text))


# =============================================================================
# TOKENIZER SIMPLES (fallback)
# =============================================================================

class _SimpleTokenizer:
    """Tokenizer simples para quando tiktoken não está instalado."""

    def encode(self, text: str) -> list[int]:
        """Estimativa de tokens (~4 chars por token)."""
        return list(range(len(text) // 4))


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Carrega documento de teste
    json_path = Path(__file__).parent.parent.parent / "data" / "output" / "resultado_extracao_vllm_v3.json"

    if not json_path.exists():
        print(f"Arquivo não encontrado: {json_path}")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    doc = LegalDocument.model_validate(data["data"])

    # Chunker sem LLM/embeddings (apenas estrutura)
    chunker = LawChunker(config=ChunkerConfig.fast())
    result = chunker.chunk_document(doc)

    print("\n" + "=" * 60)
    print("RESULTADO DO CHUNKING")
    print("=" * 60)
    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("CHUNKS GERADOS")
    print("=" * 60)

    for chunk in result.chunks:
        print(f"\n{chunk.chunk_id}")
        print(f"  Level: {chunk.chunk_level.name}")
        print(f"  Items: {chunk.item_count}, Paragraphs: {chunk.paragraph_count}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Text preview: {chunk.text[:100]}...")
