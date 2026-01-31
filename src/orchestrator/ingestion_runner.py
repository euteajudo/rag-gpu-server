"""
Orquestrador principal do pipeline de ingestão.

PR3 v2.1 - Rebase: Usa SpanParser robusto + bridge para ChunkParts

Pipeline: PDF → Canonical → SpanParser → Bridge → Chunks → Embeddings → [Milvus, Neo4j]

Alterações PR3 v2.1:
- SpanExtractor (spans/) DEPRECATED em favor de SpanParser (parsing/)
- SpanParser é determinístico (regex-based) com 4 camadas anti-alucinação
- Bridge module converte ParsedDocument → ChunkPart[]
- Neo4j edges no nível lógico (por Span), não físico (por ChunkPart)
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from ..canonical import (
    Canonicalizer,
    CanonicalizationResult,
    get_pipeline_version,
    generate_ingest_run_id,
    SCHEMA_VERSION,
    get_prefix_for_document_type,
)
from ..parsing import SpanParser, SpanType, ParsedDocument
from ..bridge import ParsedDocumentChunkPartsBuilder, map_span_type_to_device_type
from ..spans import ChunkPart, DeviceType
# RetrievalTextBuilder removido - agora usamos _build_retrieval_contexts inline
from ..embeddings import EmbeddingClient, EmbeddingResult
from ..sinks import MilvusWriter, MilvusChunk, Neo4jEdgeWriter, LegalNodePayload, EdgeCandidate
from ..storage import ObjectStorageClient
from ..registry import DocumentRegistryService, DocumentStatus
from ..manifest import ManifestBuilder, IngestManifest

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuração do pipeline de ingestão."""

    # Milvus
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection: str = "leis_v4"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "vectorgov"
    minio_secure: bool = False

    # PostgreSQL
    postgres_url: str = "postgresql://rag:rag@localhost:5432/rag_legal"

    # Processamento
    batch_size: int = 32  # Para embeddings


@dataclass
class IngestionResult:
    """Resultado da ingestão de um documento."""

    success: bool
    document_id: str
    ingest_run_id: str
    manifest: Optional[IngestManifest] = None
    error_message: Optional[str] = None


class IngestionRunner:
    """
    Orquestrador do pipeline de ingestão.

    Executa todas as etapas em ordem:
    1. Canonicalização (PDF → Markdown)
    2. Armazenamento no MinIO
    3. Registro no PostgreSQL
    4. Extração de spans
    5. Criação de chunks físicos
    6. Construção de retrieval_text
    7. Geração de embeddings
    8. Upsert no Milvus
    9. Criação de edges no Neo4j
    10. Atualização do manifest
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        """
        Inicializa o runner.

        Args:
            config: Configuração do pipeline
        """
        self.config = config or IngestionConfig()

        # Componentes (lazy-loaded)
        self._canonicalizer: Optional[Canonicalizer] = None
        self._embedder: Optional[EmbeddingClient] = None
        self._milvus_writer: Optional[MilvusWriter] = None
        self._neo4j_writer: Optional[Neo4jEdgeWriter] = None
        self._storage: Optional[ObjectStorageClient] = None
        self._registry: Optional[DocumentRegistryService] = None

    def _get_canonicalizer(self) -> Canonicalizer:
        """Lazy-load do canonicalizer."""
        if self._canonicalizer is None:
            self._canonicalizer = Canonicalizer()
        return self._canonicalizer

    def _get_embedder(self) -> EmbeddingClient:
        """Lazy-load do embedder."""
        if self._embedder is None:
            self._embedder = EmbeddingClient()
        return self._embedder

    def _get_milvus_writer(self) -> MilvusWriter:
        """Lazy-load do Milvus writer."""
        if self._milvus_writer is None:
            self._milvus_writer = MilvusWriter(
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                collection_name=self.config.milvus_collection,
            )
        return self._milvus_writer

    def _get_neo4j_writer(self) -> Neo4jEdgeWriter:
        """Lazy-load do Neo4j writer."""
        if self._neo4j_writer is None:
            self._neo4j_writer = Neo4jEdgeWriter(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database,
            )
        return self._neo4j_writer

    def _get_storage(self) -> ObjectStorageClient:
        """Lazy-load do storage client."""
        if self._storage is None:
            self._storage = ObjectStorageClient(
                endpoint=self.config.minio_endpoint,
                access_key=self.config.minio_access_key,
                secret_key=self.config.minio_secret_key,
                bucket=self.config.minio_bucket,
                secure=self.config.minio_secure,
            )
        return self._storage

    def _get_registry(self) -> DocumentRegistryService:
        """Lazy-load do registry service."""
        if self._registry is None:
            self._registry = DocumentRegistryService(self.config.postgres_url)
        return self._registry

    def run(
        self,
        pdf_bytes: bytes,
        document_id: str,
        document_type: str = "LEI",
        filename: Optional[str] = None,
    ) -> IngestionResult:
        """
        Executa o pipeline completo de ingestão.

        Args:
            pdf_bytes: Conteúdo do PDF
            document_id: ID único do documento (ex: LEI-14133-2021)
            document_type: Tipo do documento (LEI, DECRETO, IN, etc.)
            filename: Nome do arquivo original (opcional)

        Returns:
            IngestionResult com status e métricas
        """
        # Gera IDs de rastreabilidade
        ingest_run_id = generate_ingest_run_id()
        pipeline_version = get_pipeline_version()

        # Inicializa manifest builder
        manifest_builder = ManifestBuilder(
            document_id=document_id,
            ingest_run_id=ingest_run_id,
            pipeline_version=pipeline_version,
            schema_version=SCHEMA_VERSION,
        )
        manifest_builder.start()

        try:
            # 1. Hash do PDF fonte
            sha256_source = hashlib.sha256(pdf_bytes).hexdigest()
            logger.info(f"Iniciando ingestão: {document_id} (sha256: {sha256_source[:16]}...)")

            # 2. Armazena PDF no MinIO
            storage = self._get_storage()
            minio_source_key = storage.put_source_pdf(document_id, pdf_bytes)
            manifest_builder.set_source_info(
                sha256=sha256_source,
                size_bytes=len(pdf_bytes),
                minio_key=minio_source_key,
            )

            # 3. Registra no PostgreSQL
            registry = self._get_registry()
            doc_record = registry.create_or_get_document(
                document_id=document_id,
                sha256_source=sha256_source,
                minio_source_key=minio_source_key,
                ingest_run_id=ingest_run_id,
                pipeline_version=pipeline_version,
            )
            registry.update_status(doc_record.id, DocumentStatus.UPLOADED)

            # 4. Canonicalização (PDF → Markdown)
            canonicalizer = self._get_canonicalizer()
            canon_result = canonicalizer.canonicalize(pdf_bytes, filename)

            # 5. Armazena markdown no MinIO
            minio_canonical_key = storage.put_canonical_md(
                document_id, canon_result.markdown
            )
            manifest_builder.set_canonical_info(
                sha256=canon_result.sha256,
                char_count=canon_result.char_count,
                page_count=canon_result.page_count,
                minio_key=minio_canonical_key,
            )

            # Atualiza registro
            registry.update_status(
                doc_record.id,
                DocumentStatus.PROCESSED,
                sha256_canonical_md=canon_result.sha256,
                minio_canonical_key=minio_canonical_key,
            )

            # 6. Extração de spans com SpanParser (PR3 v2.1 - robusto, determinístico)
            parser = SpanParser()
            parsed_doc = parser.parse(canon_result.markdown)

            # Calcula métricas por tipo
            article_count = len(parsed_doc.get_spans_by_type(SpanType.ARTIGO))
            paragraph_count = len(parsed_doc.get_spans_by_type(SpanType.PARAGRAFO))
            inciso_count = len(parsed_doc.get_spans_by_type(SpanType.INCISO))
            alinea_count = len(parsed_doc.get_spans_by_type(SpanType.ALINEA))

            manifest_builder.set_span_metrics(
                span_count=len(parsed_doc.spans),
                article_count=article_count,
                paragraph_count=paragraph_count,
                inciso_count=inciso_count,
                alinea_count=alinea_count,
            )

            # 7. Criação de chunks físicos via bridge (PR3 v2.1)
            builder = ParsedDocumentChunkPartsBuilder(
                document_id=document_id,
                document_type=document_type,
            )
            chunk_parts = builder.build(parsed_doc)

            # 8. Construção de retrieval_text (para enriquecimento)
            # Cria mapa span_id -> texto para lookup
            retrieval_contexts = self._build_retrieval_contexts(
                parsed_doc=parsed_doc,
                document_id=document_id,
                document_type=document_type,
            )

            # 9. Geração de embeddings
            embedder = self._get_embedder()
            milvus_chunks = self._generate_embeddings(
                chunk_parts=chunk_parts,
                retrieval_contexts=retrieval_contexts,
                embedder=embedder,
                ingest_run_id=ingest_run_id,
                pipeline_version=pipeline_version,
            )

            # 10. Upsert no Milvus
            milvus_writer = self._get_milvus_writer()
            chunks_inserted = milvus_writer.upsert_batch(milvus_chunks)
            milvus_writer.flush()

            manifest_builder.set_chunk_metrics(
                chunk_count=chunks_inserted,
                split_count=sum(1 for cp in chunk_parts if cp.part_total > 1),
            )

            registry.update_status(
                doc_record.id,
                DocumentStatus.INDEXED,
                chunk_count=chunks_inserted,
            )

            # 11. Criação de nós e edges no Neo4j (PR3 v2.1 - nível lógico)
            neo4j_writer = self._get_neo4j_writer()
            node_count, edge_count = self._write_to_neo4j(
                parsed_doc=parsed_doc,
                neo4j_writer=neo4j_writer,
                document_id=document_id,
                document_type=document_type,
                ingest_run_id=ingest_run_id,
                pipeline_version=pipeline_version,
            )

            manifest_builder.set_graph_metrics(
                node_count=node_count,
                edge_count=edge_count,
            )

            registry.update_status(
                doc_record.id,
                DocumentStatus.GRAPH_SYNCED,
                edge_count=edge_count,
            )

            # 12. Salva manifest no MinIO
            manifest = manifest_builder.complete()
            minio_manifest_key = storage.put_manifest(document_id, manifest.to_json())
            manifest.minio_manifest_key = minio_manifest_key

            logger.info(
                f"Ingestão concluída: {document_id} - "
                f"{chunks_inserted} chunks, {edge_count} edges, "
                f"{manifest.duration_seconds:.2f}s"
            )

            return IngestionResult(
                success=True,
                document_id=document_id,
                ingest_run_id=ingest_run_id,
                manifest=manifest,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Erro na ingestão de {document_id}: {error_msg}")

            # Marca como falha no registry
            try:
                registry = self._get_registry()
                registry.mark_failed(document_id, error_msg)
            except Exception:
                pass

            # Gera manifest de falha
            manifest = manifest_builder.fail(error_msg)

            return IngestionResult(
                success=False,
                document_id=document_id,
                ingest_run_id=ingest_run_id,
                manifest=manifest,
                error_message=error_msg,
            )

    def _build_retrieval_contexts(
        self,
        parsed_doc: ParsedDocument,
        document_id: str,  # Reservado para uso futuro (enriquecimento)
        document_type: str,  # Reservado para uso futuro (contexto de tipo)
    ) -> dict:
        """
        Constrói contextos de retrieval a partir do ParsedDocument.

        PR3 v2.1: Usa ParsedDocument para criar contextos hierárquicos.

        Returns:
            Dict mapeando span_id -> RetrievalContext
        """
        # Parâmetros document_id e document_type reservados para futuro enriquecimento
        _ = document_id, document_type

        contexts = {}

        for span in parsed_doc.spans:
            # Mapeia SpanType para DeviceType para filtrar
            device_type = map_span_type_to_device_type(span.span_type)
            if device_type == DeviceType.UNKNOWN:
                continue

            # Busca texto do parent se existir
            parent_text = None
            if span.parent_id:
                parent_span = parsed_doc.get_span(span.parent_id)
                if parent_span:
                    parent_text = parent_span.text[:500] if parent_span.text else None

            # Constrói retrieval_text com contexto hierárquico
            context_parts = []
            if parent_text:
                context_parts.append(f"[CONTEXTO PARENT]\n{parent_text}")
            context_parts.append(span.text)

            retrieval_text = "\n\n".join(context_parts)

            # Cria objeto de contexto simples
            class RetrievalContext:
                def __init__(self, rt, pt):
                    self.retrieval_text = rt
                    self.parent_text = pt

            contexts[span.span_id] = RetrievalContext(
                rt=retrieval_text,
                pt=parent_text,
            )

        logger.info(f"Construídos {len(contexts)} contextos de retrieval")
        return contexts

    def _generate_embeddings(
        self,
        chunk_parts: list[ChunkPart],
        retrieval_contexts: dict,
        embedder: EmbeddingClient,
        ingest_run_id: str,
        pipeline_version: str,
    ) -> list[MilvusChunk]:
        """
        Gera embeddings e cria MilvusChunks.
        """
        milvus_chunks = []

        # Prepara textos para embedding em batch
        texts_to_embed = []
        for part in chunk_parts:
            # Busca retrieval_context para este span
            ctx = retrieval_contexts.get(part.span_id)
            if ctx:
                texts_to_embed.append(ctx.retrieval_text)
            else:
                texts_to_embed.append(part.text)

        # Gera embeddings em batch
        embedding_results = embedder.embed_batch(texts_to_embed)

        # Cria MilvusChunks
        for i, part in enumerate(chunk_parts):
            emb = embedding_results[i]
            ctx = retrieval_contexts.get(part.span_id)

            milvus_chunk = MilvusChunk(
                # IDs
                node_id=part.node_id,
                logical_node_id=part.logical_node_id,
                chunk_id=part.chunk_id,
                parent_chunk_id=part.parent_chunk_id,

                # Split info
                part_index=part.part_index,
                part_total=part.part_total,

                # Texto
                text=part.text,
                retrieval_text=ctx.retrieval_text if ctx else part.text,
                parent_text=ctx.parent_text if ctx else None,

                # Vetores
                dense_vector=emb.dense_vector,
                sparse_vector=emb.sparse_vector,

                # Metadados
                document_id=part.document_id,
                span_id=part.span_id,
                device_type=part.device_type.value if hasattr(part.device_type, 'value') else str(part.device_type),
                article_number=part.article_number,
                document_type=part.document_type,

                # Proveniência
                ingest_run_id=ingest_run_id,
                pipeline_version=pipeline_version,

                # Posição
                char_start=part.char_start,
                char_end=part.char_end,
            )
            milvus_chunks.append(milvus_chunk)

        logger.info(f"Gerados embeddings para {len(milvus_chunks)} chunks")
        return milvus_chunks

    def _write_to_neo4j(
        self,
        parsed_doc: ParsedDocument,
        neo4j_writer: Neo4jEdgeWriter,
        document_id: str,
        document_type: str,
        ingest_run_id: str,
        pipeline_version: str,
    ) -> tuple[int, int]:
        """
        Escreve nós e edges no Neo4j.

        PR3 v2.1: Opera no nível LÓGICO (ParsedDocument.spans), não físico (ChunkParts).
        Isso evita duplicação lógica causada pelo overlap em chunks físicos.

        Returns:
            Tupla (node_count, edge_count)
        """
        from ..canonical import build_logical_node_id, get_prefix_for_document_type
        from ..bridge import find_root_article_span_id

        prefix = get_prefix_for_document_type(document_type)
        node_count = 0
        edge_count = 0

        # Cria nós lógicos a partir dos spans do ParsedDocument
        for span in parsed_doc.spans:
            # Mapeia SpanType para DeviceType
            device_type = map_span_type_to_device_type(span.span_type)
            if device_type == DeviceType.UNKNOWN:
                continue

            # Constrói logical_node_id
            logical_node_id = build_logical_node_id(prefix, document_id, span.span_id)

            # Encontra número do artigo raiz
            article_span_id = find_root_article_span_id(span, parsed_doc)
            article_number = None
            if article_span_id and article_span_id.startswith("ART-"):
                article_number = article_span_id.replace("ART-", "")

            node = LegalNodePayload(
                node_id=logical_node_id,
                document_id=document_id,
                span_id=span.span_id,
                device_type=device_type.value,
                document_type=document_type,
                text_preview=span.text[:200] if span.text else "",
                article_number=article_number,
                ingest_run_id=ingest_run_id,
                pipeline_version=pipeline_version,
            )
            if neo4j_writer.upsert_node(node):
                node_count += 1

        # Cria edges para relações parent-child (hierarquia)
        edges = []
        for span in parsed_doc.spans:
            device_type = map_span_type_to_device_type(span.span_type)
            if device_type == DeviceType.UNKNOWN:
                continue

            if span.parent_id:
                # Constrói logical_node_ids
                child_logical_id = build_logical_node_id(prefix, document_id, span.span_id)
                parent_logical_id = build_logical_node_id(prefix, document_id, span.parent_id)

                edge = EdgeCandidate(
                    source_node_id=child_logical_id,
                    target_node_id=parent_logical_id,
                    relation_type="CHILD_OF",
                    confidence=1.0,
                    extraction_method="hierarchy",
                    ingest_run_id=ingest_run_id,
                )
                edges.append(edge)

        # PR3 v2.1: Extrai citações normativas do span.text e cria edges "CITA"
        from ..chunking.citation_extractor import extract_citations_from_chunk

        citation_edge_count = 0
        for span in parsed_doc.spans:
            device_type = map_span_type_to_device_type(span.span_type)
            if device_type == DeviceType.UNKNOWN:
                continue

            if not span.text:
                continue

            # Constrói logical_node_id do span atual
            source_logical_id = build_logical_node_id(prefix, document_id, span.span_id)

            # Resolve parent logical_id (para filtrar parent-loops)
            parent_logical_id = None
            if span.parent_id:
                parent_logical_id = build_logical_node_id(prefix, document_id, span.parent_id)

            # Extrai citações do texto do span
            citations = extract_citations_from_chunk(
                text=span.text,
                document_id=document_id,
                chunk_node_id=source_logical_id,
                parent_chunk_id=parent_logical_id,
                document_type=document_type,
            )

            # Cria edges de tipo "CITA" para cada citação
            for target_node_id in citations:
                # Evita self-loops (já removidos por normalize_citations, mas double-check)
                if target_node_id == source_logical_id:
                    continue

                edge = EdgeCandidate(
                    source_node_id=source_logical_id,
                    target_node_id=target_node_id,
                    relation_type="CITA",
                    confidence=0.9,  # Citações têm confiança alta mas não perfeita
                    extraction_method="citation_extractor",
                    ingest_run_id=ingest_run_id,
                )
                edges.append(edge)
                citation_edge_count += 1

        logger.info(f"Neo4j: Extraídas {citation_edge_count} citações normativas")

        if edges:
            edge_count = neo4j_writer.create_edges_batch(edges)

        logger.info(f"Neo4j: {node_count} nós, {edge_count} edges (nível lógico)")
        return node_count, edge_count

    def close(self) -> None:
        """Fecha todas as conexões."""
        if self._milvus_writer:
            self._milvus_writer.close()
        if self._neo4j_writer:
            self._neo4j_writer.close()
        if self._registry:
            self._registry.close()
