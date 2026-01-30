"""
ChunkEnricher - Enriquece chunks com contexto usando LLM.

===========================================================================
DEPRECATED / CÓDIGO LEGADO - NÃO USAR EM CÓDIGO NOVO
===========================================================================
Este módulo foi descontinuado em favor do retrieval_text determinístico.

Motivo da descontinuação:
- Custo: 3+ chamadas LLM por chunk
- Não-determinístico: resultados variam entre execuções
- Lento: adiciona latência significativa ao pipeline

Estratégia atual (retrieval_text):
- 100% determinístico, sem LLM
- Texto construído em chunk_materializer.build_retrieval_text()
- Pipeline usa: retrieval_text → text (fallback legados)

Este arquivo será removido em versão futura.
===========================================================================

Campos que eram adicionados a MaterializedChunk (agora não usados):
- context_header
- thesis_text
- thesis_type
- synthetic_questions
- enriched_text (combinacao para embedding)
"""

import time
from dataclasses import dataclass
from typing import Optional

from ..llm.vllm_client import VLLMClient, LLMConfig
from ..chunking.enrichment_prompts import (
    build_enrichment_prompt,
    build_batch_enrichment_prompt,
    parse_enrichment_response,
    parse_batch_enrichment_response,
    build_enriched_text,
)


@dataclass
class DocumentMetadata:
    """Metadados do documento para enriquecimento."""
    document_id: str
    document_type: str  # LEI, IN, DECRETO
    number: str
    year: int
    issuing_body: str = "PRESIDENCIA DA REPUBLICA"


@dataclass
class EnrichmentResult:
    """Resultado do enriquecimento de um chunk."""
    context_header: str
    thesis_text: str
    thesis_type: str
    synthetic_questions: str  # Perguntas separadas por \n
    enriched_text: str


class ChunkEnricher:
    """
    Enriquece chunks com contexto usando LLM.

    Implementa Contextual Retrieval da Anthropic para melhorar
    a qualidade da busca semantica.

    Uso:
        enricher = ChunkEnricher()

        # Enriquecimento individual
        result = enricher.enrich_chunk(chunk, doc_meta)

        # Enriquecimento em batch (mais eficiente)
        results = enricher.enrich_batch(chunks, doc_meta, batch_size=10)
    """

    def __init__(
        self,
        llm_client: Optional[VLLMClient] = None,
        config: Optional[LLMConfig] = None,
    ):
        """
        Inicializa o enricher.

        Args:
            llm_client: Cliente LLM (cria um novo se nao fornecido)
            config: Configuracao do LLM
        """
        if llm_client:
            self.llm = llm_client
            self._owns_client = False
        else:
            self.llm = VLLMClient(config or LLMConfig.for_enrichment())
            self._owns_client = True

        self.stats = {
            "chunks_processed": 0,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "errors": 0,
            "total_time": 0.0,
        }

    def close(self):
        """Fecha o cliente LLM se foi criado internamente."""
        if self._owns_client and self.llm:
            self.llm.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def enrich_chunk(
        self,
        chunk,  # MaterializedChunk
        doc_meta: DocumentMetadata,
        chapter_number: str = "",
        chapter_title: str = "",
    ) -> EnrichmentResult:
        """
        Enriquece um chunk individual.

        Args:
            chunk: MaterializedChunk a ser enriquecido
            doc_meta: Metadados do documento
            chapter_number: Numero do capitulo (opcional)
            chapter_title: Titulo do capitulo (opcional)

        Returns:
            EnrichmentResult com todos os campos preenchidos
        """
        start = time.time()

        # Constroi prompt
        system_prompt, user_prompt = build_enrichment_prompt(
            text=chunk.text,
            document_type=doc_meta.document_type,
            number=doc_meta.number,
            year=doc_meta.year,
            issuing_body=doc_meta.issuing_body,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            article_number=getattr(chunk, 'article_number', '') or chunk.span_id,
            article_title=None,
        )

        # Chama LLM
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.0,
        )

        # Parse resposta
        try:
            data = parse_enrichment_response(response)
        except Exception as e:
            self.stats["errors"] += 1
            # Fallback para valores vazios
            data = {
                "context_header": "",
                "thesis_text": "",
                "thesis_type": "disposicao",
                "synthetic_questions": "",
            }
            print(f"    [WARN] Erro no parse: {e}")

        # Monta texto enriquecido
        questions_list = data["synthetic_questions"].split("\n") if data["synthetic_questions"] else []
        enriched_text = build_enriched_text(
            text=chunk.text,
            context_header=data["context_header"],
            synthetic_questions=questions_list,
        )

        # Atualiza estatisticas
        self.stats["chunks_processed"] += 1
        self.stats["total_time"] += time.time() - start

        return EnrichmentResult(
            context_header=data["context_header"],
            thesis_text=data["thesis_text"],
            thesis_type=data["thesis_type"],
            synthetic_questions=data["synthetic_questions"],
            enriched_text=enriched_text,
        )

    def enrich_single(
        self,
        text: str,
        device_type: str,
        article_number: str,
        doc_meta: DocumentMetadata,
        chapter_number: str = "",
        chapter_title: str = "",
    ) -> Optional[dict]:
        """
        Enriquece um texto diretamente (sem precisar de MaterializedChunk).

        Util para enriquecer chunks ja indexados no Milvus.

        Args:
            text: Texto do chunk
            device_type: Tipo (article, paragraph, inciso)
            article_number: Numero do artigo
            doc_meta: Metadados do documento
            chapter_number: Numero do capitulo (opcional)
            chapter_title: Titulo do capitulo (opcional)

        Returns:
            Dict com campos de enriquecimento ou None se falhar
        """
        start = time.time()

        # Constroi prompt
        system_prompt, user_prompt = build_enrichment_prompt(
            text=text,
            document_type=doc_meta.document_type,
            number=doc_meta.number,
            year=doc_meta.year,
            issuing_body=doc_meta.issuing_body,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            article_number=article_number,
            article_title=None,
        )

        # Chama LLM
        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
        except Exception as e:
            self.stats["errors"] += 1
            print(f"    [WARN] Erro no LLM: {e}")
            return None

        # Parse resposta
        try:
            data = parse_enrichment_response(response)
        except Exception as e:
            self.stats["errors"] += 1
            print(f"    [WARN] Erro no parse: {e}")
            return None

        # Atualiza estatisticas
        self.stats["chunks_processed"] += 1
        self.stats["total_time"] += time.time() - start

        return {
            "context_header": data.get("context_header", ""),
            "thesis_text": data.get("thesis_text", ""),
            "thesis_type": data.get("thesis_type", "disposicao"),
            "synthetic_questions": data.get("synthetic_questions", "").split("\n") if data.get("synthetic_questions") else [],
        }

    def _build_enriched_text(
        self,
        original_text: str,
        context_header: str,
        synthetic_questions: list[str],
    ) -> str:
        """Monta texto enriquecido para embedding."""
        return build_enriched_text(
            text=original_text,
            context_header=context_header,
            synthetic_questions=synthetic_questions,
        )

    def enrich_batch(
        self,
        chunks: list,  # list[MaterializedChunk]
        doc_meta: DocumentMetadata,
        batch_size: int = 5,
        chapter_info: Optional[dict] = None,
    ) -> list[EnrichmentResult]:
        """
        Enriquece multiplos chunks em batches.

        Mais eficiente que enrich_chunk individual pois
        agrupa multiplos chunks em uma unica chamada LLM.

        Args:
            chunks: Lista de MaterializedChunk
            doc_meta: Metadados do documento
            batch_size: Tamanho do batch (max chunks por chamada LLM)
            chapter_info: Dict chunk_id -> {chapter_number, chapter_title}

        Returns:
            Lista de EnrichmentResult na mesma ordem dos chunks
        """
        results = []
        total = len(chunks)

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            print(f"    Batch {batch_num}/{total_batches} ({len(batch)} chunks)")

            batch_results = self._enrich_batch_internal(
                batch, doc_meta, chapter_info
            )
            results.extend(batch_results)

        return results

    def _enrich_batch_internal(
        self,
        chunks: list,
        doc_meta: DocumentMetadata,
        chapter_info: Optional[dict] = None,
    ) -> list[EnrichmentResult]:
        """Processa um batch de chunks."""
        start = time.time()

        # Prepara dados para o prompt batch
        chunks_data = []
        for chunk in chunks:
            ch_info = chapter_info.get(chunk.chunk_id, {}) if chapter_info else {}
            chunks_data.append({
                "text": chunk.text,
                "chapter_number": ch_info.get("chapter_number", ""),
                "chapter_title": ch_info.get("chapter_title", ""),
                "article_number": getattr(chunk, 'article_number', '') or chunk.span_id,
                "article_title": None,
            })

        # Constroi prompt batch
        system_prompt, user_prompt = build_batch_enrichment_prompt(
            chunks=chunks_data,
            document_type=doc_meta.document_type,
            number=doc_meta.number,
            year=doc_meta.year,
            issuing_body=doc_meta.issuing_body,
        )

        # Chama LLM
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048 * len(chunks),  # Escala com batch size
            temperature=0.0,
        )

        # Parse resposta
        results = []
        try:
            data_list = parse_batch_enrichment_response(response, len(chunks))

            for data, chunk in zip(data_list, chunks):
                questions_list = data["synthetic_questions"].split("\n") if data["synthetic_questions"] else []
                enriched_text = build_enriched_text(
                    text=chunk.text,
                    context_header=data["context_header"],
                    synthetic_questions=questions_list,
                )

                results.append(EnrichmentResult(
                    context_header=data["context_header"],
                    thesis_text=data["thesis_text"],
                    thesis_type=data["thesis_type"],
                    synthetic_questions=data["synthetic_questions"],
                    enriched_text=enriched_text,
                ))

        except Exception as e:
            print(f"    [WARN] Erro no batch: {e}")
            print(f"    [WARN] Fallback para enriquecimento individual")
            self.stats["errors"] += 1

            # Fallback: processa individualmente
            for chunk in chunks:
                try:
                    result = self.enrich_chunk(chunk, doc_meta)
                    results.append(result)
                except Exception as e2:
                    print(f"    [ERR] Chunk {chunk.chunk_id}: {e2}")
                    results.append(EnrichmentResult(
                        context_header="",
                        thesis_text="",
                        thesis_type="disposicao",
                        synthetic_questions="",
                        enriched_text=chunk.text,
                    ))

        # Atualiza estatisticas
        self.stats["chunks_processed"] += len(chunks)
        self.stats["total_time"] += time.time() - start

        return results

    def apply_to_chunks(
        self,
        chunks: list,  # list[MaterializedChunk]
        doc_meta: DocumentMetadata,
        batch_size: int = 5,
        chapter_info: Optional[dict] = None,
    ) -> list:
        """
        Enriquece chunks e aplica resultados diretamente nos objetos.

        Modifica os chunks in-place e tambem retorna a lista.

        Args:
            chunks: Lista de MaterializedChunk
            doc_meta: Metadados do documento
            batch_size: Tamanho do batch
            chapter_info: Info de capitulos

        Returns:
            A mesma lista de chunks (modificados in-place)
        """
        results = self.enrich_batch(
            chunks, doc_meta, batch_size, chapter_info
        )

        for chunk, result in zip(chunks, results):
            chunk.context_header = result.context_header
            chunk.thesis_text = result.thesis_text
            chunk.thesis_type = result.thesis_type
            chunk.synthetic_questions = result.synthetic_questions
            chunk.enriched_text = result.enriched_text

        return chunks

    def get_stats(self) -> dict:
        """Retorna estatisticas de processamento."""
        stats = self.stats.copy()
        if stats["chunks_processed"] > 0:
            stats["avg_time_per_chunk"] = stats["total_time"] / stats["chunks_processed"]
        return stats
