#!/usr/bin/env python3
"""
Warmup Script - Pré-baixa e carrega modelos.

Executar antes de iniciar o servidor para garantir que os modelos
estão baixados e funcionando.

Uso:
    python scripts/warmup.py
"""

import sys
import os

# Adiciona src ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("RAG GPU Server - Warmup")
    logger.info("=" * 60)

    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # 1. Embedder (BGE-M3)
    # -------------------------------------------------------------------------
    logger.info("\n>>> Carregando BGE-M3 Embedder...")
    embed_start = time.perf_counter()

    try:
        from src.embedder import get_embedder

        embedder = get_embedder()
        result = embedder.encode(["Teste de warmup para embeddings."])

        embed_time = time.perf_counter() - embed_start
        logger.info(f"    BGE-M3 carregado em {embed_time:.2f}s")
        logger.info(f"    Dimensão: {len(result.dense_embeddings[0])}")
        logger.info(f"    Sparse tokens: {len(result.sparse_embeddings[0])}")

    except Exception as e:
        logger.error(f"    ERRO ao carregar BGE-M3: {e}")
        return 1

    # -------------------------------------------------------------------------
    # 2. Reranker (BGE-Reranker)
    # -------------------------------------------------------------------------
    logger.info("\n>>> Carregando BGE-Reranker...")
    rerank_start = time.perf_counter()

    try:
        from src.reranker import get_reranker

        reranker = get_reranker()
        result = reranker.rerank(
            query="O que é ETP?",
            documents=[
                "ETP é o Estudo Técnico Preliminar.",
                "ETP significa documento de planejamento.",
            ],
        )

        rerank_time = time.perf_counter() - rerank_start
        logger.info(f"    BGE-Reranker carregado em {rerank_time:.2f}s")
        logger.info(f"    Scores: {result.scores}")
        logger.info(f"    Rankings: {result.rankings}")

    except Exception as e:
        logger.error(f"    ERRO ao carregar BGE-Reranker: {e}")
        return 1

    # -------------------------------------------------------------------------
    # Resumo
    # -------------------------------------------------------------------------
    total_time = time.perf_counter() - total_start

    logger.info("\n" + "=" * 60)
    logger.info("Warmup completo!")
    logger.info(f"  BGE-M3:      {embed_time:.2f}s")
    logger.info(f"  Reranker:    {rerank_time:.2f}s")
    logger.info(f"  Total:       {total_time:.2f}s")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
