"""
Módulo de sinks (destinos de dados).

PR3 v2 - Hard Reset RAG Architecture

Sinks disponíveis:
- MilvusWriter: Chunks físicos para busca vetorial
- Neo4jEdgeWriter: Relações lógicas no grafo
"""

from .milvus_writer import MilvusWriter, MilvusChunk
from .neo4j_writer import Neo4jEdgeWriter, EdgeCandidate, LegalNodePayload

__all__ = [
    # Milvus
    "MilvusWriter",
    "MilvusChunk",
    # Neo4j
    "Neo4jEdgeWriter",
    "EdgeCandidate",
    "LegalNodePayload",
]
