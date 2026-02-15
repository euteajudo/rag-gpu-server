"""
DEPRECATED (15/02/2026) - Este módulo é código morto.

O pipeline de sync Milvus → Neo4j agora usa:
  extracao/src/graph/sync_service.py (Neo4jSyncService)

Este arquivo criava nós :LegalNode e :Document com document_ids contendo
pontos (ex: LEI-14.133-2021), formato incompatível com o sistema atual.
Mantido apenas como referência para testes unitários legados.

---

(Original) Writer para Neo4j - edges (relações).

PR3 v2 - Hard Reset RAG Architecture
PR3 v2.1 - Patches de robustez (confidence tiers, document_version, extraction_method padronizado)

Neo4j armazena relações lógicas entre dispositivos legais.
Usa logical_node_id (SEM @Pxx) como identificador dos nós.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfidenceTier(str, Enum):
    """
    Tiers de confiança para edges.

    Usado para filtros e queries eficientes.
    """
    HIGH = "HIGH"      # >= 0.8: regex exato, NLI confirmado
    MEDIUM = "MEDIUM"  # 0.5 - 0.8: heurística, contexto
    LOW = "LOW"        # < 0.5: inferência, incerto

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceTier":
        """Converte score numérico para tier."""
        if score >= 0.8:
            return cls.HIGH
        elif score >= 0.5:
            return cls.MEDIUM
        return cls.LOW


class ExtractionMethod(str, Enum):
    """
    Métodos de extração padronizados.

    Evita variações como "regex" vs "Regex" vs "REGEX".
    """
    REGEX = "REGEX"          # Expressão regular exata
    HEURISTIC = "HEURISTIC"  # Regras heurísticas
    NLI = "NLI"              # Natural Language Inference
    MANUAL = "MANUAL"        # Anotação manual
    UNKNOWN = "UNKNOWN"      # Fallback

    @classmethod
    def from_string(cls, value: str) -> "ExtractionMethod":
        """Converte string para ExtractionMethod padronizado."""
        if not value:
            return cls.UNKNOWN

        normalized = value.upper().strip()

        mapping = {
            "REGEX": cls.REGEX,
            "HEURISTIC": cls.HEURISTIC,
            "NLI": cls.NLI,
            "MANUAL": cls.MANUAL,
        }

        return mapping.get(normalized, cls.UNKNOWN)


@dataclass
class LegalNodePayload:
    """
    Payload para criação/atualização de nó no Neo4j.

    Nós representam dispositivos legais (artigos, parágrafos, etc.)

    PR3 v2.1: Adicionado document_version para custódia.
    """

    # Identificação (obrigatórios)
    node_id: str  # logical_node_id (SEM @Pxx)
    document_id: str
    span_id: str

    # Tipo
    device_type: str  # ART|PAR|INC|ALI (padronizado)
    document_type: str  # LEI, DECRETO, IN

    # Texto (resumo para display)
    text_preview: str = ""  # Primeiros 200 chars

    # Metadados
    article_number: Optional[str] = None

    # Custódia e Proveniência (PR3 v2.1)
    document_version: Optional[str] = None  # Versão do documento
    ingest_run_id: Optional[str] = None
    pipeline_version: Optional[str] = None


@dataclass
class EdgeCandidate:
    """
    Candidato a edge (relação) entre dois nós.

    Extraído do texto de um chunk físico.

    PR3 v2.1 Patches:
    - confidence_tier derivado automaticamente do score
    - extraction_method padronizado em CAPS
    - document_id e document_version para custódia
    - pipeline_version para rastreabilidade
    """

    # Nós (obrigatórios)
    source_node_id: str  # logical_node_id do chunk que contém a citação
    target_node_id: str  # logical_node_id do dispositivo citado

    # Metadados da relação
    relation_type: str = "CITA"  # Tipo da relação
    confidence: float = 1.0  # Confiança da extração (score numérico)
    confidence_tier: str = ""  # HIGH|MEDIUM|LOW (derivado do score)
    extraction_method: str = "REGEX"  # Padronizado: REGEX|HEURISTIC|NLI|MANUAL

    # Contexto da citação
    citation_text: str = ""  # Texto exato da citação
    context_text: str = ""  # Contexto ao redor (para debug)

    # Proveniência
    source_chunk_id: str = ""  # chunk_id (com @Pxx) de onde foi extraído
    ingest_run_id: Optional[str] = None

    # Custódia (PR3 v2.1)
    document_id: str = ""  # document_id do documento fonte
    document_version: str = ""  # Versão do documento (hash ou timestamp)
    pipeline_version: str = ""  # Versão do pipeline (ex: "pr3-v2.1")

    def __post_init__(self):
        """Validações e normalizações pós-inicialização."""
        # Deriva confidence_tier do score
        if not self.confidence_tier:
            self.confidence_tier = ConfidenceTier.from_score(self.confidence).value

        # Normaliza extraction_method para CAPS
        if self.extraction_method:
            normalized = ExtractionMethod.from_string(self.extraction_method)
            self.extraction_method = normalized.value


class Neo4jEdgeWriter:
    """
    Writer para inserção de nós e edges no Neo4j.

    Usa MERGE para garantir idempotência.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ):
        """
        Inicializa o writer.

        Args:
            uri: URI do Neo4j
            user: Usuário
            password: Senha
            database: Nome do database
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None
        self._connected = False

    def _ensure_connected(self) -> None:
        """Conecta ao Neo4j se necessário."""
        if self._connected:
            return

        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )

            # Verifica conexão
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")

            self._connected = True
            logger.info(f"Conectado ao Neo4j: {self.uri}")

        except Exception as e:
            logger.error(f"Erro ao conectar ao Neo4j: {e}")
            raise

    def upsert_node(self, node: LegalNodePayload) -> bool:
        """
        Insere ou atualiza um nó no Neo4j.

        Args:
            node: Payload do nó

        Returns:
            True se sucesso
        """
        self._ensure_connected()

        query = """
        MERGE (n:LegalNode {node_id: $node_id})
        SET n.document_id = $document_id,
            n.span_id = $span_id,
            n.device_type = $device_type,
            n.document_type = $document_type,
            n.text_preview = $text_preview,
            n.article_number = $article_number,
            n.document_version = $document_version,
            n.ingest_run_id = $ingest_run_id,
            n.pipeline_version = $pipeline_version,
            n.updated_at = datetime()
        RETURN n.node_id
        """

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    node_id=node.node_id,
                    document_id=node.document_id,
                    span_id=node.span_id,
                    device_type=node.device_type,
                    document_type=node.document_type,
                    text_preview=node.text_preview[:200],
                    article_number=node.article_number,
                    document_version=node.document_version,
                    ingest_run_id=str(node.ingest_run_id) if node.ingest_run_id else None,
                    pipeline_version=node.pipeline_version,
                )
                result.consume()

            logger.debug(f"Nó upserted: {node.node_id}")
            return True

        except Exception as e:
            logger.error(f"Erro ao upsertar nó {node.node_id}: {e}")
            return False

    def create_edge(self, edge: EdgeCandidate) -> bool:
        """
        Cria um edge (relação) entre dois nós.

        Usa MERGE para evitar duplicatas.
        Garante que ambos os nós existem antes de criar o edge.

        Args:
            edge: Candidato a edge

        Returns:
            True se sucesso
        """
        self._ensure_connected()

        # Previne self-loops
        if edge.source_node_id == edge.target_node_id:
            logger.warning(f"Self-loop ignorado: {edge.source_node_id}")
            return False

        query = """
        MERGE (source:LegalNode {node_id: $source_node_id})
        ON CREATE SET source.stub = true,
                      source.created_at = datetime()
        MERGE (target:LegalNode {node_id: $target_node_id})
        ON CREATE SET target.stub = true,
                      target.created_at = datetime()
        MERGE (source)-[r:CITA]->(target)
        SET r.confidence = $confidence,
            r.confidence_tier = $confidence_tier,
            r.extraction_method = $extraction_method,
            r.citation_text = $citation_text,
            r.source_chunk_id = $source_chunk_id,
            r.ingest_run_id = $ingest_run_id,
            r.document_id = $document_id,
            r.document_version = $document_version,
            r.pipeline_version = $pipeline_version,
            r.updated_at = datetime()
        RETURN type(r)
        """

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    confidence=edge.confidence,
                    confidence_tier=edge.confidence_tier,
                    extraction_method=edge.extraction_method,
                    citation_text=edge.citation_text[:500],
                    source_chunk_id=edge.source_chunk_id,
                    ingest_run_id=str(edge.ingest_run_id) if edge.ingest_run_id else None,
                    document_id=edge.document_id,
                    document_version=edge.document_version,
                    pipeline_version=edge.pipeline_version,
                )
                result.consume()

            logger.debug(f"Edge criado: {edge.source_node_id} -> {edge.target_node_id}")
            return True

        except Exception as e:
            logger.error(
                f"Erro ao criar edge {edge.source_node_id} -> {edge.target_node_id}: {e}"
            )
            return False

    def create_edges_batch(self, edges: list[EdgeCandidate]) -> int:
        """
        Cria múltiplos edges em batch.

        Args:
            edges: Lista de candidatos a edge

        Returns:
            Número de edges criados com sucesso
        """
        self._ensure_connected()

        if not edges:
            return 0

        # Filtra self-loops
        valid_edges = [e for e in edges if e.source_node_id != e.target_node_id]
        skipped = len(edges) - len(valid_edges)

        if skipped > 0:
            logger.warning(f"{skipped} self-loops ignorados")

        # Usa UNWIND para batch
        query = """
        UNWIND $edges AS edge
        MERGE (source:LegalNode {node_id: edge.source_node_id})
        ON CREATE SET source.stub = true,
                      source.created_at = datetime()
        MERGE (target:LegalNode {node_id: edge.target_node_id})
        ON CREATE SET target.stub = true,
                      target.created_at = datetime()
        MERGE (source)-[r:CITA]->(target)
        SET r.confidence = edge.confidence,
            r.confidence_tier = edge.confidence_tier,
            r.extraction_method = edge.extraction_method,
            r.citation_text = edge.citation_text,
            r.source_chunk_id = edge.source_chunk_id,
            r.ingest_run_id = edge.ingest_run_id,
            r.document_id = edge.document_id,
            r.document_version = edge.document_version,
            r.pipeline_version = edge.pipeline_version,
            r.updated_at = datetime()
        RETURN count(r) AS created
        """

        try:
            edges_data = [
                {
                    "source_node_id": e.source_node_id,
                    "target_node_id": e.target_node_id,
                    "confidence": e.confidence,
                    "confidence_tier": e.confidence_tier,
                    "extraction_method": e.extraction_method,
                    "citation_text": e.citation_text[:500],
                    "source_chunk_id": e.source_chunk_id,
                    "ingest_run_id": str(e.ingest_run_id) if e.ingest_run_id else None,
                    "document_id": e.document_id,
                    "document_version": e.document_version,
                    "pipeline_version": e.pipeline_version,
                }
                for e in valid_edges
            ]

            with self._driver.session(database=self.database) as session:
                result = session.run(query, edges=edges_data)
                record = result.single()
                created = record["created"] if record else 0

            logger.info(f"Batch de {created} edges criado")
            return created

        except Exception as e:
            logger.error(f"Erro ao criar batch de edges: {e}")
            return 0

    def close(self) -> None:
        """Fecha conexão com Neo4j."""
        if self._driver:
            self._driver.close()
            self._connected = False
            logger.info("Conexão Neo4j fechada")

    def __enter__(self):
        """Context manager entry."""
        self._ensure_connected()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
