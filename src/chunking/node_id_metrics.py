"""
Métricas e logging de node_id para observabilidade.

Este módulo fornece:
1. Contadores para métricas Prometheus (se disponível)
2. Logger de batch para auditoria
3. Resumo de ingestão com estatísticas de node_id

Uso no pipeline:
    from chunking.node_id_metrics import NodeIdMetricsCollector

    collector = NodeIdMetricsCollector(document_id="IN-65-2021")

    for chunk in chunks:
        if not chunk.get("node_id"):
            collector.record_missing()
        elif chunk["node_id"] != f"leis:{chunk['chunk_id']}":
            collector.record_mismatch()
        else:
            collector.record_valid()

    # No final da ingestão
    collector.log_summary()

@author: Equipe VectorGov
@since: 22/01/2025
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

logger = logging.getLogger(__name__)

# Tenta importar prometheus_client (opcional)
try:
    from prometheus_client import Counter, Gauge

    PROMETHEUS_AVAILABLE = True

    # Contadores Prometheus
    CHUNKS_INGESTED = Counter(
        "vectorgov_chunks_ingested_total",
        "Total de chunks ingeridos",
        ["document_id", "device_type"]
    )
    CHUNKS_MISSING_NODE_ID = Counter(
        "vectorgov_chunks_missing_node_id_total",
        "Chunks sem node_id (fallback gerado)",
        ["document_id"]
    )
    CHUNKS_NODE_ID_MISMATCH = Counter(
        "vectorgov_chunks_node_id_mismatch_total",
        "Chunks com node_id inconsistente",
        ["document_id"]
    )
    CHUNKS_VALID_NODE_ID = Counter(
        "vectorgov_chunks_valid_node_id_total",
        "Chunks com node_id válido",
        ["document_id"]
    )

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client não disponível, métricas desabilitadas")


@dataclass
class NodeIdStats:
    """Estatísticas de node_id para um documento."""

    document_id: str
    total_chunks: int = 0
    valid_node_id: int = 0
    missing_node_id: int = 0
    mismatch_node_id: int = 0
    by_device_type: Dict[str, int] = field(default_factory=dict)

    # Detalhes dos erros (para auditoria)
    missing_details: List[str] = field(default_factory=list)
    mismatch_details: List[Dict[str, str]] = field(default_factory=list)

    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável."""
        return {
            "document_id": self.document_id,
            "total_chunks": self.total_chunks,
            "valid_node_id": self.valid_node_id,
            "missing_node_id": self.missing_node_id,
            "mismatch_node_id": self.mismatch_node_id,
            "by_device_type": self.by_device_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "success_rate": self.success_rate,
        }

    @property
    def success_rate(self) -> float:
        """Taxa de sucesso (chunks válidos / total)."""
        if self.total_chunks == 0:
            return 0.0
        return (self.valid_node_id / self.total_chunks) * 100

    @property
    def has_errors(self) -> bool:
        """Verifica se houve erros."""
        return self.missing_node_id > 0 or self.mismatch_node_id > 0


class NodeIdMetricsCollector:
    """
    Coletor de métricas de node_id para um documento.

    Registra:
    - Chunks com node_id válido
    - Chunks sem node_id (fallback)
    - Chunks com node_id inconsistente

    Exporta para:
    - Prometheus (se disponível)
    - Logs estruturados
    - JSON para auditoria
    """

    def __init__(self, document_id: str, batch_id: Optional[str] = None):
        self.document_id = document_id
        self.batch_id = batch_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.stats = NodeIdStats(document_id=document_id)

    def record_valid(self, chunk_id: str, device_type: str = "unknown"):
        """Registra chunk com node_id válido."""
        self.stats.total_chunks += 1
        self.stats.valid_node_id += 1
        self.stats.by_device_type[device_type] = self.stats.by_device_type.get(device_type, 0) + 1

        if PROMETHEUS_AVAILABLE:
            CHUNKS_INGESTED.labels(document_id=self.document_id, device_type=device_type).inc()
            CHUNKS_VALID_NODE_ID.labels(document_id=self.document_id).inc()

    def record_missing(self, chunk_id: str, span_id: str = "", device_type: str = "unknown"):
        """Registra chunk sem node_id (fallback será gerado)."""
        self.stats.total_chunks += 1
        self.stats.missing_node_id += 1
        self.stats.by_device_type[device_type] = self.stats.by_device_type.get(device_type, 0) + 1

        # Guarda detalhes para auditoria (limitado a 100)
        if len(self.stats.missing_details) < 100:
            self.stats.missing_details.append(f"{chunk_id}#{span_id}")

        if PROMETHEUS_AVAILABLE:
            CHUNKS_INGESTED.labels(document_id=self.document_id, device_type=device_type).inc()
            CHUNKS_MISSING_NODE_ID.labels(document_id=self.document_id).inc()

        logger.warning(
            f"[node_id] Missing: document={self.document_id}, chunk_id={chunk_id}, "
            f"span_id={span_id}, device_type={device_type}"
        )

    def record_mismatch(
        self,
        chunk_id: str,
        actual_node_id: str,
        expected_node_id: str,
        device_type: str = "unknown"
    ):
        """Registra chunk com node_id inconsistente."""
        self.stats.total_chunks += 1
        self.stats.mismatch_node_id += 1
        self.stats.by_device_type[device_type] = self.stats.by_device_type.get(device_type, 0) + 1

        # Guarda detalhes para auditoria (limitado a 100)
        if len(self.stats.mismatch_details) < 100:
            self.stats.mismatch_details.append({
                "chunk_id": chunk_id,
                "actual": actual_node_id,
                "expected": expected_node_id,
            })

        if PROMETHEUS_AVAILABLE:
            CHUNKS_INGESTED.labels(document_id=self.document_id, device_type=device_type).inc()
            CHUNKS_NODE_ID_MISMATCH.labels(document_id=self.document_id).inc()

        logger.error(
            f"[node_id] Mismatch: document={self.document_id}, chunk_id={chunk_id}, "
            f"expected='{expected_node_id}', actual='{actual_node_id}'"
        )

    def finish(self):
        """Finaliza a coleta e registra timestamp."""
        self.stats.finished_at = datetime.utcnow().isoformat()

    def log_summary(self):
        """Loga resumo da ingestão."""
        self.finish()

        level = logging.ERROR if self.stats.has_errors else logging.INFO
        status = "WITH_ERRORS" if self.stats.has_errors else "SUCCESS"

        logger.log(
            level,
            f"[node_id] Batch {self.batch_id} {status}: "
            f"document={self.document_id}, "
            f"total={self.stats.total_chunks}, "
            f"valid={self.stats.valid_node_id}, "
            f"missing={self.stats.missing_node_id}, "
            f"mismatch={self.stats.mismatch_node_id}, "
            f"success_rate={self.stats.success_rate:.1f}%"
        )

        # Log detalhado se houver erros
        if self.stats.missing_details:
            logger.warning(
                f"[node_id] Missing node_ids (primeiros 10): "
                f"{self.stats.missing_details[:10]}"
            )

        if self.stats.mismatch_details:
            logger.error(
                f"[node_id] Mismatched node_ids (primeiros 10): "
                f"{json.dumps(self.stats.mismatch_details[:10], indent=2)}"
            )

    def to_json(self) -> str:
        """Exporta estatísticas para JSON."""
        return json.dumps(self.stats.to_dict(), indent=2, ensure_ascii=False)

    def get_stats(self) -> NodeIdStats:
        """Retorna estatísticas coletadas."""
        return self.stats


def validate_and_collect_metrics(
    chunks: List[Dict[str, Any]],
    document_id: str,
) -> NodeIdMetricsCollector:
    """
    Valida node_ids de uma lista de chunks e coleta métricas.

    Args:
        chunks: Lista de chunks (dicts com chunk_id, node_id, etc)
        document_id: ID do documento sendo processado

    Returns:
        NodeIdMetricsCollector com estatísticas coletadas
    """
    collector = NodeIdMetricsCollector(document_id=document_id)

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        node_id = chunk.get("node_id")
        expected_node_id = f"leis:{chunk_id}"
        device_type = chunk.get("device_type", "unknown")

        if not node_id:
            collector.record_missing(chunk_id, chunk.get("span_id", ""), device_type)
        elif node_id != expected_node_id:
            collector.record_mismatch(chunk_id, node_id, expected_node_id, device_type)
        else:
            collector.record_valid(chunk_id, device_type)

    collector.log_summary()
    return collector
