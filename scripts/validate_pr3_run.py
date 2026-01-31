#!/usr/bin/env python3
"""
Script de validação end-to-end para PR3 v2.

Valida que toda a cadeia de custódia está correta:
1. PostgreSQL Registry - documento registrado com status correto
2. MinIO - objetos (PDF, canonical.md, manifest.json) existem
3. Milvus - chunks físicos com campos obrigatórios
4. Neo4j - edges com source_chunk_id apontando para Milvus

Uso:
    python scripts/validate_pr3_run.py --document-id LEI-14133-2021
    python scripts/validate_pr3_run.py --document-id LEI-14133-2021 --verbose
    python scripts/validate_pr3_run.py --ingest-run-id <uuid>
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@dataclass
class ValidationResult:
    """Resultado de uma validação."""

    component: str
    check: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Relatório completo de validação."""

    document_id: str
    started_at: datetime
    results: list[ValidationResult] = field(default_factory=list)
    passed: bool = True

    def add(self, result: ValidationResult) -> None:
        """Adiciona resultado e atualiza status geral."""
        self.results.append(result)
        if not result.passed:
            self.passed = False

    def summary(self) -> str:
        """Gera resumo do relatório."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        status = "✅ PASS" if self.passed else "❌ FAIL"

        lines = [
            "=" * 70,
            f"VALIDAÇÃO PR3 v2 - {self.document_id}",
            "=" * 70,
            f"Status: {status}",
            f"Checks: {passed} passed, {failed} failed",
            f"Tempo: {(datetime.now() - self.started_at).total_seconds():.2f}s",
            "",
        ]

        for r in self.results:
            icon = "✅" if r.passed else "❌"
            lines.append(f"{icon} [{r.component}] {r.check}: {r.message}")

            if not r.passed and r.details:
                for k, v in r.details.items():
                    lines.append(f"      {k}: {v}")

        lines.append("=" * 70)
        return "\n".join(lines)


class PR3Validator:
    """
    Validador end-to-end para PR3 v2.

    Verifica consistência entre PostgreSQL, MinIO, Milvus e Neo4j.
    """

    def __init__(
        self,
        postgres_uri: Optional[str] = None,
        minio_endpoint: Optional[str] = None,
        minio_access_key: Optional[str] = None,
        minio_secret_key: Optional[str] = None,
        minio_bucket: Optional[str] = None,
        milvus_host: Optional[str] = None,
        milvus_port: Optional[int] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        verbose: bool = False,
    ):
        # PostgreSQL
        self.postgres_uri = postgres_uri or os.getenv(
            "DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag_legal"
        )

        # MinIO
        self.minio_endpoint = minio_endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.minio_access_key = minio_access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.minio_secret_key = minio_secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.minio_bucket = minio_bucket or os.getenv("MINIO_BUCKET", "documents")

        # Milvus
        self.milvus_host = milvus_host or os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = milvus_port or int(os.getenv("MILVUS_PORT", "19530"))

        # Neo4j
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def validate(self, document_id: str) -> ValidationReport:
        """
        Executa validação completa para um documento.

        Args:
            document_id: ID do documento (ex: LEI-14133-2021)

        Returns:
            ValidationReport com todos os resultados
        """
        report = ValidationReport(document_id=document_id, started_at=datetime.now())

        # 1. PostgreSQL Registry
        self._validate_postgres(document_id, report)

        # 2. MinIO Objects
        self._validate_minio(document_id, report)

        # 3. Milvus Chunks
        milvus_node_ids = self._validate_milvus(document_id, report)

        # 4. Neo4j Edges
        self._validate_neo4j(document_id, milvus_node_ids, report)

        return report

    def _validate_postgres(self, document_id: str, report: ValidationReport) -> None:
        """Valida registro no PostgreSQL."""
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(self.postgres_uri)

            with engine.connect() as conn:
                # Busca documento
                result = conn.execute(
                    text("""
                        SELECT document_id, version, status, ingest_run_id,
                               sha256_source, sha256_canonical_md,
                               minio_source_key, minio_canonical_key,
                               chunk_count, edge_count, error_message
                        FROM documents
                        WHERE document_id = :doc_id
                        ORDER BY version DESC
                        LIMIT 1
                    """),
                    {"doc_id": document_id},
                )
                row = result.fetchone()

            if not row:
                report.add(
                    ValidationResult(
                        component="PostgreSQL",
                        check="document_exists",
                        passed=False,
                        message=f"Documento não encontrado: {document_id}",
                    )
                )
                return

            doc = dict(row._mapping)

            # Check: documento existe
            report.add(
                ValidationResult(
                    component="PostgreSQL",
                    check="document_exists",
                    passed=True,
                    message=f"Documento encontrado (v{doc['version']})",
                    details={"status": doc["status"], "ingest_run_id": str(doc["ingest_run_id"])},
                )
            )

            # Check: status final
            valid_statuses = ["indexed", "graph_synced", "completed"]
            status_ok = doc["status"] in valid_statuses
            report.add(
                ValidationResult(
                    component="PostgreSQL",
                    check="status_valid",
                    passed=status_ok,
                    message=f"Status: {doc['status']}",
                    details={"expected": valid_statuses} if not status_ok else {},
                )
            )

            # Check: hashes presentes
            hashes_ok = bool(doc["sha256_source"] and doc["sha256_canonical_md"])
            report.add(
                ValidationResult(
                    component="PostgreSQL",
                    check="hashes_present",
                    passed=hashes_ok,
                    message="Hashes SHA256 presentes" if hashes_ok else "Hashes faltando",
                    details={
                        "sha256_source": doc["sha256_source"][:16] + "..." if doc["sha256_source"] else None,
                        "sha256_canonical_md": doc["sha256_canonical_md"][:16] + "..." if doc["sha256_canonical_md"] else None,
                    },
                )
            )

            # Check: MinIO keys presentes
            keys_ok = bool(doc["minio_source_key"] and doc["minio_canonical_key"])
            report.add(
                ValidationResult(
                    component="PostgreSQL",
                    check="minio_keys_present",
                    passed=keys_ok,
                    message="MinIO keys registradas" if keys_ok else "MinIO keys faltando",
                )
            )

            # Check: contagens
            if doc["chunk_count"] and doc["chunk_count"] > 0:
                report.add(
                    ValidationResult(
                        component="PostgreSQL",
                        check="chunk_count",
                        passed=True,
                        message=f"{doc['chunk_count']} chunks registrados",
                    )
                )
            else:
                report.add(
                    ValidationResult(
                        component="PostgreSQL",
                        check="chunk_count",
                        passed=False,
                        message="chunk_count é 0 ou NULL",
                    )
                )

        except ImportError:
            report.add(
                ValidationResult(
                    component="PostgreSQL",
                    check="connection",
                    passed=False,
                    message="SQLAlchemy não instalado",
                )
            )
        except Exception as e:
            report.add(
                ValidationResult(
                    component="PostgreSQL",
                    check="connection",
                    passed=False,
                    message=f"Erro de conexão: {e}",
                )
            )

    def _validate_minio(self, document_id: str, report: ValidationReport) -> None:
        """Valida objetos no MinIO."""
        try:
            from minio import Minio

            client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=False,
            )

            # Objetos esperados
            expected_objects = [
                f"documents/{document_id}/source.pdf",
                f"documents/{document_id}/canonical.md",
                f"documents/{document_id}/manifest.json",
            ]

            for obj_key in expected_objects:
                try:
                    stat = client.stat_object(self.minio_bucket, obj_key)
                    report.add(
                        ValidationResult(
                            component="MinIO",
                            check=f"object_{obj_key.split('/')[-1]}",
                            passed=True,
                            message=f"Objeto existe: {obj_key}",
                            details={"size": stat.size, "etag": stat.etag[:8] + "..."},
                        )
                    )
                except Exception:
                    report.add(
                        ValidationResult(
                            component="MinIO",
                            check=f"object_{obj_key.split('/')[-1]}",
                            passed=False,
                            message=f"Objeto não encontrado: {obj_key}",
                        )
                    )

            # Valida manifest
            try:
                manifest_key = f"documents/{document_id}/manifest.json"
                response = client.get_object(self.minio_bucket, manifest_key)
                manifest = json.loads(response.read().decode("utf-8"))
                response.close()

                # Verifica campos do manifest
                required_fields = ["document_id", "ingest_run_id", "sha256_source", "sha256_canonical"]
                missing = [f for f in required_fields if f not in manifest]

                if missing:
                    report.add(
                        ValidationResult(
                            component="MinIO",
                            check="manifest_valid",
                            passed=False,
                            message=f"Campos faltando no manifest: {missing}",
                        )
                    )
                else:
                    report.add(
                        ValidationResult(
                            component="MinIO",
                            check="manifest_valid",
                            passed=True,
                            message="Manifest válido",
                            details={
                                "span_count": manifest.get("span_count"),
                                "chunk_count": manifest.get("chunk_count"),
                                "edge_count": manifest.get("edge_count"),
                            },
                        )
                    )

            except Exception as e:
                report.add(
                    ValidationResult(
                        component="MinIO",
                        check="manifest_valid",
                        passed=False,
                        message=f"Erro ao ler manifest: {e}",
                    )
                )

        except ImportError:
            report.add(
                ValidationResult(
                    component="MinIO",
                    check="connection",
                    passed=False,
                    message="minio SDK não instalado",
                )
            )
        except Exception as e:
            report.add(
                ValidationResult(
                    component="MinIO",
                    check="connection",
                    passed=False,
                    message=f"Erro de conexão: {e}",
                )
            )

    def _validate_milvus(self, document_id: str, report: ValidationReport) -> set[str]:
        """
        Valida chunks no Milvus.

        Returns:
            Set de node_ids encontrados (para validação Neo4j)
        """
        node_ids = set()

        try:
            from pymilvus import Collection, connections

            connections.connect(host=self.milvus_host, port=self.milvus_port)

            # Tenta leis_v4 primeiro, depois leis_v3
            collection_name = None
            for name in ["leis_v4", "leis_v3", "acordaos_v1"]:
                try:
                    col = Collection(name)
                    col.load()
                    # Testa se tem o documento
                    result = col.query(
                        expr=f'document_id == "{document_id}"',
                        output_fields=["node_id"],
                        limit=1,
                    )
                    if result:
                        collection_name = name
                        break
                except Exception:
                    continue

            if not collection_name:
                report.add(
                    ValidationResult(
                        component="Milvus",
                        check="collection_found",
                        passed=False,
                        message=f"Documento não encontrado em nenhuma collection",
                    )
                )
                return node_ids

            collection = Collection(collection_name)
            collection.load()

            # Busca todos os chunks do documento
            chunks = collection.query(
                expr=f'document_id == "{document_id}"',
                output_fields=[
                    "node_id",
                    "chunk_id",
                    "span_id",
                    "device_type",
                    "text",
                    "retrieval_text",
                ],
                limit=10000,
            )

            if not chunks:
                report.add(
                    ValidationResult(
                        component="Milvus",
                        check="chunks_exist",
                        passed=False,
                        message=f"Nenhum chunk encontrado para {document_id}",
                    )
                )
                return node_ids

            report.add(
                ValidationResult(
                    component="Milvus",
                    check="chunks_exist",
                    passed=True,
                    message=f"{len(chunks)} chunks encontrados em {collection_name}",
                )
            )

            # Coleta node_ids para validação Neo4j
            for chunk in chunks:
                node_ids.add(chunk["node_id"])

            # Valida campos obrigatórios
            required_fields = ["node_id", "chunk_id", "span_id", "device_type", "text"]
            chunks_with_missing = []

            for chunk in chunks:
                missing = [f for f in required_fields if not chunk.get(f)]
                if missing:
                    chunks_with_missing.append({"chunk_id": chunk.get("chunk_id"), "missing": missing})

            if chunks_with_missing:
                report.add(
                    ValidationResult(
                        component="Milvus",
                        check="required_fields",
                        passed=False,
                        message=f"{len(chunks_with_missing)} chunks com campos faltando",
                        details={"examples": chunks_with_missing[:3]},
                    )
                )
            else:
                report.add(
                    ValidationResult(
                        component="Milvus",
                        check="required_fields",
                        passed=True,
                        message="Todos os campos obrigatórios presentes",
                    )
                )

            # Valida formato de node_id (deve ter @Pxx)
            invalid_node_ids = [nid for nid in node_ids if "@P" not in nid]
            if invalid_node_ids:
                report.add(
                    ValidationResult(
                        component="Milvus",
                        check="node_id_format",
                        passed=False,
                        message=f"{len(invalid_node_ids)} node_ids sem @Pxx",
                        details={"examples": list(invalid_node_ids)[:3]},
                    )
                )
            else:
                report.add(
                    ValidationResult(
                        component="Milvus",
                        check="node_id_format",
                        passed=True,
                        message="Todos os node_ids têm formato correto (@Pxx)",
                    )
                )

            # Valida distribuição de device_types
            device_counts = {}
            for chunk in chunks:
                dt = chunk.get("device_type", "unknown")
                device_counts[dt] = device_counts.get(dt, 0) + 1

            report.add(
                ValidationResult(
                    component="Milvus",
                    check="device_types",
                    passed=True,
                    message=f"Distribuição: {device_counts}",
                )
            )

            connections.disconnect("default")

        except ImportError:
            report.add(
                ValidationResult(
                    component="Milvus",
                    check="connection",
                    passed=False,
                    message="pymilvus não instalado",
                )
            )
        except Exception as e:
            report.add(
                ValidationResult(
                    component="Milvus",
                    check="connection",
                    passed=False,
                    message=f"Erro de conexão: {e}",
                )
            )

        return node_ids

    def _validate_neo4j(
        self, document_id: str, milvus_node_ids: set[str], report: ValidationReport
    ) -> None:
        """Valida edges no Neo4j e verifica referência ao Milvus."""
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )

            with driver.session() as session:
                # Conta nós do documento
                result = session.run(
                    """
                    MATCH (n:LegalNode)
                    WHERE n.document_id = $doc_id
                    RETURN count(n) as count
                    """,
                    doc_id=document_id,
                )
                node_count = result.single()["count"]

                if node_count == 0:
                    report.add(
                        ValidationResult(
                            component="Neo4j",
                            check="nodes_exist",
                            passed=False,
                            message=f"Nenhum nó encontrado para {document_id}",
                        )
                    )
                else:
                    report.add(
                        ValidationResult(
                            component="Neo4j",
                            check="nodes_exist",
                            passed=True,
                            message=f"{node_count} nós encontrados",
                        )
                    )

                # Conta edges do documento
                result = session.run(
                    """
                    MATCH (source:LegalNode)-[r:CITA]->(target:LegalNode)
                    WHERE source.document_id = $doc_id
                    RETURN count(r) as count
                    """,
                    doc_id=document_id,
                )
                edge_count = result.single()["count"]

                report.add(
                    ValidationResult(
                        component="Neo4j",
                        check="edges_count",
                        passed=True,
                        message=f"{edge_count} edges :CITA encontrados",
                    )
                )

                # Verifica self-loops
                result = session.run(
                    """
                    MATCH (n:LegalNode)-[r:CITA]->(n)
                    WHERE n.document_id = $doc_id
                    RETURN count(r) as count
                    """,
                    doc_id=document_id,
                )
                self_loops = result.single()["count"]

                if self_loops > 0:
                    report.add(
                        ValidationResult(
                            component="Neo4j",
                            check="no_self_loops",
                            passed=False,
                            message=f"ERRO: {self_loops} self-loops detectados!",
                        )
                    )
                else:
                    report.add(
                        ValidationResult(
                            component="Neo4j",
                            check="no_self_loops",
                            passed=True,
                            message="Nenhum self-loop",
                        )
                    )

                # Busca 5 exemplos de edges com source_chunk_id
                result = session.run(
                    """
                    MATCH (source:LegalNode)-[r:CITA]->(target:LegalNode)
                    WHERE source.document_id = $doc_id
                    RETURN source.node_id as source_node_id,
                           target.node_id as target_node_id,
                           r.source_chunk_id as source_chunk_id,
                           r.confidence as confidence,
                           r.extraction_method as method,
                           r.citation_text as citation_text
                    LIMIT 5
                    """,
                    doc_id=document_id,
                )
                edges = [dict(record) for record in result]

                if edges:
                    # Verifica se source_chunk_id aponta para Milvus
                    edges_with_valid_ref = 0
                    edges_with_invalid_ref = 0

                    for edge in edges:
                        source_chunk_id = edge.get("source_chunk_id")
                        if source_chunk_id:
                            # source_chunk_id deve estar no Milvus (ou ser derivável)
                            # Extrai logical_node_id do source_chunk_id e verifica
                            if source_chunk_id in milvus_node_ids:
                                edges_with_valid_ref += 1
                            else:
                                # Tenta match parcial (node_id pode ter prefixo diferente)
                                base_id = source_chunk_id.split("@")[0] if "@" in source_chunk_id else source_chunk_id
                                if any(base_id in nid for nid in milvus_node_ids):
                                    edges_with_valid_ref += 1
                                else:
                                    edges_with_invalid_ref += 1
                        else:
                            edges_with_invalid_ref += 1

                    if edges_with_invalid_ref > 0:
                        report.add(
                            ValidationResult(
                                component="Neo4j",
                                check="source_chunk_id_valid",
                                passed=False,
                                message=f"{edges_with_invalid_ref}/{len(edges)} edges com referência inválida ao Milvus",
                            )
                        )
                    else:
                        report.add(
                            ValidationResult(
                                component="Neo4j",
                                check="source_chunk_id_valid",
                                passed=True,
                                message=f"Todos os {len(edges)} edges amostrados têm referência válida",
                            )
                        )

                    # Mostra exemplos
                    report.add(
                        ValidationResult(
                            component="Neo4j",
                            check="edge_samples",
                            passed=True,
                            message=f"{len(edges)} exemplos de edges",
                            details={"samples": edges},
                        )
                    )

            driver.close()

        except ImportError:
            report.add(
                ValidationResult(
                    component="Neo4j",
                    check="connection",
                    passed=False,
                    message="neo4j driver não instalado",
                )
            )
        except Exception as e:
            report.add(
                ValidationResult(
                    component="Neo4j",
                    check="connection",
                    passed=False,
                    message=f"Erro de conexão: {e}",
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description="Validação end-to-end PR3 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python validate_pr3_run.py --document-id LEI-14133-2021
    python validate_pr3_run.py --document-id IN-58-2022 --verbose
        """,
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help="ID do documento para validar (ex: LEI-14133-2021)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostra detalhes adicionais",
    )
    parser.add_argument(
        "--postgres-uri",
        help="URI do PostgreSQL (default: $DATABASE_URL)",
    )
    parser.add_argument(
        "--milvus-host",
        help="Host do Milvus (default: $MILVUS_HOST ou localhost)",
    )
    parser.add_argument(
        "--neo4j-uri",
        help="URI do Neo4j (default: $NEO4J_URI ou bolt://localhost:7687)",
    )

    args = parser.parse_args()

    # Configura logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Executa validação
    validator = PR3Validator(
        postgres_uri=args.postgres_uri,
        milvus_host=args.milvus_host,
        neo4j_uri=args.neo4j_uri,
        verbose=args.verbose,
    )

    report = validator.validate(args.document_id)

    # Imprime relatório
    print(report.summary())

    # Imprime detalhes dos edges se verbose
    if args.verbose:
        print("\n--- DETALHES DOS EDGES ---")
        for r in report.results:
            if r.check == "edge_samples" and r.details.get("samples"):
                for i, edge in enumerate(r.details["samples"], 1):
                    print(f"\nEdge {i}:")
                    print(f"  source: {edge.get('source_node_id')}")
                    print(f"  target: {edge.get('target_node_id')}")
                    print(f"  source_chunk_id: {edge.get('source_chunk_id')}")
                    print(f"  confidence: {edge.get('confidence')}")
                    print(f"  method: {edge.get('method')}")
                    citation = edge.get("citation_text", "")
                    if citation:
                        print(f"  citation: {citation[:100]}...")

    # Exit code
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
