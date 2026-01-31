"""
Versionamento do pipeline e geração de IDs de execução.

PR3 v2 - Hard Reset RAG Architecture
"""

import os
import subprocess
import uuid

# Schema version para identificar formato dos dados
SCHEMA_VERSION = "2.0.0"


def get_pipeline_version() -> str:
    """
    Retorna a versão do pipeline.

    Prioridade:
    1. Variável de ambiente PIPELINE_VERSION
    2. Git SHA curto do HEAD
    3. "unknown" se não conseguir determinar

    Returns:
        String com a versão (ex: "abc1234" ou "v1.0.0")
    """
    # Primeiro tenta variável de ambiente (útil em containers)
    env_version = os.getenv("PIPELINE_VERSION")
    if env_version:
        return env_version

    # Tenta obter SHA do git
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return "unknown"


def generate_ingest_run_id() -> str:
    """
    Gera um ID único para uma execução de ingestão.

    Este ID é usado como "cola" para vincular:
    - Chunks no Milvus
    - Nodes e edges no Neo4j
    - Registro no PostgreSQL

    Returns:
        UUID v4 como string (ex: "550e8400-e29b-41d4-a716-446655440000")
    """
    return str(uuid.uuid4())


def get_version_info() -> dict:
    """
    Retorna informações completas de versão para logging/debug.

    Returns:
        Dict com pipeline_version, schema_version
    """
    return {
        "pipeline_version": get_pipeline_version(),
        "schema_version": SCHEMA_VERSION,
    }
