"""
Modulo de enriquecimento de chunks com LLM.

Implementa Contextual Retrieval da Anthropic:
- context_header: Frase contextualizando o chunk
- thesis_text: Resumo/tese do dispositivo
- thesis_type: Classificacao do tipo de norma
- synthetic_questions: Perguntas que o chunk responde

Pipeline Celery para enriquecimento paralelo:
    # 1. Inicie o Redis
    docker run -d --name redis -p 6379:6379 redis:alpine

    # 2. Inicie o worker Celery
    cd extracao
    celery -A src.enrichment.celery_app worker --loglevel=info --concurrency=2

    # 3. Dispare as tasks
    python scripts/run_enrichment_celery.py

    # 4. (Opcional) Monitore com Flower
    celery -A src.enrichment.celery_app flower
"""

# Imports sob demanda para evitar problemas de path com Celery
__all__ = [
    "ChunkEnricher",
    "DocumentMetadata",
    "EnrichmentResult",
]

def __getattr__(name):
    """Lazy imports para evitar erros de modulo."""
    if name in ("ChunkEnricher", "DocumentMetadata", "EnrichmentResult"):
        from .chunk_enricher import ChunkEnricher, DocumentMetadata, EnrichmentResult
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
