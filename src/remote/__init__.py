"""
Remote Clients - Clientes para GPU Server remoto.

Arquitetura:
    - GPU Server: Embeddings (BGE-M3) + Reranking (BGE-Reranker)
    - vLLM Server: LLM (Qwen3-8B-AWQ)

Uso:
    from remote import RemoteEmbedder, RemoteReranker, RemoteLLM

    # Embeddings
    embedder = RemoteEmbedder()
    result = embedder.encode(["texto1", "texto2"])
    print(result.dense_embeddings)  # [[0.1, 0.2, ...], ...]
    print(result.sparse_embeddings)  # [{1: 0.5, 3: 0.2}, ...]

    # Reranking
    reranker = RemoteReranker()
    result = reranker.rerank("query", ["doc1", "doc2"])
    print(result.scores)    # [0.95, 0.72]
    print(result.rankings)  # [0, 1]

    # LLM
    llm = RemoteLLM()
    response = llm.chat([{"role": "user", "content": "Olá"}])
    print(response.content)  # "Olá! Como posso ajudar?"

Configuração:
    Use variáveis de ambiente:

    export GPU_SERVER_URL=https://gpu-server-xxx.run.app
    export VLLM_BASE_URL=http://xxx.runpod.io:8000/v1

    # Ou configure manualmente:
    config = RemoteEmbedderConfig(gpu_server_url="https://...")
    embedder = RemoteEmbedder(config)
"""

from .embedder import RemoteEmbedder, RemoteEmbedderConfig, EmbeddingResult
from .reranker import RemoteReranker, RemoteRerankerConfig, RerankResult
from .llm import RemoteLLM, RemoteLLMConfig, LLMResponse

__all__ = [
    # Embedder
    "RemoteEmbedder",
    "RemoteEmbedderConfig",
    "EmbeddingResult",
    # Reranker
    "RemoteReranker",
    "RemoteRerankerConfig",
    "RerankResult",
    # LLM
    "RemoteLLM",
    "RemoteLLMConfig",
    "LLMResponse",
]
