"""
Configurações do GPU Server.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Configuração do servidor GPU."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Rate Limiting
    gpu_rate_limit: int = 100  # Requisições por minuto por API key/IP

    # String Size Limits (segurança contra VRAM overflow)
    max_text_length: int = 10000  # Máximo de caracteres por texto

    # Models
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # vLLM (container separado)
    vllm_base_url: str = "http://localhost:8002/v1"
    vllm_model: str = "Qwen/Qwen3-VL-8B-Instruct"

    # VLM Pipeline
    use_vlm_pipeline: bool = False     # Feature flag: True = PyMuPDF+Qwen3-VL, False = Docling+SpanParser
    vlm_page_dpi: int = 300            # DPI para renderização de páginas
    vlm_max_retries: int = 3           # Retries por página no VLM

    # Hardware
    use_fp16: bool = True
    device: str = "cuda"

    # Cache
    cache_dir: str = "/root/.cache/huggingface"

    @classmethod
    def from_env(cls) -> "Config":
        """Carrega configuração de variáveis de ambiente."""
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            gpu_rate_limit=int(os.getenv("GPU_RATE_LIMIT", "100")),
            max_text_length=int(os.getenv("MAX_TEXT_LENGTH", "10000")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8002/v1"),
            vllm_model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            use_vlm_pipeline=os.getenv("USE_VLM_PIPELINE", "false").lower() == "true",
            vlm_page_dpi=int(os.getenv("VLM_PAGE_DPI", "300")),
            vlm_max_retries=int(os.getenv("VLM_MAX_RETRIES", "3")),
            use_fp16=os.getenv("USE_FP16", "true").lower() == "true",
            device=os.getenv("DEVICE", "cuda"),
            cache_dir=os.getenv("HF_HOME", "/root/.cache/huggingface"),
        )


# Singleton
config = Config.from_env()
