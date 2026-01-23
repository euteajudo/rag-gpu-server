"""
Modulo de integracao com LLMs.

Clientes para vLLM e Ollama com API OpenAI-compatible.
"""

from .vllm_client import VLLMClient, LLMConfig

__all__ = ["VLLMClient", "LLMConfig"]
