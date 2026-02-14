"""
Remote LLM - Cliente para vLLM (API OpenAI-compatible).

Chama o endpoint /v1/chat/completions do vLLM para geração de texto.

Uso:
    from remote import RemoteLLM

    llm = RemoteLLM()

    # Chat simples
    response = llm.chat([
        {"role": "user", "content": "O que é ETP?"}
    ])

    # Com streaming
    for chunk in llm.stream([
        {"role": "user", "content": "Explique..."}
    ]):
        print(chunk, end="", flush=True)
"""

import os
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Iterator, Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RemoteLLMConfig:
    """Configuração do cliente LLM remoto."""

    vllm_base_url: str = "http://localhost:8002/v1"
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    timeout: float = 300.0  # LLM pode demorar
    max_retries: int = 2
    retry_delay: float = 2.0

    # Parâmetros de geração padrão
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9

    @classmethod
    def from_env(cls) -> "RemoteLLMConfig":
        """Carrega configuração de variáveis de ambiente."""
        return cls(
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8002/v1"),
            model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            timeout=float(os.getenv("VLLM_TIMEOUT", "300")),
            max_retries=int(os.getenv("VLLM_MAX_RETRIES", "2")),
            retry_delay=float(os.getenv("VLLM_RETRY_DELAY", "2.0")),
            temperature=float(os.getenv("VLLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "2048")),
            top_p=float(os.getenv("VLLM_TOP_P", "0.9")),
        )

    @classmethod
    def for_generation(cls) -> "RemoteLLMConfig":
        """Configuração otimizada para geração de respostas."""
        config = cls.from_env()
        config.temperature = 0.7
        config.max_tokens = 2048
        return config

    @classmethod
    def for_extraction(cls) -> "RemoteLLMConfig":
        """Configuração otimizada para extração estruturada."""
        config = cls.from_env()
        config.temperature = 0.0
        config.max_tokens = 4096
        return config


@dataclass
class LLMResponse:
    """Resposta do LLM."""

    content: str
    model: str
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    finish_reason: str = "stop"


class RemoteLLM:
    """
    Cliente para vLLM - API OpenAI-compatible.

    Gera texto usando Qwen3-8B-AWQ no servidor GPU remoto.
    Compatível com a interface do VLLMClient local.
    """

    def __init__(self, config: Optional[RemoteLLMConfig] = None):
        self.config = config or RemoteLLMConfig.from_env()
        self._client: Optional[httpx.Client] = None
        self._stream_client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Cliente HTTP com lazy initialization."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.config.timeout,
                limits=httpx.Limits(max_connections=10),
            )
        return self._client

    @property
    def stream_client(self) -> httpx.Client:
        """Cliente HTTP para streaming."""
        if self._stream_client is None:
            self._stream_client = httpx.Client(
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        return self._stream_client

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Gera resposta via vLLM (chat completion).

        Args:
            messages: Lista de mensagens [{"role": "user", "content": "..."}]
            temperature: Temperatura de amostragem (0.0 = determinístico)
            max_tokens: Máximo de tokens na resposta
            top_p: Nucleus sampling
            **kwargs: Parâmetros adicionais para a API

        Returns:
            LLMResponse com conteúdo gerado

        Raises:
            httpx.HTTPError: Se falhar após retries
        """
        start_time = time.perf_counter()

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": top_p if top_p is not None else self.config.top_p,
            **kwargs,
        }

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post(
                    f"{self.config.vllm_base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})

                elapsed = (time.perf_counter() - start_time) * 1000
                logger.debug(
                    f"LLM remoto: {data.get('usage', {}).get('total_tokens', 0)} tokens "
                    f"em {elapsed:.2f}ms"
                )

                return LLMResponse(
                    content=message.get("content", ""),
                    model=data.get("model", self.config.model),
                    usage=data.get("usage", {}),
                    latency_ms=elapsed,
                    finish_reason=choice.get("finish_reason", "stop"),
                )

            except httpx.HTTPError as e:
                last_error = e
                logger.warning(
                    f"Erro no LLM remoto (tentativa {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        raise last_error or Exception("Falha no LLM remoto")

    def chat_with_schema(
        self,
        messages: list[dict],
        schema: Any,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        """
        Gera resposta estruturada (JSON) via vLLM com guided decoding.

        Args:
            messages: Lista de mensagens
            schema: Pydantic model ou dict com JSON schema
            temperature: Temperatura (0.0 para determinístico)
            **kwargs: Parâmetros adicionais

        Returns:
            Dict com resposta parseada
        """
        # Converte Pydantic model para JSON schema se necessário
        if hasattr(schema, "model_json_schema"):
            json_schema = schema.model_json_schema()
        elif hasattr(schema, "schema"):
            json_schema = schema.schema()
        else:
            json_schema = schema

        payload_extra = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": json_schema,
                },
            },
        }

        response = self.chat(
            messages=messages,
            temperature=temperature,
            **payload_extra,
            **kwargs,
        )

        # Parseia JSON da resposta
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao parsear JSON: {e}")
            logger.error(f"Conteúdo: {response.content[:500]}")
            raise

    def stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Gera resposta em streaming via vLLM.

        Args:
            messages: Lista de mensagens
            temperature: Temperatura
            max_tokens: Máximo de tokens
            **kwargs: Parâmetros adicionais

        Yields:
            Chunks de texto conforme são gerados
        """
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "stream": True,
            **kwargs,
        }

        with self.stream_client.stream(
            "POST",
            f"{self.config.vllm_base_url}/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or line == "data: [DONE]":
                    continue

                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Gera texto a partir de prompt simples.

        Args:
            prompt: Prompt de texto
            **kwargs: Parâmetros adicionais

        Returns:
            Texto gerado
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, **kwargs)
        return response.content

    def health_check(self) -> dict:
        """Verifica status do vLLM."""
        try:
            response = self.client.get(
                f"{self.config.vllm_base_url}/models",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            models = data.get("data", [])
            model_ids = [m.get("id", "") for m in models]

            return {
                "status": "online",
                "server_url": self.config.vllm_base_url,
                "models": model_ids,
                "configured_model": self.config.model,
            }

        except Exception as e:
            return {
                "status": "offline",
                "error": str(e),
                "server_url": self.config.vllm_base_url,
            }

    def close(self):
        """Fecha os clientes HTTP."""
        if self._client:
            self._client.close()
            self._client = None
        if self._stream_client:
            self._stream_client.close()
            self._stream_client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Singleton
_remote_llm: Optional[RemoteLLM] = None


def get_remote_llm(config: Optional[RemoteLLMConfig] = None) -> RemoteLLM:
    """Retorna instância singleton do RemoteLLM."""
    global _remote_llm
    if _remote_llm is None:
        _remote_llm = RemoteLLM(config)
    return _remote_llm
