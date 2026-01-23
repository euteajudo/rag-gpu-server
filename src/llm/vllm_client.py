"""
Cliente vLLM com API OpenAI-compatible.

O vLLM expoe uma API compativel com OpenAI, permitindo usar
o mesmo codigo para vLLM, Ollama, ou OpenAI.

Uso:
    from llm import VLLMClient

    client = VLLMClient(base_url="http://localhost:8000/v1")

    response = client.chat([
        {"role": "system", "content": "Voce e um assistente."},
        {"role": "user", "content": "O que e ETP?"}
    ])

    print(response)
"""

import os
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Any
import httpx

logger = logging.getLogger(__name__)


def _strip_thinking_block(text: str) -> str:
    """Remove bloco <think>...</think> da resposta do Qwen 3.
    
    Tambem remove blocos incompletos (sem </think>) para evitar vazamento.
    """
    # Remove blocos completos
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.DOTALL)
    # Remove blocos incompletos (sem </think>) - pega tudo apos <think>
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.DOTALL)
    return text.strip()


# =============================================================================
# CONFIGURACAO
# =============================================================================

@dataclass
class LLMConfig:
    """Configuracao do cliente LLM."""

    # Conexao
    base_url: str = field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = "not-needed"  # vLLM nao precisa de API key
    timeout: float = 120.0  # segundos

    # Modelo
    model: str = field(default_factory=lambda: os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ"))  # Modelo padrao (LOCAL: 8B para 12GB GPU)

    # Geracao
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0

    # Thinking Mode (Qwen 3)
    enable_thinking: bool = False  # True = pensa antes, False = /no_think

    # Retry
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def for_enrichment(cls, model: str = None) -> "LLMConfig":
        """Config otimizada para enriquecimento de chunks (no_think)."""
        if model is None:
            model = __import__("os").getenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
        if model is None:
            model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
        return cls(
            model=model,
            temperature=0.0,
            max_tokens=1024,
            timeout=300.0,
            enable_thinking=True,
        )

    @classmethod
    def for_extraction(cls, model: str = None) -> "LLMConfig":
        """Config para extracao estruturada - JSON complexo (no_think)."""
        if model is None:
            model = __import__("os").getenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
        if model is None:
            model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
        return cls(
            model=model,
            temperature=0.0,
            max_tokens=12288,
            enable_thinking=True,
        )

    @classmethod
    def for_generation(cls, model: str = None) -> "LLMConfig":
        """Config para geracao de resposta ao usuario (thinking habilitado com tokens extras)."""
        if model is None:
            model = __import__("os").getenv("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
        return cls(
            model=model,
            temperature=0.3,
            max_tokens=12288,
            timeout=180.0,
            enable_thinking=True,
        )




# =============================================================================
# RESPONSE COM METRICAS
# =============================================================================

@dataclass
class LLMResponse:
    """Resposta do LLM com metricas de tokens."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0  # Tokens usados no bloco <think>
    response_tokens: int = 0  # Tokens da resposta final (sem <think>)
    elapsed_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds > 0:
            return self.completion_tokens / self.elapsed_seconds
        return 0.0

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "thinking_tokens": self.thinking_tokens,
            "response_tokens": self.response_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "tokens_per_second": round(self.tokens_per_second, 1),
        }


def _count_thinking_tokens(text: str) -> int:
    """Estima tokens no bloco <think> baseado em caracteres/4."""
    match = re.search(r"<think>[\s\S]*?</think>", text, flags=re.DOTALL)
    if match:
        # Estimativa: ~4 caracteres por token em ingles/portugues
        return len(match.group()) // 4
    return 0


# =============================================================================
# CLIENTE VLLM
# =============================================================================

class VLLMClient:
    """
    Cliente para vLLM com API OpenAI-compatible.

    Suporta:
    - Chat completions
    - Retry automatico
    - Logging de metricas
    - Compativel com Ollama tambem

    Attributes:
        config: Configuracao do cliente
        _client: Cliente HTTP
    """

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Inicializa o cliente.

        Args:
            config: Configuracao. Se nao fornecido, usa default.
            **kwargs: Sobrescreve campos da config (ex: base_url="...")
        """
        self.config = config or LLMConfig()

        # Aplica kwargs sobre a config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
        )

        thinking_mode = "thinking" if self.config.enable_thinking else "no_think"
        logger.info(f"VLLMClient inicializado: {self.config.base_url} (mode={thinking_mode})")

    def _prepare_messages(
        self,
        messages: list[dict],
        enable_thinking: Optional[bool] = None,
    ) -> list[dict]:
        """
        Prepara mensagens para envio, adicionando /no_think se necessario.

        Args:
            messages: Lista de mensagens original
            enable_thinking: Override do config (None = usa config)

        Returns:
            Mensagens preparadas (copia, nao modifica original)
        """
        use_thinking = enable_thinking if enable_thinking is not None else self.config.enable_thinking

        if use_thinking:
            # Thinking habilitado - retorna copia sem modificacao
            return [msg.copy() for msg in messages]

        # Thinking desabilitado - adiciona /no_think na ultima mensagem do user
        prepared = []
        for i, msg in enumerate(messages):
            msg_copy = msg.copy()
            # Adiciona /no_think apenas na ultima mensagem do usuario
            if msg_copy.get("role") == "user" and i == len(messages) - 1:
                content = msg_copy.get("content", "")
                if not content.endswith("/no_think"):
                    msg_copy["content"] = content + " /no_think"
            prepared.append(msg_copy)

        return prepared

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Envia mensagens e retorna resposta.

        Args:
            messages: Lista de mensagens [{"role": "...", "content": "..."}]
            temperature: Temperatura (default: config)
            max_tokens: Max tokens (default: config)
            model: Modelo (default: config)
            enable_thinking: Override do thinking mode (None = usa config)
            **kwargs: Parametros extras para a API

        Returns:
            Texto da resposta do modelo (sem bloco <think> se no_think)
        """
        # Prepara mensagens com /no_think se necessario
        prepared_messages = self._prepare_messages(messages, enable_thinking)

        payload = {
            "model": model or self.config.model,
            "messages": prepared_messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
            **kwargs,
        }

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()

                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()

                elapsed = time.time() - start_time
                data = response.json()

                # Extrai texto da resposta
                content = data["choices"][0]["message"]["content"]

                # Remove bloco <think> se presente (Qwen 3 sempre gera, mesmo vazio)
                content = _strip_thinking_block(content)

                # Log metricas
                usage = data.get("usage", {})
                logger.debug(
                    f"LLM response: {elapsed:.2f}s, "
                    f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
                    f"completion_tokens={usage.get('completion_tokens', '?')}"
                )

                return content

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

            except httpx.TimeoutException as e:
                logger.warning(f"Timeout (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

            except Exception as e:
                logger.error(f"Erro inesperado: {e}")
                raise

        raise RuntimeError("Falha apos todas as tentativas")


    def chat_with_metrics(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ) -> "LLMResponse":
        """
        Envia mensagens e retorna resposta COM metricas de tokens.

        Similar a chat(), mas retorna LLMResponse com:
        - prompt_tokens: tokens do input
        - completion_tokens: tokens totais gerados
        - thinking_tokens: tokens estimados no bloco <think>
        - response_tokens: tokens da resposta final
        - tokens_per_second: velocidade de geracao

        Args:
            messages: Lista de mensagens
            temperature: Temperatura (default: config)
            max_tokens: Max tokens (default: config)
            model: Modelo (default: config)
            enable_thinking: Override do thinking mode
            **kwargs: Parametros extras

        Returns:
            LLMResponse com texto e metricas
        """
        prepared_messages = self._prepare_messages(messages, enable_thinking)

        payload = {
            "model": model or self.config.model,
            "messages": prepared_messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
            **kwargs,
        }

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()

                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()

                elapsed = time.time() - start_time
                data = response.json()

                # Extrai texto ANTES de remover <think>
                raw_content = data["choices"][0]["message"]["content"]

                # Estima tokens no bloco <think>
                thinking_tokens = _count_thinking_tokens(raw_content)

                # Remove bloco <think>
                content = _strip_thinking_block(raw_content)

                # Metricas da API
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Tokens da resposta = total - thinking
                response_tokens = max(0, completion_tokens - thinking_tokens)

                logger.debug(
                    f"LLM response: {elapsed:.2f}s, "
                    f"prompt={prompt_tokens}, completion={completion_tokens}, "
                    f"thinking~{thinking_tokens}, response~{response_tokens}, "
                    f"speed={completion_tokens/elapsed:.1f} tok/s"
                )

                return LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    thinking_tokens=thinking_tokens,
                    response_tokens=response_tokens,
                    elapsed_seconds=elapsed,
                )

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

            except httpx.TimeoutException as e:
                logger.warning(f"Timeout (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

            except Exception as e:
                logger.error(f"Erro inesperado: {e}")
                raise

        raise RuntimeError("Falha apos todas as tentativas")

    def chat_json(
        self,
        messages: list[dict],
        **kwargs,
    ) -> dict:
        """
        Envia mensagens e retorna resposta parseada como JSON.

        NOTA: Este metodo faz parsing POS-RESPOSTA (pode falhar).
        Para extracao estruturada, use chat_with_schema() que forca
        o modelo a gerar JSON valido via guided_json.

        Args:
            messages: Lista de mensagens
            **kwargs: Parametros para chat()

        Returns:
            Dict parseado da resposta JSON
        """
        response = self.chat(messages, **kwargs)

        # Tenta parsear JSON
        response = response.strip()

        # Remove markdown code blocks se presentes
        if response.startswith("```"):
            import re
            response = re.sub(r"^```\w*\n?", "", response)
            response = re.sub(r"\n?```$", "", response)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Tenta encontrar JSON na resposta
            import re
            # Tenta objeto
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                return json.loads(match.group())
            # Tenta array
            match = re.search(r"\[[\s\S]*\]", response)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Nao foi possivel parsear JSON: {response[:200]}") from e

    def chat_with_schema(
        self,
        messages: list[dict],
        schema: type,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> dict:
        """
        Envia mensagens com response_format json_schema para forcar output estruturado.

        IMPORTANTE: Este metodo usa o recurso json_schema do vLLM que
        forca o modelo a gerar APENAS tokens validos para o schema.
        Muito mais confiavel que chat_json() para extracao estruturada.

        Uso:
            from pydantic import BaseModel

            class Article(BaseModel):
                article_number: str
                content: str

            result = client.chat_with_schema(
                messages=[{"role": "user", "content": "Extraia..."}],
                schema=Article,
            )

        Args:
            messages: Lista de mensagens
            schema: Classe Pydantic ou dict com JSON Schema
            temperature: Temperatura (default: 0 para extracao)
            max_tokens: Max tokens (default: config)
            model: Modelo (default: config)

        Returns:
            Dict validado contra o schema
        """
        # Extrai JSON Schema do Pydantic ou usa dict direto
        if hasattr(schema, "model_json_schema"):
            json_schema = schema.model_json_schema()
            schema_name = schema.__name__
        elif isinstance(schema, dict):
            json_schema = schema
            schema_name = "extraction_schema"
        else:
            raise ValueError(f"Schema deve ser Pydantic BaseModel ou dict, recebido: {type(schema)}")

        # Extracao estruturada SEMPRE usa no_think (nao precisa pensar, so extrair)
        prepared_messages = self._prepare_messages(messages, enable_thinking=True)

        payload = {
            "model": model or self.config.model,
            "messages": prepared_messages,
            "temperature": temperature if temperature is not None else 0.0,  # 0 para extracao
            "max_tokens": max_tokens or self.config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                },
            },
        }

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()

                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()

                elapsed = time.time() - start_time
                data = response.json()

                # Extrai texto da resposta
                content = data["choices"][0]["message"]["content"]

                # Remove bloco <think> se presente (mesmo com /no_think, pode vir vazio)
                content = _strip_thinking_block(content)

                # Log metricas
                usage = data.get("usage", {})
                logger.debug(
                    f"LLM json_schema response: {elapsed:.2f}s, "
                    f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
                    f"completion_tokens={usage.get('completion_tokens', '?')}"
                )

                # Com json_schema, o output JA e JSON valido
                return json.loads(content)

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

            except json.JSONDecodeError as e:
                # Isso nao deveria acontecer com json_schema
                logger.error(f"JSON invalido mesmo com json_schema: {e}")
                raise

            except Exception as e:
                logger.error(f"Erro inesperado: {e}")
                raise

        raise RuntimeError("Falha apos todas as tentativas")

    def list_models(self) -> list[str]:
        """Lista modelos disponiveis no servidor."""
        try:
            response = self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Erro ao listar modelos: {e}")
            return []

    def health_check(self) -> bool:
        """Verifica se o servidor esta respondendo."""
        try:
            models = self.list_models()
            return len(models) > 0
        except Exception:
            return False

    def close(self):
        """Fecha o cliente HTTP."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        thinking = "thinking" if self.config.enable_thinking else "no_think"
        return f"VLLMClient(url={self.config.base_url!r}, model={self.config.model!r}, mode={thinking})"


# =============================================================================
# CLIENTE OLLAMA (mesmo protocolo)
# =============================================================================

class OllamaClient(VLLMClient):
    """
    Cliente para Ollama (mesmo protocolo OpenAI-compatible).

    Uso:
        client = OllamaClient(model="qwen3:8b")
        response = client.chat([...])
    """

    def __init__(self, model: str = "qwen3:8b", **kwargs):
        config = LLMConfig(
            base_url="http://localhost:11434/v1",
            model=model,
            **{k: v for k, v in kwargs.items() if hasattr(LLMConfig, k)}
        )
        super().__init__(config=config)


# =============================================================================
# FACTORY
# =============================================================================

def get_llm_client(
    provider: str = "vllm",
    model: Optional[str] = None,
    **kwargs,
) -> VLLMClient:
    """
    Factory para criar cliente LLM.

    Args:
        provider: "vllm" ou "ollama"
        model: Nome do modelo
        **kwargs: Config adicional

    Returns:
        Cliente configurado
    """
    if provider == "ollama":
        return OllamaClient(model=model or "qwen3:8b", **kwargs)

    # vLLM default
    config = LLMConfig(model=model or "Qwen/Qwen3-8B", **kwargs)
    return VLLMClient(config=config)


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Teste do VLLMClient")
    print("=" * 60)

    # Testa conexao
    client = VLLMClient()

    print(f"\nCliente: {client}")
    print(f"Health check: {client.health_check()}")

    models = client.list_models()
    print(f"Modelos disponiveis: {models}")

    if models:
        # Teste simples
        print("\n--- Teste de Chat ---")
        response = client.chat([
            {"role": "user", "content": "O que significa ETP em licitacoes?"}
        ], max_tokens=200)
        print(f"Resposta: {response[:300]}...")

        # Teste JSON
        print("\n--- Teste de JSON ---")
        try:
            json_response = client.chat_json([
                {"role": "system", "content": "Responda apenas com JSON valido."},
                {"role": "user", "content": 'Classifique: {"tipo": "definicao ou procedimento", "confianca": 0.0-1.0} para: "ETP e o Estudo Tecnico Preliminar"'}
            ], max_tokens=100)
            print(f"JSON: {json_response}")
        except Exception as e:
            print(f"Erro: {e}")

    client.close()
