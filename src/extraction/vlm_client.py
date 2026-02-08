"""
VLM Client - Cliente assíncrono para Qwen3-VL via vLLM (multimodal).

Cliente separado do VLLMClient existente (src/llm/vllm_client.py) porque:
- O VLLMClient assume content como string e faz .endswith("/no_think")
- O formato multimodal requer content como lista de dicts:
  [{"type": "image_url", ...}, {"type": "text", ...}]

Este cliente envia imagens de páginas PDF + prompt de extração ao Qwen3-VL
e retorna JSON estruturado com dispositivos legais identificados.
"""

import json
import logging
import re
import time
from typing import Optional

import httpx

from .vlm_prompts import SYSTEM_PROMPT, PAGE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def _strip_thinking_block(text: str) -> str:
    """Remove bloco <think>...</think> da resposta do Qwen 3.

    Também remove blocos incompletos (sem </think>) para evitar vazamento.
    Reutiliza a mesma lógica de src/llm/vllm_client.py.
    """
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_json(text: str) -> dict:
    """Extrai JSON da resposta do VLM, lidando com markdown code blocks."""
    text = text.strip()

    # Remove markdown code blocks se presentes
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Tenta encontrar JSON na resposta
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Não foi possível parsear JSON da resposta VLM: {text[:200]}")


class VLMClient:
    """Cliente assíncrono para Qwen3-VL via vLLM (multimodal)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8002/v1",
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Args:
            base_url: URL base do vLLM (ex: http://localhost:8002/v1)
            model: Nome do modelo VLM
            timeout: Timeout por request em segundos
            max_retries: Número máximo de tentativas por página
            retry_delay: Delay base entre retries (multiplica por tentativa)
        """
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._timeout = timeout
        self._client = self._make_client()
        logger.info(f"VLMClient inicializado: {base_url} (model={model})")

    def _make_client(self) -> httpx.AsyncClient:
        """Cria um httpx.AsyncClient novo (desvinculado de event loops anteriores)."""
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            headers={"Content-Type": "application/json"},
        )

    def reset_client(self) -> None:
        """Recria o httpx.AsyncClient para uso em um novo event loop."""
        self._client = self._make_client()

    async def extract_page(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> dict:
        """
        Envia imagem de uma página + prompt ao Qwen3-VL e retorna JSON.

        Args:
            image_base64: Imagem da página em base64 (PNG)
            prompt: Prompt de extração (default: PAGE_PROMPT_TEMPLATE)
            temperature: Temperatura de geração (0.0 para determinístico)
            max_tokens: Máximo de tokens na resposta

        Returns:
            Dict com campo "devices" contendo dispositivos extraídos

        Raises:
            RuntimeError: Se todas as tentativas falharem
        """
        if prompt is None:
            prompt = PAGE_PROMPT_TEMPLATE

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = await self._client.post(
                    "/chat/completions",
                    json=payload,
                )
                response.raise_for_status()

                elapsed = time.time() - start_time
                data = response.json()

                # Extrai conteúdo da resposta
                content = data["choices"][0]["message"]["content"]

                # Remove bloco <think> se presente
                content = _strip_thinking_block(content)

                # Log métricas
                usage = data.get("usage", {})
                logger.debug(
                    f"VLM response: {elapsed:.2f}s, "
                    f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
                    f"completion_tokens={usage.get('completion_tokens', '?')}"
                )

                # Parseia JSON
                result = _extract_json(content)

                # Garante que tem o campo "devices"
                if "devices" not in result:
                    result = {"devices": []}

                return result

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(
                    f"VLM HTTP error (attempt {attempt + 1}/{self.max_retries}): "
                    f"{e.response.status_code} - {e.response.text[:200]}"
                )
                if attempt < self.max_retries - 1:
                    import asyncio
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"VLM timeout (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    import asyncio
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            except ValueError as e:
                # JSON parse error — retry com temperatura mais baixa
                last_error = e
                logger.warning(
                    f"VLM JSON parse error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    # Reduz temperatura para tentar resposta mais determinística
                    payload["temperature"] = 0.0
                    import asyncio
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                last_error = e
                logger.error(f"VLM unexpected error: {e}")
                raise

        raise RuntimeError(
            f"VLM falhou após {self.max_retries} tentativas: {last_error}"
        )

    async def health_check(self) -> bool:
        """Verifica se o servidor VLM está respondendo."""
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return len(models) > 0
        except Exception:
            return False

    async def close(self):
        """Fecha o cliente HTTP."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
