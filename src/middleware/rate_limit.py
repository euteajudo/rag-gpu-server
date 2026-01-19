"""
Rate Limiting Middleware para GPU Server.

Implementa rate limiting in-memory para endpoints de inferência (/embed, /rerank).
Usa janela deslizante sem dependência de Redis (servidor GPU é isolado).

Uso:
    rate_limiter = InMemoryRateLimiter(max_requests=100, window_seconds=60)
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

Headers de resposta:
    - X-RateLimit-Limit: Limite de requisições por minuto
    - X-RateLimit-Remaining: Requisições restantes na janela atual
    - Retry-After: Segundos até reset (quando 429)
"""

import os
import time
import asyncio
import logging
from collections import defaultdict
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Rate limit padrão via env var ou fallback
DEFAULT_GPU_RATE_LIMIT = int(os.getenv("GPU_RATE_LIMIT", "100"))  # req/min


class InMemoryRateLimiter:
    """
    Rate limiter in-memory com janela deslizante.

    Não requer Redis - ideal para o GPU server que é um serviço isolado.
    Usa um dicionário com timestamps de requisições por chave.

    Attributes:
        max_requests: Máximo de requisições permitidas na janela
        window: Duração da janela em segundos
        requests: Dict de chave -> lista de timestamps
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_GPU_RATE_LIMIT,
        window_seconds: int = 60,
    ):
        """
        Inicializa o rate limiter.

        Args:
            max_requests: Máximo de requisições por janela
            window_seconds: Duração da janela em segundos
        """
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_counter = 0
        self._cleanup_interval = 100  # Limpa a cada N requisições

        logger.info(
            f"InMemoryRateLimiter initialized: max_requests={max_requests}/min, "
            f"window={window_seconds}s"
        )

    async def is_allowed(self, key: str) -> tuple[bool, int, int]:
        """
        Verifica se uma requisição é permitida para a chave dada.

        Args:
            key: Identificador único (API key prefix, IP, etc)

        Returns:
            Tupla (is_allowed, current_count, remaining)
        """
        async with self._lock:
            now = time.time()
            cutoff = now - self.window

            # Remove requisições antigas da janela
            self.requests[key] = [t for t in self.requests[key] if t > cutoff]

            current_count = len(self.requests[key])

            if current_count >= self.max_requests:
                return False, current_count, 0

            # Registra nova requisição
            self.requests[key].append(now)
            remaining = self.max_requests - current_count - 1

            # Cleanup periódico para evitar memory leak
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_interval:
                await self._cleanup_old_entries()
                self._cleanup_counter = 0

            return True, current_count + 1, remaining

    async def _cleanup_old_entries(self) -> None:
        """Remove entradas antigas de chaves que não têm requisições recentes."""
        now = time.time()
        cutoff = now - self.window * 2  # Margem de 2x a janela

        keys_to_remove = []
        for key, timestamps in self.requests.items():
            if not timestamps or max(timestamps) < cutoff:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.requests[key]

        if keys_to_remove:
            logger.debug(f"Rate limiter cleanup: removed {len(keys_to_remove)} stale keys")

    def get_stats(self) -> dict:
        """Retorna estatísticas do rate limiter."""
        now = time.time()
        cutoff = now - self.window

        active_keys = 0
        total_requests = 0

        for key, timestamps in self.requests.items():
            recent = [t for t in timestamps if t > cutoff]
            if recent:
                active_keys += 1
                total_requests += len(recent)

        return {
            "active_keys": active_keys,
            "total_requests_in_window": total_requests,
            "max_requests_per_key": self.max_requests,
            "window_seconds": self.window,
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware de rate limiting para GPU Server.

    Aplica rate limiting nos endpoints de inferência (/embed, /rerank).
    Usa API key ou IP como identificador.

    Attributes:
        rate_limiter: Instância do InMemoryRateLimiter
        protected_paths: Lista de prefixos de path protegidos
    """

    # Endpoints que devem ser rate-limited
    PROTECTED_PATHS = ["/embed", "/rerank"]

    def __init__(
        self,
        app,
        rate_limiter: Optional[InMemoryRateLimiter] = None,
        max_requests: int = DEFAULT_GPU_RATE_LIMIT,
    ):
        """
        Inicializa o middleware.

        Args:
            app: Aplicação FastAPI
            rate_limiter: Instância do rate limiter (ou cria um novo)
            max_requests: Máximo de requisições/min (se criar novo rate limiter)
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter or InMemoryRateLimiter(max_requests=max_requests)
        logger.info("RateLimitMiddleware initialized for GPU Server")

    def _should_rate_limit(self, path: str) -> bool:
        """Verifica se o path deve ser rate-limited."""
        return any(path.startswith(prefix) for prefix in self.PROTECTED_PATHS)

    def _get_client_key(self, request: Request) -> str:
        """
        Obtém chave de identificação do cliente.

        Prioridade:
        1. API Key (header X-GPU-API-Key)
        2. IP do cliente
        """
        # Tenta API key primeiro
        api_key = request.headers.get("x-gpu-api-key")
        if api_key:
            # Usa prefixo da key para anonimização
            return f"key:{api_key[:12]}" if len(api_key) >= 12 else f"key:{api_key}"

        # Fallback para IP
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        return f"ip:{client_ip}"

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Processa a requisição aplicando rate limiting.

        Args:
            request: Request HTTP
            call_next: Próximo handler na cadeia

        Returns:
            Response HTTP (200 se OK, 429 se rate limited)
        """
        # Só aplica rate limiting em endpoints protegidos
        if not self._should_rate_limit(request.url.path):
            return await call_next(request)

        # Obtém identificador do cliente
        client_key = self._get_client_key(request)

        # Verifica rate limit
        is_allowed, current_count, remaining = await self.rate_limiter.is_allowed(client_key)

        if not is_allowed:
            logger.warning(
                f"Rate limit exceeded: key={client_key}, "
                f"count={current_count}/{self.rate_limiter.max_requests}"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Limit: {self.rate_limiter.max_requests} requests/minute",
                    "limit": self.rate_limiter.max_requests,
                    "retry_after": self.rate_limiter.window,
                },
                headers={
                    "X-RateLimit-Limit": str(self.rate_limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(self.rate_limiter.window),
                },
            )

        # Processa requisição
        response = await call_next(request)

        # Adiciona headers de rate limit
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response
