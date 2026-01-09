"""
Middleware de autenticacao para GPU Server.
Protege endpoints com API Key.

IMPORTANTE: Em BaseHTTPMiddleware, HTTPException nao funciona corretamente.
Usamos JSONResponse diretamente para retornar erros de autenticacao.
"""

import os
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# API Keys validas (da variavel de ambiente ou default)
VALID_API_KEYS = set(
    os.getenv("GPU_API_KEYS", "vg_gpu_internal_2025").split(",")
)

API_KEY_HEADER = "X-GPU-API-Key"

# Endpoints publicos (nao precisam de auth)
PUBLIC_ENDPOINTS = {
    "/",
    "/health",
    "/healthz",
    "/readyz",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware que valida API Key em endpoints protegidos."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Endpoints publicos nao precisam de auth
        if path in PUBLIC_ENDPOINTS:
            return await call_next(request)

        # Verifica API Key
        api_key = request.headers.get(API_KEY_HEADER)

        if not api_key:
            logger.warning(
                f"Request sem API key: {path} from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=401,
                content={"detail": f"Missing {API_KEY_HEADER} header"},
            )

        if api_key not in VALID_API_KEYS:
            logger.warning(
                f"API key invalida: {api_key[:12]}... from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"},
            )

        # Key valida - processa request
        return await call_next(request)
