"""
Middleware de autenticacao para GPU Server.
Protege endpoints com API Key + IP Allowlist.

IMPORTANTE: Em BaseHTTPMiddleware, HTTPException nao funciona corretamente.
Usamos JSONResponse diretamente para retornar erros de autenticacao.

Seguranca em camadas:
1. IP Allowlist - Apenas IPs autorizados podem acessar
2. API Key - Autenticacao via header X-GPU-API-Key
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

# IPs permitidos (da variavel de ambiente)
# Formato: "77.37.43.160,10.0.0.1" ou "*" para permitir todos
# Default: "*" (sem restricao de IP, apenas API key)
_allowed_ips_raw = os.getenv("ALLOWED_IPS", "*")
ALLOWED_IPS = None if _allowed_ips_raw == "*" else set(_allowed_ips_raw.split(","))

# Se IP allowlist esta ativo, loga os IPs permitidos
if ALLOWED_IPS:
    logger.info(f"IP Allowlist ATIVO: {ALLOWED_IPS}")
else:
    logger.info("IP Allowlist DESATIVADO (ALLOWED_IPS=*)")

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


def get_client_ip(request: Request) -> str:
    """
    Extrai o IP real do cliente, considerando proxies (Cloudflare, nginx).

    Ordem de prioridade:
    1. CF-Connecting-IP (Cloudflare)
    2. X-Real-IP (nginx)
    3. X-Forwarded-For (primeiro IP da lista)
    4. request.client.host (conexao direta)
    """
    # Cloudflare
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    # Nginx
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # X-Forwarded-For (pode ter multiplos IPs: "client, proxy1, proxy2")
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    # Conexao direta
    if request.client:
        return request.client.host

    return "unknown"


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware que valida IP Allowlist + API Key em endpoints protegidos.

    Ordem de verificacao:
    1. Endpoints publicos passam direto
    2. Verifica IP allowlist (se configurado)
    3. Verifica API Key
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        client_ip = get_client_ip(request)

        # Endpoints publicos nao precisam de auth
        if path in PUBLIC_ENDPOINTS:
            return await call_next(request)

        # 1. Verifica IP Allowlist (se configurado)
        if ALLOWED_IPS is not None and client_ip not in ALLOWED_IPS:
            logger.warning(
                f"IP bloqueado: {client_ip} tentou acessar {path}"
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "IP not allowed"},
            )

        # 2. Verifica API Key
        api_key = request.headers.get(API_KEY_HEADER)

        if not api_key:
            logger.warning(
                f"Request sem API key: {path} from {client_ip}"
            )
            return JSONResponse(
                status_code=401,
                content={"detail": f"Missing {API_KEY_HEADER} header"},
            )

        if api_key not in VALID_API_KEYS:
            logger.warning(
                f"API key invalida: {api_key[:12]}... from {client_ip}"
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"},
            )

        # IP + Key validos - processa request
        logger.debug(f"Request autorizado: {path} from {client_ip}")
        return await call_next(request)
