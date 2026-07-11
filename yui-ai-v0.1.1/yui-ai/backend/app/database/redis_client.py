"""Cliente Redis compartilhado (memória de curto prazo).

A criação é protegida por lock para evitar que requisições concorrentes
durante o cold start criem múltiplos clientes.
"""
import asyncio

from redis.asyncio import Redis, from_url

from app.core.config import get_settings

_redis: Redis | None = None
_lock = asyncio.Lock()


async def get_redis() -> Redis:
    global _redis  # noqa: PLW0603 — singleton de processo, protegido por lock
    if _redis is None:
        async with _lock:
            if _redis is None:
                _redis = from_url(get_settings().redis_url, decode_responses=True)
    return _redis


async def close_redis() -> None:
    global _redis  # noqa: PLW0603
    if _redis is not None:
        await _redis.aclose()
        _redis = None
