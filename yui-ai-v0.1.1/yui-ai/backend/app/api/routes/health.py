"""Verificação de saúde: liveness e readiness com semânticas distintas.

- /health        → 200 sempre que o processo responde (liveness).
- /health/ready  → 503 quando uma dependência crítica está indisponível
                   (readiness — orquestradores tiram a instância do balanceador).
"""
from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.core.config import get_settings
from app.database.redis_client import get_redis
from app.database.session import engine

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "version": settings.version}


@router.get("/health/ready")
async def readiness(response: Response) -> dict[str, str]:
    components: dict[str, str] = {}

    # Captura ampla intencional: o health check reporta a dependência como
    # indisponível em vez de derrubar a rota.
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        components["postgres"] = "ok"
    except Exception:  # noqa: BLE001
        components["postgres"] = "unavailable"

    try:
        redis = await get_redis()
        await redis.ping()
        components["redis"] = "ok"
    except Exception:  # noqa: BLE001
        components["redis"] = "unavailable"

    if any(v != "ok" for v in components.values()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        components["status"] = "degraded"
    else:
        components["status"] = "ok"
    return components
