"""Verificação de saúde da aplicação e das dependências."""
from fastapi import APIRouter
from sqlalchemy import text

from app.database.redis_client import get_redis
from app.database.session import engine

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    status: dict[str, str] = {"app": "ok"}

    # Captura ampla intencional: o health check nunca deve derrubar a app,
    # apenas reportar a dependência como indisponível.
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        status["postgres"] = "ok"
    except Exception:  # noqa: BLE001
        status["postgres"] = "unavailable"

    try:
        redis = await get_redis()
        await redis.ping()
        status["redis"] = "ok"
    except Exception:  # noqa: BLE001
        status["redis"] = "unavailable"

    return status
