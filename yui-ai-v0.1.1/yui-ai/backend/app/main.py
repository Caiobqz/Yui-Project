"""Ponto de entrada da API da Yui."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, health, memories
from app.core.config import get_settings
from app.core.personality import get_personality
from app.database.redis_client import close_redis
from app.database.session import engine
from app.models import Base
from app.services.llm.base import LLMError

logger = logging.getLogger("yui")
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Valida a personalidade no boot (falha cedo se o YAML estiver inválido).
    get_personality()

    # Em desenvolvimento, cria as tabelas automaticamente.
    # Em produção, use exclusivamente as migrations do Alembic.
    if settings.is_development:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    yield

    await close_redis()
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(LLMError)
async def llm_error_handler(_request: Request, exc: LLMError) -> JSONResponse:
    """Converte falhas do provedor de IA em 502 com mensagem clara.

    Cobre tanto erros durante a geração quanto erros na resolução de
    dependências (ex.: API key não configurada), que de outra forma
    virariam um 500 genérico.
    """
    logger.error("Falha no provedor de IA: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


app.include_router(health.router)
app.include_router(chat.router, prefix=settings.api_v1_prefix)
app.include_router(memories.router, prefix=settings.api_v1_prefix)
