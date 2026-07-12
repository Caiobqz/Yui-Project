"""Ponto de entrada da API da Yui."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import auth, chat, health, memories, notes, tasks
from app.core.config import get_settings
from app.core.exceptions import ConversationNotFoundError, RateLimitExceededError
from app.core.personality import get_personality
from app.database.redis_client import close_redis
from app.database.session import engine
from app.models import Base
from app.services.embeddings.factory import get_embedding_provider
from app.services.llm.base import LLMError
from app.services.llm.factory import get_llm_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("yui")
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fail-fast: configuração, personalidade e provedores (LLM e embeddings)
    # são validados no boot. Uma API key ausente falha AQUI, não na primeira
    # requisição.
    settings.validate_runtime()
    get_personality()
    get_llm_provider()
    get_embedding_provider()

    # Em desenvolvimento, cria as tabelas automaticamente.
    # Em produção, use exclusivamente as migrations do Alembic.
    if settings.is_development:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    logger.info("Yui %s iniciada (env=%s).", settings.version, settings.environment)
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


# --- Handlers de erro: detalhe no log, mensagem genérica ao cliente ---------


@app.exception_handler(LLMError)
async def llm_error_handler(_request: Request, exc: LLMError) -> JSONResponse:
    # O detalhe (que pode conter request ids e metadados do provedor) fica
    # apenas no log; o cliente recebe uma mensagem estável e genérica.
    logger.error("Falha no provedor de IA: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "Falha ao comunicar com o provedor de IA. Tente novamente."},
    )


@app.exception_handler(ConversationNotFoundError)
async def conversation_not_found_handler(
    _request: Request, exc: ConversationNotFoundError
) -> JSONResponse:
    logger.info("Conversa não encontrada: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "Conversa não encontrada."},
    )


@app.exception_handler(RateLimitExceededError)
async def rate_limit_handler(
    _request: Request, exc: RateLimitExceededError
) -> JSONResponse:
    headers = {}
    if exc.retry_after_seconds is not None:
        headers["Retry-After"] = str(exc.retry_after_seconds)
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)},
        headers=headers,
    )


app.include_router(health.router)
app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(chat.router, prefix=settings.api_v1_prefix)
app.include_router(memories.router, prefix=settings.api_v1_prefix)
app.include_router(tasks.router, prefix=settings.api_v1_prefix)
app.include_router(notes.router, prefix=settings.api_v1_prefix)
