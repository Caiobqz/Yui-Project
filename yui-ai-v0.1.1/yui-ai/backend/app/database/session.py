"""Conexão assíncrona com o PostgreSQL e injeção de sessão no FastAPI."""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    echo=_settings.database_echo,
    pool_pre_ping=True,
    # Pool explícito: o fluxo do chat NÃO segura conexão durante a chamada ao
    # LLM (ver YuiCore), então o pool atende requisições curtas de leitura e
    # escrita — dimensionado por config, com timeout para falhar rápido.
    pool_size=_settings.db_pool_size,
    max_overflow=_settings.db_max_overflow,
    pool_timeout=_settings.db_pool_timeout_seconds,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependência do FastAPI: uma sessão por requisição, com rollback em erro.

    Uso: rotas de CRUD rápido (auth, memórias). O fluxo de chat usa
    `async_session_factory` diretamente para não segurar conexão durante a
    chamada ao provedor de IA.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
