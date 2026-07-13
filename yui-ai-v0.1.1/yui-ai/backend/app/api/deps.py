"""Dependências injetáveis da API."""
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.memory_agent import MemoryAgent
from app.agents.yui_core import YuiCore
from app.core.security import decode_access_token
from app.database.redis_client import get_redis
from app.database.session import async_session_factory, get_db_session
from app.memory.short_term import ShortTermMemory
from app.models.user import User
from app.services.embeddings.factory import get_embedding_provider
from app.services.llm.factory import get_llm_provider, get_utility_llm_provider
from app.services.rate_limiter import RateLimiter
from app.tools.registry import build_default_registry

DbSession = Annotated[AsyncSession, Depends(get_db_session)]

# auto_error=False para controlarmos a mensagem e o status do 401.
_bearer_scheme = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Não autenticado.",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    session: DbSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)
    ] = None,
) -> User:
    """Identifica o usuário exclusivamente pelo token JWT.

    O cliente nunca informa user_id: toda autorização deriva daqui.
    """
    if credentials is None:
        raise _UNAUTHORIZED
    try:
        user_id = decode_access_token(credentials.credentials)
    except jwt.InvalidTokenError:
        raise _UNAUTHORIZED from None

    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise _UNAUTHORIZED
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_yui_core() -> YuiCore:
    """Monta o orquestrador do chat.

    Recebe a *factory* de sessões (não uma sessão da requisição): o YuiCore
    abre conexões curtas antes e depois da chamada ao LLM, sem segurar
    conexão do pool durante a geração.
    """
    redis = await get_redis()
    return YuiCore(
        session_factory=async_session_factory,
        short_term=ShortTermMemory(redis),
        llm=get_llm_provider(),
        rate_limiter=RateLimiter(redis),
        embeddings=get_embedding_provider(),
        registry=build_default_registry(),
        utility_llm=get_utility_llm_provider(),
    )


Yui = Annotated[YuiCore, Depends(get_yui_core)]


def get_memory_agent() -> MemoryAgent:
    """MemoryAgent para a rota de memórias (mesma triagem/dedupe da conversa)."""
    return MemoryAgent(
        llm=get_llm_provider(),
        embeddings=get_embedding_provider(),
        session_factory=async_session_factory,
    )


MemoryAgentDep = Annotated[MemoryAgent, Depends(get_memory_agent)]
