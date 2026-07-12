"""Infraestrutura de ferramentas (tool calling).

Uma ferramenta é a especificação exposta ao modelo (`ToolSpec`) + um handler
assíncrono que recebe o contexto do turno e os argumentos validados. Handlers
sempre RETORNAM texto (inclusive em erro) — o resultado volta ao modelo, que
decide como comunicar ao usuário. Handlers nunca derrubam o turno.
"""
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.services.llm.base import LLMProvider, ToolSpec


@dataclass(frozen=True)
class ToolContext:
    """Contexto passado a cada execução de ferramenta.

    Carrega a *factory* de sessões (handlers abrem conexões curtas, mantendo
    a postura de não segurar conexão durante chamadas de IA) e o LLM, para
    ferramentas que precisam de raciocínio (ex.: planner).
    """

    user_id: uuid.UUID
    conversation_id: uuid.UUID
    session_factory: async_sessionmaker[AsyncSession]
    llm: LLMProvider


ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: ToolHandler
