"""Histórico de conversa: Redis como cache, PostgreSQL como fonte de verdade.

Fluxo de leitura:
    1. Tenta o Redis (rápido, TTL curto).
    2. Cache frio (TTL expirado ou restart do Redis) → rehidrata a partir da
       tabela `messages` e repõe o cache.

Também prepara o histórico para envio ao modelo: corte por orçamento de
caracteres (~4 chars/token) e sanitização da alternância de papéis.
"""
import logging
import uuid
from typing import Literal, cast

from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.memory.short_term import ShortTermMemory
from app.models.conversation import Message
from app.services.llm.base import ChatMessage

logger = logging.getLogger("yui.history")


def trim_history(messages: list[ChatMessage], max_chars: int) -> list[ChatMessage]:
    """Mantém as mensagens mais recentes dentro do orçamento de caracteres.

    A mensagem mais recente entra sempre, mesmo que sozinha estoure o
    orçamento — cortá-la deixaria o modelo sem a pergunta do usuário.
    """
    total = 0
    kept: list[ChatMessage] = []
    for message in reversed(messages):
        if kept and total + len(message.content) > max_chars:
            break
        kept.append(message)
        total += len(message.content)
    kept.reverse()
    return kept

def sanitize_history(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Garante que o histórico comece com mensagem do usuário.

    Cortes por orçamento ou por LTRIM podem deixar uma mensagem 'assistant'
    órfã no início — os provedores rejeitam históricos assim (HTTP 400).
    """
    start = 0
    while start < len(messages) and messages[start].role != "user":
        start += 1
    return messages[start:]


def prepare_for_llm(messages: list[ChatMessage], max_chars: int) -> list[ChatMessage]:
    return sanitize_history(trim_history(messages, max_chars))


class HistoryService:
    def __init__(self, short_term: ShortTermMemory) -> None:
        self._short_term = short_term

    async def load(
        self, session: AsyncSession, conversation_id: uuid.UUID
    ) -> list[ChatMessage]:
        """Retorna o histórico recente; rehidrata o cache quando necessário."""
        key = str(conversation_id)
        try:
            cached = await self._short_term.get_history(key)
        except RedisError as exc:
            logger.warning(
                "Falha ao ler cache de histórico; usando PostgreSQL "
                "(redis_error=%s).",
                type(exc).__name__,
            )
            cached = []
        if cached:
            return cached

        settings = get_settings()
        result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.sequence.desc())
            .limit(settings.short_term_max_messages)
        )
        rows = list(result.scalars())
        if not rows:
            return []

        rows.reverse()  # mais antiga primeiro
        messages = [
            ChatMessage(
                role=cast(Literal["user", "assistant"], row.role),
                content=row.content,
            )
            for row in rows
            if row.role in ("user", "assistant")
        ]
        logger.info(
            "Cache de histórico frio; rehidratando %d mensagens (conversa %s).",
            len(messages),
            conversation_id,
        )
        try:
            await self._short_term.append_many(key, messages)
        except RedisError as exc:
            logger.warning(
                "Histórico recuperado do PostgreSQL, mas o cache não pôde "
                "ser reidratado (redis_error=%s).",
                type(exc).__name__,
            )
        return messages
