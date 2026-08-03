"""Memória de curto prazo: histórico recente da conversa, mantido no Redis.

Cada conversa é uma lista Redis limitada a N mensagens, com TTL.
É a memória de trabalho enviada ao modelo em cada turno.
"""
import json

from redis.asyncio import Redis

from app.core.config import get_settings
from app.services.llm.base import ChatMessage


class ShortTermMemory:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        settings = get_settings()
        self._max_messages = settings.short_term_max_messages
        self._ttl = settings.short_term_ttl_seconds

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"yui:stm:{conversation_id}"

    async def append_many(
        self, conversation_id: str, messages: list[ChatMessage]
    ) -> None:
        """Adiciona mensagens em um único pipeline (uma ida ao Redis).

        Gravar o par usuário/assistente junto evita histórico inconsistente
        caso uma segunda chamada falhasse no meio do turno.
        """
        if not messages:
            return
        key = self._key(conversation_id)
        payloads = [
            json.dumps({"role": m.role, "content": m.content}) for m in messages
        ]
        pipe = self._redis.pipeline()
        pipe.rpush(key, *payloads)
        pipe.ltrim(key, -self._max_messages, -1)  # mantém apenas as N mais recentes
        pipe.expire(key, self._ttl)
        await pipe.execute()

    async def append(self, conversation_id: str, message: ChatMessage) -> None:
        await self.append_many(conversation_id, [message])

    async def get_history(self, conversation_id: str) -> list[ChatMessage]:
        raw = await self._redis.lrange(self._key(conversation_id), 0, -1)
        history: list[ChatMessage] = []
        for item in raw:
            data = json.loads(item)
            history.append(ChatMessage(role=data["role"], content=data["content"]))
        return history

    async def clear(self, conversation_id: str) -> None:
        await self._redis.delete(self._key(conversation_id))
