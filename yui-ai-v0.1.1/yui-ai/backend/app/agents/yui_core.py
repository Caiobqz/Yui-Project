"""Yui Core — agente central que orquestra um turno de conversa.

Fluxo de cada mensagem:
    1. Garante que a conversa existe (PostgreSQL).
    2. Recupera o histórico recente (Redis).
    3. Recupera memórias de longo prazo relevantes (PostgreSQL).
    4. Monta o contexto (personalidade + memórias).
    5. Chama o provedor de IA através da camada abstrata.
    6. Persiste as mensagens (histórico completo e memória de curto prazo).

Este módulo não conhece HTTP nem SDKs de IA: apenas coordena serviços.
"""
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.personality import get_personality
from app.memory.short_term import ShortTermMemory
from app.models.conversation import Conversation, Message
from app.services.context_service import build_system_prompt
from app.services.llm.base import ChatMessage, LLMProvider
from app.services.memory_service import MemoryService


@dataclass(frozen=True)
class YuiReply:
    conversation_id: uuid.UUID
    content: str
    model: str
    memories_used: int


class YuiCore:
    def __init__(
        self,
        session: AsyncSession,
        short_term: ShortTermMemory,
        llm: LLMProvider,
    ) -> None:
        self._session = session
        self._short_term = short_term
        self._memory = MemoryService(session)
        self._llm = llm

    async def _get_or_create_conversation(
        self, user_id: str, conversation_id: uuid.UUID | None
    ) -> Conversation:
        if conversation_id is not None:
            result = await self._session.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            conversation = result.scalar_one_or_none()
            if conversation is not None:
                return conversation

        conversation = Conversation(user_id=user_id)
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def process_message(
        self,
        user_id: str,
        text: str,
        conversation_id: uuid.UUID | None = None,
    ) -> YuiReply:
        conversation = await self._get_or_create_conversation(user_id, conversation_id)
        conv_key = str(conversation.id)

        # Contexto: histórico recente + memórias relevantes + personalidade
        history = await self._short_term.get_history(conv_key)
        memories = await self._memory.retrieve_relevant(user_id, text)
        system_prompt = build_system_prompt(get_personality(), memories)

        user_message = ChatMessage(role="user", content=text)
        response = await self._llm.generate(system_prompt, [*history, user_message])
        assistant_message = ChatMessage(role="assistant", content=response.content)

        # Persistência: histórico completo (PostgreSQL)
        self._session.add_all(
            [
                Message(conversation_id=conversation.id, role="user", content=text),
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=response.content,
                ),
            ]
        )

        # Memória de curto prazo (Redis) — par gravado atomicamente
        await self._short_term.append_many(
            conv_key, [user_message, assistant_message]
        )

        return YuiReply(
            conversation_id=conversation.id,
            content=response.content,
            model=response.model,
            memories_used=len(memories),
        )
