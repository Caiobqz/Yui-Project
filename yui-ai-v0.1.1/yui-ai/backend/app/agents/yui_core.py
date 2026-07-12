"""Yui Core — agente central que orquestra um turno de conversa.

Fluxo de cada mensagem, em três fases:

    Fase 1 (conexão curta) — resolve a conversa, recupera memórias relevantes
        e carrega o histórico (Redis com rehidratação do PostgreSQL).
    Fase 2 (SEM conexão de banco) — monta o contexto e chama o provedor de IA.
        A chamada pode levar segundos; nenhuma conexão do pool fica presa.
    Fase 3 (conexão curta) — persiste o turno com sequence determinística,
        registra o uso (tokens/custo) e atualiza o cache.

Este módulo não conhece HTTP nem SDKs de IA: apenas coordena serviços.
"""
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.exceptions import ConversationNotFoundError
from app.memory.short_term import ShortTermMemory
from app.models.base import utcnow
from app.models.conversation import Conversation, Message
from app.services.context_service import build_system_prompt
from app.services.history_service import HistoryService, prepare_for_llm
from app.services.llm.base import ChatMessage, LLMProvider, LLMResponse
from app.services.memory_service import MemoryService
from app.services.rate_limiter import RateLimiter
from app.services.usage_service import build_usage_record


@dataclass(frozen=True)
class YuiReply:
    conversation_id: uuid.UUID
    content: str
    model: str
    memories_used: int


class YuiCore:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        short_term: ShortTermMemory,
        llm: LLMProvider,
        rate_limiter: RateLimiter,
    ) -> None:
        self._session_factory = session_factory
        self._short_term = short_term
        self._history = HistoryService(short_term)
        self._llm = llm
        self._rate_limiter = rate_limiter

    async def process_message(
        self,
        user_id: uuid.UUID,
        plan: str,
        text: str,
        conversation_id: uuid.UUID | None = None,
    ) -> YuiReply:
        settings = get_settings()

        # Proteção de uso/custo antes de qualquer trabalho.
        await self._rate_limiter.enforce(user_id, plan)

        # ---- Fase 1: leitura de contexto (conexão curta) -------------------
        async with self._session_factory() as session:
            conversation_id = await self._resolve_conversation(
                session, user_id, conversation_id
            )
            memories = await MemoryService(session).retrieve_relevant(user_id, text)
            history = await self._history.load(session, conversation_id)
            await session.commit()

        # ---- Fase 2: chamada ao modelo (sem conexão de banco) --------------
        system_prompt = build_system_prompt(memories)
        user_message = ChatMessage(role="user", content=text)
        llm_history = prepare_for_llm(
            [*history, user_message], settings.llm_max_history_chars
        )
        response = await self._llm.generate(system_prompt, llm_history)
        assistant_message = ChatMessage(role="assistant", content=response.content)

        # ---- Fase 3: persistência (conexão curta) --------------------------
        async with self._session_factory() as session:
            await self._persist_turn(
                session, user_id, conversation_id, text, response
            )
            await session.commit()

        # Cache de curto prazo (par gravado atomicamente) e contador de tokens.
        await self._short_term.append_many(
            str(conversation_id), [user_message, assistant_message]
        )
        await self._rate_limiter.register_tokens(
            user_id, (response.input_tokens or 0) + (response.output_tokens or 0)
        )

        return YuiReply(
            conversation_id=conversation_id,
            content=response.content,
            model=response.model,
            memories_used=len(memories),
        )

    async def _resolve_conversation(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID | None,
    ) -> uuid.UUID:
        """Valida a conversa informada ou cria uma nova.

        Conversa inexistente OU de outro usuário → 404 (a resposta não revela
        se o id existe). Antes deste fix, o fluxo criava silenciosamente uma
        conversa nova e fragmentava o histórico do cliente.
        """
        if conversation_id is not None:
            conversation = await session.get(Conversation, conversation_id)
            if conversation is None or conversation.user_id != user_id:
                raise ConversationNotFoundError(
                    f"Conversa {conversation_id} não encontrada."
                )
            return conversation.id

        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        await session.flush()
        return conversation.id

    async def _persist_turn(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_text: str,
        response: LLMResponse,
    ) -> None:
        # Lock da linha da conversa: serializa turnos concorrentes na mesma
        # conversa e garante sequence sem colisão (no SQLite dos testes o
        # FOR UPDATE é no-op — aceitável, o banco é single-writer).
        result = await session.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .with_for_update()
        )
        conversation = result.scalar_one()

        last_sequence = (
            await session.scalar(
                select(func.max(Message.sequence)).where(
                    Message.conversation_id == conversation_id
                )
            )
        ) or 0

        session.add_all(
            [
                Message(
                    conversation_id=conversation_id,
                    sequence=last_sequence + 1,
                    role="user",
                    content=user_text,
                ),
                Message(
                    conversation_id=conversation_id,
                    sequence=last_sequence + 2,
                    role="assistant",
                    content=response.content,
                ),
            ]
        )
        conversation.updated_at = utcnow()
        session.add(build_usage_record(user_id, conversation_id, response))
