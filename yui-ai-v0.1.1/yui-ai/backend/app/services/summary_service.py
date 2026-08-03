"""Compactação de contexto: resumo incremental de conversas longas.

Mensagens fora da janela de curto prazo deixariam de chegar ao modelo; em
vez de perdê-las, um resumo incremental (resumo anterior + mensagens que
saíram da janela) é mantido em `conversations.summary` e injetado no system
prompt. Roda em background (pós-turno), nunca no caminho da resposta.
"""
import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.models.conversation import Conversation, Message
from app.services.llm.base import ChatMessage, LLMProvider
from app.services.usage_service import build_usage_record

logger = logging.getLogger("yui.summarizer")

_SUMMARY_SYSTEM_PROMPT = (
    "Você é o componente de compactação de contexto da Yui. Resuma a conversa "
    "preservando fatos, decisões, preferências, nomes e pendências relevantes. "
    "Escreva em português, em terceira pessoa, em no máximo 200 palavras. "
    "Responda APENAS com o resumo."
)


class ConversationSummarizer:
    def __init__(
        self,
        llm: LLMProvider,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._llm = llm
        self._session_factory = session_factory

    async def maybe_summarize(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> bool:
        """Atualiza o resumo se mensagens novas saíram da janela de contexto.

        Retorna True quando um novo resumo foi gerado.
        """
        settings = get_settings()

        # Fase de leitura (conexão curta).
        async with self._session_factory() as session:
            conversation = await session.get(Conversation, conversation_id)
            if conversation is None or conversation.user_id != user_id:
                return False
            max_sequence = (
                await session.scalar(
                    select(func.max(Message.sequence)).where(
                        Message.conversation_id == conversation_id
                    )
                )
            ) or 0
            cutoff = max_sequence - settings.short_term_max_messages
            if cutoff <= conversation.summary_up_to_sequence:
                return False

            result = await session.execute(
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.sequence > conversation.summary_up_to_sequence,
                    Message.sequence <= cutoff,
                )
                .order_by(Message.sequence)
            )
            overflow = list(result.scalars())
            previous_summary = conversation.summary

        if not overflow:
            return False

        # Chamada ao modelo (sem conexão de banco aberta).
        transcript = "\n".join(
            f"{'Usuário' if m.role == 'user' else 'Yui'}: {m.content}"
            for m in overflow
        )
        parts = []
        if previous_summary:
            parts.append(f"Resumo anterior:\n{previous_summary}")
        parts.append(f"Novas mensagens:\n{transcript}")
        response = await self._llm.generate(
            _SUMMARY_SYSTEM_PROMPT,
            [ChatMessage(role="user", content="\n\n".join(parts))],
        )
        summary = response.content.strip()
        if not summary:
            return False

        # Fase de escrita (conexão curta).
        async with self._session_factory() as session:
            conversation = await session.get(Conversation, conversation_id)
            if conversation is None:
                return False
            conversation.summary = summary
            conversation.summary_up_to_sequence = cutoff
            session.add(build_usage_record(user_id, conversation_id, response))
            await session.commit()

        logger.info(
            "Resumo atualizado (conversa %s, até sequence %d).",
            conversation_id,
            cutoff,
        )
        return True
