"""Testes da compactação de contexto (resumo incremental de conversas)."""
import uuid

from sqlalchemy import select

from app.models.conversation import Conversation, Message
from app.models.user import User
from app.services.llm.base import LLMResponse
from app.services.summary_service import ConversationSummarizer
from tests.fakes import FakeLLM


async def _seed_conversation(
    session_factory, message_count: int
) -> tuple[uuid.UUID, uuid.UUID]:
    async with session_factory() as session:
        user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
        session.add(user)
        await session.flush()
        conversation = Conversation(user_id=user.id)
        session.add(conversation)
        await session.flush()
        for i in range(1, message_count + 1):
            session.add(
                Message(
                    conversation_id=conversation.id,
                    sequence=i,
                    role="user" if i % 2 else "assistant",
                    content=f"mensagem {i}",
                )
            )
        await session.commit()
        return user.id, conversation.id


async def test_short_conversation_is_not_summarized(session_factory) -> None:
    user_id, conversation_id = await _seed_conversation(session_factory, 6)
    summarizer = ConversationSummarizer(FakeLLM(), session_factory)
    assert await summarizer.maybe_summarize(user_id, conversation_id) is False


async def test_long_conversation_gets_incremental_summary(session_factory) -> None:
    # 26 mensagens, janela de 20 → 6 mensagens fora da janela.
    user_id, conversation_id = await _seed_conversation(session_factory, 26)
    llm = FakeLLM(
        script=[LLMResponse(content="Resumo: o usuário falou de X.", model="fake-model")]
    )
    summarizer = ConversationSummarizer(llm, session_factory)

    assert await summarizer.maybe_summarize(user_id, conversation_id) is True

    async with session_factory() as session:
        conversation = (
            await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
        ).scalar_one()
        assert conversation.summary == "Resumo: o usuário falou de X."
        assert conversation.summary_up_to_sequence == 6

    # As mensagens fora da janela foram enviadas ao modelo.
    _, messages, _ = llm.calls[0]
    assert "mensagem 6" in messages[0].content
    assert "mensagem 7" not in messages[0].content

    # Sem mensagens novas fora da janela, não resuma de novo.
    assert await summarizer.maybe_summarize(user_id, conversation_id) is False


async def test_summary_of_other_user_conversation_is_refused(session_factory) -> None:
    _, conversation_id = await _seed_conversation(session_factory, 26)
    summarizer = ConversationSummarizer(FakeLLM(), session_factory)
    assert await summarizer.maybe_summarize(uuid.uuid4(), conversation_id) is False
