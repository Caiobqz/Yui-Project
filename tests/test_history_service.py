"""Testes do histórico: orçamento, sanitização e rehidratação do cache."""
import uuid

from app.memory.short_term import ShortTermMemory
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.services.history_service import (
    HistoryService,
    prepare_for_llm,
    sanitize_history,
    trim_history,
)
from app.services.llm.base import ChatMessage
from tests.fakes import FakeRedis


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)  # type: ignore[arg-type]


def test_trim_keeps_most_recent_within_budget() -> None:
    messages = [_msg("user", "a" * 100), _msg("assistant", "b" * 100), _msg("user", "c" * 100)]
    trimmed = trim_history(messages, max_chars=250)
    assert [m.content[0] for m in trimmed] == ["b", "c"]


def test_trim_never_drops_the_latest_message() -> None:
    messages = [_msg("user", "x" * 5000)]
    assert trim_history(messages, max_chars=10) == messages


def test_sanitize_drops_leading_assistant_messages() -> None:
    messages = [_msg("assistant", "órfã"), _msg("user", "oi"), _msg("assistant", "olá")]
    sanitized = sanitize_history(messages)
    assert sanitized[0].role == "user"
    assert len(sanitized) == 2


def test_prepare_for_llm_combines_trim_and_sanitize() -> None:
    # O corte por orçamento deixa uma 'assistant' na frente; a sanitização remove.
    messages = [
        _msg("user", "a" * 300),
        _msg("assistant", "b" * 100),
        _msg("user", "c" * 100),
    ]
    prepared = prepare_for_llm(messages, max_chars=250)
    assert prepared[0].role == "user"


async def test_load_rehydrates_from_postgres_and_repopulates_cache(
    db_session,
) -> None:
    user = User(email="x@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()
    conversation = Conversation(user_id=user.id)
    db_session.add(conversation)
    await db_session.flush()
    db_session.add_all(
        [
            Message(conversation_id=conversation.id, sequence=1, role="user", content="oi"),
            Message(conversation_id=conversation.id, sequence=2, role="assistant", content="olá!"),
        ]
    )
    await db_session.commit()

    redis = FakeRedis()
    short_term = ShortTermMemory(redis)  # type: ignore[arg-type]
    service = HistoryService(short_term)

    # Cache frio → carrega do banco na ordem correta.
    history = await service.load(db_session, conversation.id)
    assert [(m.role, m.content) for m in history] == [("user", "oi"), ("assistant", "olá!")]

    # Cache foi reposto: segunda leitura vem do Redis.
    assert redis.lists  # repovoado
    cached = await service.load(db_session, conversation.id)
    assert cached == history


async def test_load_unknown_conversation_returns_empty(db_session) -> None:
    service = HistoryService(ShortTermMemory(FakeRedis()))  # type: ignore[arg-type]
    assert await service.load(db_session, uuid.uuid4()) == []
