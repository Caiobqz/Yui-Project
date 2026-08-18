"""Testes do histórico: orçamento, sanitização e rehidratação do cache."""
import logging
import uuid

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from app.agents.yui_core import YuiCore
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
from app.services.rate_limiter import RateLimiter
from tests.fakes import FakeEmbeddings, FakeLLM, FakeRedis


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


class _UnavailableRedis(FakeRedis):
    """Redis indisponível em todos os comandos usados pelo turno."""

    @staticmethod
    def _raise() -> None:
        raise RedisConnectionError("redis offline")

    def pipeline(self):
        self._raise()

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        self._raise()

    async def incr(self, key: str) -> int:
        self._raise()

    async def incrby(self, key: str, amount: int) -> int:
        self._raise()

    async def get(self, key: str) -> int | None:
        self._raise()


class _AppendUnavailableShortTerm:
    async def get_history(self, conversation_id: str) -> list[ChatMessage]:
        return []

    async def append_many(
        self, conversation_id: str, messages: list[ChatMessage]
    ) -> None:
        raise RedisConnectionError("redis offline: conteúdo-super-secreto")


async def _stored_conversation(db_session, secret: str = "olá") -> Conversation:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()
    conversation = Conversation(user_id=user.id)
    db_session.add(conversation)
    await db_session.flush()
    db_session.add(
        Message(
            conversation_id=conversation.id,
            sequence=1,
            role="user",
            content=secret,
        )
    )
    await db_session.commit()
    return conversation


async def test_load_uses_postgres_when_redis_get_fails(
    db_session, caplog: pytest.LogCaptureFixture
) -> None:
    conversation = await _stored_conversation(db_session)
    service = HistoryService(
        ShortTermMemory(_UnavailableRedis())  # type: ignore[arg-type]
    )

    with caplog.at_level(logging.WARNING, logger="yui.history"):
        history = await service.load(db_session, conversation.id)

    assert [(message.role, message.content) for message in history] == [("user", "olá")]
    assert "usando PostgreSQL" in caplog.text


async def test_load_returns_postgres_history_when_cache_repopulation_fails(
    db_session, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "segredo-que-nao-pode-ir-ao-log"
    conversation = await _stored_conversation(db_session, secret)
    service = HistoryService(_AppendUnavailableShortTerm())  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING, logger="yui.history"):
        history = await service.load(db_session, conversation.id)

    assert [message.content for message in history] == [secret]
    assert "não pôde ser reidratado" in caplog.text
    assert secret not in caplog.text
    assert "conteúdo-super-secreto" not in caplog.text


async def test_load_does_not_mask_postgres_failure() -> None:
    class _BrokenSession:
        async def execute(self, statement):
            raise OperationalError("SELECT", {}, RuntimeError("postgres offline"))

    service = HistoryService(ShortTermMemory(FakeRedis()))  # type: ignore[arg-type]

    with pytest.raises(OperationalError, match="postgres offline"):
        await service.load(_BrokenSession(), uuid.uuid4())  # type: ignore[arg-type]


async def test_complete_yui_turn_degrades_to_postgres_when_redis_is_unavailable(
    session_factory, caplog: pytest.LogCaptureFixture
) -> None:
    redis = _UnavailableRedis()
    llm = FakeLLM()
    core = YuiCore(
        session_factory=session_factory,
        short_term=ShortTermMemory(redis),  # type: ignore[arg-type]
        llm=llm,
        rate_limiter=RateLimiter(redis),  # type: ignore[arg-type]
        embeddings=FakeEmbeddings(),
    )
    async with session_factory() as session:
        user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
        session.add(user)
        await session.commit()
        user_id = user.id

    with caplog.at_level(logging.WARNING):
        first = await core.process_message(user_id, "free", "primeiro turno")
        second = await core.process_message(
            user_id,
            "free",
            "segundo turno",
            conversation_id=first.conversation_id,
        )

    assert first.content == "eco: primeiro turno"
    assert second.content == "eco: segundo turno"
    _, second_messages, _ = llm.calls[-1]
    assert [(message.role, message.content) for message in second_messages] == [
        ("user", "primeiro turno"),
        ("assistant", "eco: primeiro turno"),
        ("user", "segundo turno"),
    ]
    async with session_factory() as session:
        stored = list(
            (
                await session.execute(
                    select(Message).order_by(Message.sequence)
                )
            ).scalars()
        )
    assert [message.content for message in stored] == [
        "primeiro turno",
        "eco: primeiro turno",
        "segundo turno",
        "eco: segundo turno",
    ]
    assert "conteúdo-super-secreto" not in caplog.text
