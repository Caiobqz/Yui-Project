"""Respostas finais vazias nunca atravessam persistência ou pós-turno."""
import uuid
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import select

from app.agents.yui_core import _EMPTY_RESPONSE_FALLBACK, YuiCore
from app.memory.short_term import ShortTermMemory
from app.models.conversation import Message
from app.models.user import User
from app.services.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    StreamChunk,
    ToolCall,
    ToolSpec,
)
from app.services.rate_limiter import RateLimiter
from app.tools.registry import build_default_registry
from tests.fakes import FakeEmbeddings, FakeLLM, FakeRedis


async def _create_user(session_factory) -> uuid.UUID:
    async with session_factory() as session:
        user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
        session.add(user)
        await session.commit()
        return user.id


def _build_core(session_factory, llm: LLMProvider) -> YuiCore:
    redis = FakeRedis()
    return YuiCore(
        session_factory=session_factory,
        short_term=ShortTermMemory(redis),  # type: ignore[arg-type]
        llm=llm,
        rate_limiter=RateLimiter(redis),  # type: ignore[arg-type]
        embeddings=FakeEmbeddings(),
        registry=build_default_registry(),
    )


@pytest.mark.parametrize("invalid_content", ["", "   ", "\n\n"])
async def test_empty_or_whitespace_final_retries_once_then_accepts_valid_text(
    session_factory, invalid_content: str
) -> None:
    llm = FakeLLM(
        script=[
            LLMResponse(content=invalid_content, model="fake"),
            LLMResponse(content="resposta válida", model="fake"),
        ]
    )
    core = _build_core(session_factory, llm)
    user_id = await _create_user(session_factory)

    reply = await core.process_message(user_id, "free", "olá")

    assert reply.content == "resposta válida"
    assert len(llm.calls) == 2


async def test_tool_call_without_text_is_allowed_before_valid_final(
    session_factory,
) -> None:
    llm = FakeLLM(
        script=[
            LLMResponse(
                content="",
                model="fake",
                tool_calls=(
                    ToolCall(id="call-1", name="list_tasks", arguments={}),
                ),
            ),
            LLMResponse(content="Você não tem tarefas.", model="fake"),
        ]
    )
    core = _build_core(session_factory, llm)
    user_id = await _create_user(session_factory)

    reply = await core.process_message(user_id, "free", "liste minhas tarefas")

    assert reply.content == "Você não tem tarefas."
    assert len(llm.calls) == 2
    assert any(message.role == "tool" for message in llm.calls[-1][1])


async def test_empty_retry_budget_is_global_across_tool_iterations(
    session_factory,
) -> None:
    llm = FakeLLM(
        script=[
            LLMResponse(content="", model="fake"),
            LLMResponse(
                content="",
                model="fake",
                tool_calls=(
                    ToolCall(id="call-1", name="list_tasks", arguments={}),
                ),
            ),
            LLMResponse(content="\n", model="fake"),
        ]
    )
    core = _build_core(session_factory, llm)
    user_id = await _create_user(session_factory)

    reply = await core.process_message(user_id, "free", "liste minhas tarefas")

    assert reply.content == _EMPTY_RESPONSE_FALLBACK
    assert len(llm.calls) == 3
    assert any(message.role == "tool" for message in llm.calls[-1][1])


async def test_repeated_empty_final_persists_fallback_and_sends_it_to_post_turn(
    session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = FakeLLM(
        script=[
            LLMResponse(content="", model="fake"),
            LLMResponse(content=" \n", model="fake"),
        ]
    )
    core = _build_core(session_factory, llm)
    user_id = await _create_user(session_factory)
    post_turn_calls: list[tuple[uuid.UUID, uuid.UUID, str, str]] = []
    monkeypatch.setattr(
        core,
        "_schedule_post_turn",
        lambda *args: post_turn_calls.append(args),
    )

    reply = await core.process_message(user_id, "free", "olá")

    assert reply.content == _EMPTY_RESPONSE_FALLBACK
    assert reply.content.strip()
    assert len(llm.calls) == 2
    assert post_turn_calls[0][-1] == _EMPTY_RESPONSE_FALLBACK
    async with session_factory() as session:
        assistant = (
            await session.execute(select(Message).where(Message.role == "assistant"))
        ).scalar_one()
    assert assistant.content == _EMPTY_RESPONSE_FALLBACK
    assert assistant.content.strip()


class _ScriptedStreamLLM(LLMProvider):
    def __init__(
        self,
        attempts: list[tuple[list[str], LLMResponse]],
    ) -> None:
        self.attempts = list(attempts)
        self.calls = 0

    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        raise AssertionError("stream_message deve usar generate_stream")

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        self.calls += 1
        deltas, response = self.attempts.pop(0)
        for delta in deltas:
            yield StreamChunk(delta=delta)
        yield StreamChunk(response=response)


async def test_stream_uses_valid_deltas_when_consolidated_content_is_empty(
    session_factory,
) -> None:
    llm = _ScriptedStreamLLM(
        [(["resposta ", "incremental"], LLMResponse(content="", model="fake"))]
    )
    core = _build_core(session_factory, llm)
    user_id = await _create_user(session_factory)

    events = [event async for event in core.stream_message(user_id, "free", "olá")]

    assert [event["text"] for event in events if event["type"] == "delta"] == [
        "resposta ",
        "incremental",
    ]
    assert llm.calls == 1
    async with session_factory() as session:
        assistant = (
            await session.execute(select(Message).where(Message.role == "assistant"))
        ).scalar_one()
    assert assistant.content == "resposta incremental"


async def test_stream_emits_whitespace_deltas_before_retrying_empty_final(
    session_factory,
) -> None:
    llm = _ScriptedStreamLLM(
        [
            (["  ", "\n"], LLMResponse(content="", model="fake")),
            (["resposta válida"], LLMResponse(content="", model="fake")),
        ]
    )
    core = _build_core(session_factory, llm)
    user_id = await _create_user(session_factory)

    events = [event async for event in core.stream_message(user_id, "free", "olá")]

    assert [event["text"] for event in events if event["type"] == "delta"] == [
        "  ",
        "\n",
        "resposta válida",
    ]
    assert llm.calls == 2
    async with session_factory() as session:
        assistant = (
            await session.execute(select(Message).where(Message.role == "assistant"))
        ).scalar_one()
    assert assistant.content == "resposta válida"


async def test_completely_empty_stream_retries_then_emits_and_persists_fallback(
    session_factory,
) -> None:
    llm = _ScriptedStreamLLM(
        [
            ([], LLMResponse(content="", model="fake")),
            ([], LLMResponse(content="\n", model="fake")),
        ]
    )
    core = _build_core(session_factory, llm)
    user_id = await _create_user(session_factory)

    events = [event async for event in core.stream_message(user_id, "free", "olá")]

    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert deltas == [_EMPTY_RESPONSE_FALLBACK]
    assert llm.calls == 2
    async with session_factory() as session:
        assistant = (
            await session.execute(select(Message).where(Message.role == "assistant"))
        ).scalar_one()
    assert assistant.content == _EMPTY_RESPONSE_FALLBACK
