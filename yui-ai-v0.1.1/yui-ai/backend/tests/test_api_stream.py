"""Testes do endpoint de streaming (Server-Sent Events)."""
import json

from httpx import AsyncClient

from app.services.llm.base import LLMResponse, ToolCall
from tests.conftest import register_and_login
from tests.fakes import FakeLLM


async def _collect_events(client: AsyncClient, payload: dict, headers: dict) -> list[dict]:
    events = []
    async with client.stream(
        "POST", "/api/v1/chat/stream", json=payload, headers=headers
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


async def test_stream_emits_deltas_and_done(client: AsyncClient) -> None:
    headers = await register_and_login(client)
    events = await _collect_events(client, {"message": "olá Yui"}, headers)

    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert "".join(deltas) == "eco: olá Yui"

    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["model"] == "fake-model"
    assert done[0]["conversation_id"]


async def test_stream_emits_tool_events(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    headers = await register_and_login(client)
    fake_llm.script.extend(
        [
            LLMResponse(
                content="",
                model="fake-model",
                tool_calls=(
                    ToolCall(id="c1", name="create_task", arguments={"title": "Ler"}),
                ),
            ),
            LLMResponse(content="Tarefa criada!", model="fake-model"),
        ]
    )
    events = await _collect_events(client, {"message": "crie uma tarefa"}, headers)

    types = [e["type"] for e in events]
    assert "tool" in types and "done" in types
    tool_event = next(e for e in events if e["type"] == "tool")
    assert tool_event["tools"] == ["create_task"]
    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "Tarefa criada!" in deltas


async def test_stream_persists_turn(client: AsyncClient, db_session) -> None:
    from sqlalchemy import select

    from app.models.conversation import Message

    headers = await register_and_login(client)
    events = await _collect_events(client, {"message": "persisto?"}, headers)
    done = next(e for e in events if e["type"] == "done")

    result = await db_session.execute(select(Message).order_by(Message.sequence))
    messages = list(result.scalars())
    assert [(m.sequence, m.role) for m in messages] == [(1, "user"), (2, "assistant")]
    assert str(messages[0].conversation_id) == done["conversation_id"]


async def test_stream_requires_auth(client: AsyncClient) -> None:
    response = await client.post("/api/v1/chat/stream", json={"message": "oi"})
    assert response.status_code == 401


async def test_stream_reports_unknown_conversation_as_error_event(
    client: AsyncClient,
) -> None:
    import uuid

    headers = await register_and_login(client)
    events = await _collect_events(
        client,
        {"message": "oi", "conversation_id": str(uuid.uuid4())},
        headers,
    )
    assert events == [{"type": "error", "detail": "Conversa não encontrada."}]
