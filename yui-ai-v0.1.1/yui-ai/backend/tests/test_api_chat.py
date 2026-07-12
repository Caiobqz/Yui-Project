"""Testes do fluxo de chat: persistência, sequência, uso e cache."""
import uuid

from httpx import AsyncClient
from sqlalchemy import select

from app.models.conversation import Message
from app.models.usage import UsageRecord
from tests.conftest import register_and_login
from tests.fakes import FakeRedis


async def test_chat_creates_conversation_and_replies(client: AsyncClient) -> None:
    headers = await register_and_login(client)
    resp = await client.post("/api/v1/chat", json={"message": "olá Yui"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "eco: olá Yui"
    assert body["model"] == "fake-model"
    uuid.UUID(body["conversation_id"])  # id válido


async def test_chat_persists_turn_with_deterministic_sequence(
    client: AsyncClient, db_session
) -> None:
    headers = await register_and_login(client)
    resp = await client.post("/api/v1/chat", json={"message": "primeira"}, headers=headers)
    conversation_id = uuid.UUID(resp.json()["conversation_id"])
    await client.post(
        "/api/v1/chat",
        json={"message": "segunda", "conversation_id": str(conversation_id)},
        headers=headers,
    )

    result = await db_session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.sequence)
    )
    messages = list(result.scalars())
    assert [(m.sequence, m.role) for m in messages] == [
        (1, "user"),
        (2, "assistant"),
        (3, "user"),
        (4, "assistant"),
    ]
    assert messages[0].content == "primeira"
    assert messages[2].content == "segunda"


async def test_chat_records_usage(client: AsyncClient, db_session) -> None:
    headers = await register_and_login(client)
    await client.post("/api/v1/chat", json={"message": "oi"}, headers=headers)

    record = (await db_session.execute(select(UsageRecord))).scalar_one()
    assert record.model == "fake-model"
    assert record.input_tokens == 10
    assert record.output_tokens == 5
    # Modelo desconhecido pela tabela de preços → custo nulo, nunca erro.
    assert record.estimated_cost_usd is None


async def test_chat_rehydrates_history_after_cache_loss(
    client: AsyncClient, fake_redis: FakeRedis, fake_llm
) -> None:
    headers = await register_and_login(client)
    resp = await client.post("/api/v1/chat", json={"message": "meu nome é Leo"}, headers=headers)
    conversation_id = resp.json()["conversation_id"]

    # Simula expiração do TTL / restart do Redis: cache é perdido.
    fake_redis.lists.clear()

    resp = await client.post(
        "/api/v1/chat",
        json={"message": "qual é o meu nome?", "conversation_id": conversation_id},
        headers=headers,
    )
    assert resp.status_code == 200

    # O histórico enviado ao modelo foi rehidratado do banco: contém o
    # turno anterior, não apenas a mensagem nova.
    _, sent_messages = fake_llm.calls[-1]
    contents = [m.content for m in sent_messages]
    assert "meu nome é Leo" in contents
    assert contents[-1] == "qual é o meu nome?"
