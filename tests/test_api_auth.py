"""Testes de autenticação e de isolamento entre usuários."""
import uuid

from httpx import AsyncClient

from tests.conftest import register_and_login


async def test_register_login_me(client: AsyncClient) -> None:
    headers = await register_and_login(client)
    resp = await client.get("/api/v1/auth/me", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "leo@example.com"
    assert body["plan"] == "free"


async def test_register_duplicate_email_conflicts(client: AsyncClient) -> None:
    await register_and_login(client)
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": "leo@example.com", "password": "outra-senha-1"},
    )
    assert resp.status_code == 409


async def test_login_wrong_password_is_generic_401(client: AsyncClient) -> None:
    await register_and_login(client)
    resp = await client.post(
        "/api/v1/auth/login",
        json={"email": "leo@example.com", "password": "senha-errada"},
    )
    assert resp.status_code == 401
    # E-mail inexistente responde exatamente igual (não revela cadastro).
    resp2 = await client.post(
        "/api/v1/auth/login",
        json={"email": "ninguem@example.com", "password": "qualquer-uma"},
    )
    assert resp2.status_code == 401
    assert resp.json() == resp2.json()


async def test_protected_routes_require_token(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/memories")).status_code == 401
    assert (
        await client.post("/api/v1/chat", json={"message": "oi"})
    ).status_code == 401
    invalid = {"Authorization": "Bearer token-invalido"}
    assert (await client.get("/api/v1/memories", headers=invalid)).status_code == 401


async def test_user_cannot_access_another_users_memories(client: AsyncClient) -> None:
    headers_a = await register_and_login(client, email="a@example.com")
    headers_b = await register_and_login(client, email="b@example.com")

    resp = await client.post(
        "/api/v1/memories",
        json={"category": "estudos", "content": "Aprendendo Python", "relevance": 0.9},
        headers=headers_a,
    )
    assert resp.status_code == 201
    memory_id = resp.json()["id"]

    # B não vê a memória de A na listagem...
    resp = await client.get("/api/v1/memories", headers=headers_b)
    assert resp.status_code == 200
    assert resp.json() == []

    # ...nem consegue apagá-la conhecendo o id.
    resp = await client.delete(f"/api/v1/memories/{memory_id}", headers=headers_b)
    assert resp.status_code == 404

    # A continua vendo a própria memória.
    resp = await client.get("/api/v1/memories", headers=headers_a)
    assert [m["id"] for m in resp.json()] == [memory_id]


async def test_user_cannot_use_another_users_conversation(client: AsyncClient) -> None:
    headers_a = await register_and_login(client, email="a@example.com")
    headers_b = await register_and_login(client, email="b@example.com")

    resp = await client.post(
        "/api/v1/chat", json={"message": "olá"}, headers=headers_a
    )
    assert resp.status_code == 200
    conversation_id = resp.json()["conversation_id"]

    # B tenta continuar a conversa de A → 404 (não revela existência).
    resp = await client.post(
        "/api/v1/chat",
        json={"message": "invadindo", "conversation_id": conversation_id},
        headers=headers_b,
    )
    assert resp.status_code == 404

    # Conversa inexistente também é 404 (fix do fork silencioso).
    resp = await client.post(
        "/api/v1/chat",
        json={"message": "oi", "conversation_id": str(uuid.uuid4())},
        headers=headers_a,
    )
    assert resp.status_code == 404
