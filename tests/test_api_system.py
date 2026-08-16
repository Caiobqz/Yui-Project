"""Testes das rotas read-only do núcleo autônomo (v0.4)."""
from httpx import AsyncClient

from tests.conftest import register_and_login


async def test_self_endpoint_returns_identity_and_is_read_only(
    client: AsyncClient,
) -> None:
    headers = await register_and_login(client)
    resp = await client.get("/api/v1/self", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Yui"
    assert body["version"] == "0.5.1"
    assert body["health"] == "operational"
    assert "create_task" in body["tools_available"]
    # Não há rota de escrita para o Self Model.
    assert (await client.put("/api/v1/self", json={}, headers=headers)).status_code in (
        404,
        405,
    )


async def test_system_routes_require_auth(client: AsyncClient) -> None:
    for path in ("/api/v1/self", "/api/v1/metrics", "/api/v1/initiatives",
                 "/api/v1/knowledge-graph"):
        assert (await client.get(path)).status_code == 401


async def test_metrics_endpoint_shape(client: AsyncClient) -> None:
    headers = await register_and_login(client)
    # Gera um turno para popular métricas de raciocínio.
    await client.post("/api/v1/chat", json={"message": "olá"}, headers=headers)
    resp = await client.get("/api/v1/metrics", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "counters" in body and "samples" in body
    assert body["counters"].get("turn.processed", 0) >= 1


async def test_initiatives_and_graph_endpoints_return_ok(client: AsyncClient) -> None:
    headers = await register_and_login(client)
    initiatives = await client.get("/api/v1/initiatives", headers=headers)
    assert initiatives.status_code == 200
    assert isinstance(initiatives.json(), list)

    graph = await client.get("/api/v1/knowledge-graph", headers=headers)
    assert graph.status_code == 200
    assert "nodes" in graph.json() and "edges" in graph.json()
