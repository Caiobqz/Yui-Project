"""Testes do Permission System: defaults, decisões do usuário e Guardian."""
from httpx import AsyncClient

from app.agents.guardian import GuardianAgent
from app.services.llm.base import LLMResponse, ToolCall, ToolSpec
from app.tools.base import Tool
from app.tools.registry import ToolRegistry
from tests.conftest import register_and_login
from tests.fakes import FakeLLM


async def _noop(ctx, args) -> str:
    return "ok"


def _sensitive_tool_registry() -> ToolRegistry:
    """Simula uma ferramenta futura de categoria sensível (deny-by-default)."""
    return ToolRegistry(
        [
            Tool(
                spec=ToolSpec(
                    name="read_file",
                    description="Lê um arquivo do sistema",
                    input_schema={"type": "object", "properties": {}, "required": []},
                ),
                handler=_noop,
                category="arquivos",
                default_allowed=False,
            )
        ]
    )


def test_sensitive_tool_is_denied_by_default() -> None:
    guardian = GuardianAgent()
    error = guardian.validate_tool_call(
        _sensitive_tool_registry(), ToolCall(id="1", name="read_file", arguments={})
    )
    assert error is not None and "não está autorizada" in error


def test_user_grant_overrides_sensitive_default() -> None:
    guardian = GuardianAgent()
    error = guardian.validate_tool_call(
        _sensitive_tool_registry(),
        ToolCall(id="1", name="read_file", arguments={}),
        permission_overrides={"read_file": True},
    )
    assert error is None


async def test_user_can_revoke_a_default_allowed_tool(
    client: AsyncClient, fake_llm: FakeLLM, db_session
) -> None:
    headers = await register_and_login(client)

    # Revoga create_task via API de permissões.
    resp = await client.put(
        "/api/v1/permissions/create_task", json={"allowed": False}, headers=headers
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "tool_name": "create_task",
        "category": "produtividade",
        "allowed": False,
        "source": "user",
    }

    # O modelo tenta usar a ferramenta → Guardian bloqueia.
    fake_llm.script.extend(
        [
            LLMResponse(
                content="",
                model="fake-model",
                tool_calls=(
                    ToolCall(id="c1", name="create_task", arguments={"title": "X"}),
                ),
            ),
            LLMResponse(content="Sem permissão para criar tarefas.", model="fake-model"),
        ]
    )
    resp = await client.post(
        "/api/v1/chat", json={"message": "crie uma tarefa"}, headers=headers
    )
    assert resp.status_code == 200
    _, msgs, _ = fake_llm.calls[-1]
    tool_messages = [m for m in msgs if m.role == "tool"]
    assert "não está autorizada" in tool_messages[0].content

    from sqlalchemy import select

    from app.models.task import Task

    assert (await db_session.execute(select(Task))).scalar_one_or_none() is None


async def test_permissions_listing_reflects_defaults_and_user_choice(
    client: AsyncClient,
) -> None:
    headers = await register_and_login(client)
    await client.put(
        "/api/v1/permissions/create_note", json={"allowed": False}, headers=headers
    )

    resp = await client.get("/api/v1/permissions", headers=headers)
    assert resp.status_code == 200
    entries = {e["tool_name"]: e for e in resp.json()}
    assert entries["create_note"]["allowed"] is False
    assert entries["create_note"]["source"] == "user"
    assert entries["create_task"]["allowed"] is True
    assert entries["create_task"]["source"] == "default"


async def test_unknown_tool_permission_is_404(client: AsyncClient) -> None:
    headers = await register_and_login(client)
    resp = await client.put(
        "/api/v1/permissions/rm_rf", json={"allowed": True}, headers=headers
    )
    assert resp.status_code == 404
