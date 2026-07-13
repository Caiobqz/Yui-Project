"""Testes do loop agêntico: tool calling de ponta a ponta via API."""

from httpx import AsyncClient
from sqlalchemy import select

from app.models.task import Task
from app.models.usage import UsageRecord
from app.services.llm.base import LLMResponse, ToolCall
from tests.conftest import register_and_login
from tests.fakes import FakeLLM


def _tool_call_response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content="",
        model="fake-model",
        input_tokens=15,
        output_tokens=8,
        tool_calls=(ToolCall(id="call_1", name=name, arguments=arguments),),
    )


def _final_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, model="fake-model", input_tokens=12, output_tokens=6)


async def test_reminder_creates_task_via_tool_call(
    client: AsyncClient, fake_llm: FakeLLM, db_session
) -> None:
    """'Me lembre de estudar Python amanhã às 18h' → create_task → resposta."""
    headers = await register_and_login(client)
    fake_llm.script.extend(
        [
            _tool_call_response(
                "create_task",
                {"title": "Estudar Python", "due_at": "2026-07-13T18:00"},
            ),
            _final_response("Anotado! Vou te lembrar amanhã às 18h."),
        ]
    )

    resp = await client.post(
        "/api/v1/chat",
        json={"message": "Me lembre de estudar Python amanhã às 18h"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "Anotado! Vou te lembrar amanhã às 18h."

    task = (await db_session.execute(select(Task))).scalar_one()
    assert task.title == "Estudar Python"
    assert task.due_at is not None and task.due_at.hour == 18

    # O resultado da ferramenta voltou ao modelo na segunda chamada.
    _, second_call_messages, _ = fake_llm.calls[-1]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert "Tarefa criada" in tool_messages[0].content
    assert tool_messages[0].tool_call_id == "call_1"

    # Cada chamada de modelo do turno foi contabilizada.
    usage_records = list((await db_session.execute(select(UsageRecord))).scalars())
    assert len(usage_records) == 2

    # A tarefa aparece na API de consulta.
    resp = await client.get("/api/v1/tasks", headers=headers)
    assert [t["title"] for t in resp.json()] == ["Estudar Python"]


async def test_unknown_tool_is_blocked_by_guardian(
    client: AsyncClient, fake_llm: FakeLLM, db_session
) -> None:
    headers = await register_and_login(client)
    fake_llm.script.extend(
        [
            _tool_call_response("delete_everything", {}),
            _final_response("Não posso executar essa ação."),
        ]
    )

    resp = await client.post(
        "/api/v1/chat", json={"message": "apague tudo"}, headers=headers
    )
    assert resp.status_code == 200

    # O Guardian devolveu o erro ao modelo em vez de executar.
    _, second_call_messages, _ = fake_llm.calls[-1]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    assert "desconhecida" in tool_messages[0].content
    assert (await db_session.execute(select(Task))).scalar_one_or_none() is None


async def test_tool_loop_stops_at_iteration_limit(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    headers = await register_and_login(client)
    # O modelo pede ferramenta para sempre: o loop precisa parar sozinho.
    fake_llm.script.extend(
        _tool_call_response("list_tasks", {}) for _ in range(20)
    )

    resp = await client.post(
        "/api/v1/chat", json={"message": "liste em loop"}, headers=headers
    )
    assert resp.status_code == 200
    assert "Não consegui concluir" in resp.json()["reply"]
    # 5 iterações (llm_max_tool_iterations), não 20.
    assert len(fake_llm.calls) == 5


async def test_create_plan_tool_creates_parent_and_steps(
    client: AsyncClient, fake_llm: FakeLLM, db_session
) -> None:
    headers = await register_and_login(client)
    plan_json = (
        '{"title": "Aprender IA", "steps": '
        '["Estudar Python", "Aprender fundamentos de ML", "Construir um projeto"]}'
    )
    fake_llm.script.extend(
        [
            _tool_call_response("create_plan", {"goal": "quero aprender IA"}),
            # Chamada interna do PlannerAgent:
            LLMResponse(content=plan_json, model="fake-model", input_tokens=9, output_tokens=9),
            _final_response("Criei um plano com 3 etapas para você!"),
        ]
    )

    resp = await client.post(
        "/api/v1/chat", json={"message": "quero aprender IA"}, headers=headers
    )
    assert resp.status_code == 200

    tasks = list((await db_session.execute(select(Task).order_by(Task.position))).scalars())
    parents = [t for t in tasks if t.parent_id is None]
    children = [t for t in tasks if t.parent_id is not None]
    assert len(parents) == 1 and parents[0].title == "Aprender IA"
    assert [c.title for c in children] == [
        "Estudar Python",
        "Aprender fundamentos de ML",
        "Construir um projeto",
    ]
    assert all(c.parent_id == parents[0].id for c in children)


async def test_user_cannot_complete_another_users_task(
    client: AsyncClient, fake_llm: FakeLLM, db_session
) -> None:
    headers_a = await register_and_login(client, email="a@example.com")
    headers_b = await register_and_login(client, email="b@example.com")

    fake_llm.script.extend(
        [
            _tool_call_response("create_task", {"title": "Tarefa privada de A"}),
            _final_response("Criada."),
        ]
    )
    await client.post("/api/v1/chat", json={"message": "crie"}, headers=headers_a)
    task = (await db_session.execute(select(Task))).scalar_one()

    fake_llm.script.extend(
        [
            _tool_call_response("complete_task", {"task_id": str(task.id)}),
            _final_response("Não encontrei essa tarefa."),
        ]
    )
    await client.post(
        "/api/v1/chat", json={"message": "conclua"}, headers=headers_b
    )

    # Isolamento: a ferramenta respondeu 'não encontrada' e o status não mudou.
    _, msgs, _ = fake_llm.calls[-1]
    tool_messages = [m for m in msgs if m.role == "tool"]
    assert "não encontrada" in tool_messages[0].content
    await db_session.refresh(task)
    assert task.status == "pending"
