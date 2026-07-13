"""Testes da evolução do Planning System (progresso e revisão)."""
from httpx import AsyncClient
from sqlalchemy import select

from app.models.task import Task
from app.services.llm.base import LLMResponse, ToolCall
from tests.conftest import register_and_login
from tests.fakes import FakeLLM


def _tool_call(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content="",
        model="fake-model",
        tool_calls=(ToolCall(id="c1", name=name, arguments=arguments),),
    )


async def _create_plan(client: AsyncClient, fake_llm: FakeLLM, headers: dict) -> None:
    plan_json = '{"title": "Aprender IA", "steps": ["Estudar Python", "Estudar ML"]}'
    fake_llm.script.extend(
        [
            _tool_call("create_plan", {"goal": "aprender IA"}),
            LLMResponse(content=plan_json, model="fake-model"),
            LLMResponse(content="Plano criado!", model="fake-model"),
        ]
    )
    await client.post("/api/v1/chat", json={"message": "quero aprender IA"}, headers=headers)


async def test_get_plan_progress_tool(
    client: AsyncClient, fake_llm: FakeLLM, db_session
) -> None:
    headers = await register_and_login(client)
    await _create_plan(client, fake_llm, headers)

    # Conclui uma etapa e consulta o progresso.
    step = (
        await db_session.execute(select(Task).where(Task.title == "Estudar Python"))
    ).scalar_one()
    fake_llm.script.extend(
        [
            _tool_call("complete_task", {"task_id": str(step.id)}),
            LLMResponse(content="Concluída!", model="fake-model"),
            _tool_call("get_plan_progress", {}),
            LLMResponse(content="Você está na metade!", model="fake-model"),
        ]
    )
    await client.post("/api/v1/chat", json={"message": "conclui a etapa 1"}, headers=headers)
    resp = await client.post(
        "/api/v1/chat", json={"message": "como está meu plano?"}, headers=headers
    )
    assert resp.status_code == 200

    _, msgs, _ = fake_llm.calls[-1]
    tool_result = next(m for m in msgs if m.role == "tool").content
    assert "Aprender IA — 1/2 etapas" in tool_result
    assert "próxima etapa: Estudar ML" in tool_result


async def test_review_plan_tool_combines_progress_and_llm_suggestion(
    client: AsyncClient, fake_llm: FakeLLM, db_session
) -> None:
    headers = await register_and_login(client)
    await _create_plan(client, fake_llm, headers)

    parent = (
        await db_session.execute(select(Task).where(Task.parent_id.is_(None)))
    ).scalar_one()
    fake_llm.script.extend(
        [
            _tool_call("review_plan", {"plan_id": str(parent.id)}),
            # Chamada interna do PlannerAgent.review_plan:
            LLMResponse(
                content="Sugiro atacar 'Estudar Python' esta semana.",
                model="fake-model",
            ),
            LLMResponse(content="Revisei seu plano!", model="fake-model"),
        ]
    )
    resp = await client.post(
        "/api/v1/chat", json={"message": "revise meu plano"}, headers=headers
    )
    assert resp.status_code == 200

    _, msgs, _ = fake_llm.calls[-1]
    tool_result = next(m for m in msgs if m.role == "tool").content
    assert "0/2 etapas concluídas" in tool_result
    assert "Sugiro atacar" in tool_result
