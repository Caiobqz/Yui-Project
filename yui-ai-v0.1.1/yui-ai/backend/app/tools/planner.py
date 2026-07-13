"""Ferramentas de planejamento — delegam ao PlannerAgent e ao TaskService.

Planning System (v0.3): criar planos, acompanhar progresso e revisar
objetivos com sugestões de melhoria.
"""
import uuid
from typing import Any

from app.services.llm.base import ToolSpec
from app.services.task_service import TaskService
from app.tools.base import Tool, ToolContext


async def _create_plan(ctx: ToolContext, args: dict[str, Any]) -> str:
    # Import tardio: evita ciclo tools -> agents -> tools.
    from app.agents.planner_agent import PlannerAgent

    agent = PlannerAgent(ctx.llm)
    return await agent.create_plan(
        session_factory=ctx.session_factory,
        user_id=ctx.user_id,
        conversation_id=ctx.conversation_id,
        goal=str(args["goal"]),
    )


async def _get_plan_progress(ctx: ToolContext, args: dict[str, Any]) -> str:
    async with ctx.session_factory() as session:
        plans = await TaskService(session).plans_with_children(ctx.user_id)
    if not plans:
        return "O usuário ainda não tem planos criados."
    lines: list[str] = []
    for parent, children in plans:
        done, total = TaskService.progress(children)
        lines.append(f"• {parent.title} — {done}/{total} etapas [id {parent.id}]")
        pending = [t for t in children if t.status == "pending"]
        if pending:
            lines.append(f"  próxima etapa: {pending[0].title}")
    return "Planos e progresso:\n" + "\n".join(lines)


async def _review_plan(ctx: ToolContext, args: dict[str, Any]) -> str:
    from app.agents.planner_agent import PlannerAgent

    try:
        plan_id = uuid.UUID(str(args["plan_id"]))
    except ValueError:
        return "Erro: 'plan_id' deve ser um UUID (use get_plan_progress para obter ids)."
    agent = PlannerAgent(ctx.llm)
    return await agent.review_plan(
        session_factory=ctx.session_factory,
        user_id=ctx.user_id,
        conversation_id=ctx.conversation_id,
        plan_id=plan_id,
    )


def planner_tools() -> list[Tool]:
    return [
        Tool(
            spec=ToolSpec(
                name="create_plan",
                description=(
                    "Cria um plano estruturado para um objetivo do usuário, "
                    "dividindo-o em etapas acompanháveis. Use quando o usuário "
                    "expressar um objetivo que exige múltiplos passos."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "O objetivo do usuário."},
                    },
                    "required": ["goal"],
                },
            ),
            handler=_create_plan,
        ),
        Tool(
            spec=ToolSpec(
                name="get_plan_progress",
                description=(
                    "Mostra os planos do usuário com o progresso de cada um "
                    "(etapas concluídas/total e próxima etapa)."
                ),
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            handler=_get_plan_progress,
        ),
        Tool(
            spec=ToolSpec(
                name="review_plan",
                description=(
                    "Revisa um plano específico: progresso atual e sugestões "
                    "de ajuste nas etapas. Use quando o usuário quiser avaliar "
                    "ou replanejar um objetivo."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string", "description": "UUID do plano."},
                    },
                    "required": ["plan_id"],
                },
            ),
            handler=_review_plan,
        ),
    ]
