"""Ferramenta de planejamento (create_plan) — delega ao PlannerAgent."""
from typing import Any

from app.services.llm.base import ToolSpec
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


def planner_tools() -> list[Tool]:
    return [
        Tool(
            spec=ToolSpec(
                name="create_plan",
                description=(
                    "Cria um plano estruturado para um objetivo do usuário, "
                    "dividindo-o em etapas acompanháveis. Use quando o usuário "
                    "expressar um objetivo que exige múltiplos passos "
                    "(ex.: 'quero aprender IA', 'quero organizar minha rotina')."
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
    ]
