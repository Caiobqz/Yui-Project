"""Planner Agent — transforma objetivos em planos com etapas acompanháveis.

O plano é persistido como uma tarefa pai com etapas filhas (ver Task);
o progresso deriva do status das etapas e pode ser consultado via
list_tasks/complete_task.
"""
import json
import logging
import re
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.services.llm.base import ChatMessage, LLMProvider
from app.services.task_service import TaskService
from app.services.usage_service import build_usage_record

logger = logging.getLogger("yui.planner")

_PLANNER_SYSTEM_PROMPT = """Você é o componente de planejamento da Yui.
Dado um objetivo do usuário, divida-o em 3 a 7 etapas concretas, ordenadas e \
acionáveis. Cada etapa deve caber em uma frase curta iniciada por verbo.

Responda SOMENTE com JSON válido, sem texto adicional:
{"title": "título curto do plano", "steps": ["etapa 1", "etapa 2", ...]}"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_plan(text: str, goal: str) -> tuple[str, list[str]]:
    """Extrai (título, etapas) do JSON; fallback: plano de etapa única."""
    candidate = text.strip()
    fence = _JSON_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1)
    try:
        data = json.loads(candidate)
        title = str(data["title"]).strip()[:255]
        steps = [str(s).strip()[:255] for s in data["steps"] if str(s).strip()]
        if title and steps:
            return title, steps[:10]
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Planner: resposta fora do formato; criando plano simples.")
    return goal[:255], [goal[:255]]


class PlannerAgent:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def create_plan(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        goal: str,
    ) -> str:
        # Chamada ao modelo SEM conexão de banco aberta.
        response = await self._llm.generate(
            _PLANNER_SYSTEM_PROMPT,
            [ChatMessage(role="user", content=f"Objetivo: {goal}")],
        )
        title, steps = _parse_plan(response.content, goal)

        async with session_factory() as session:
            parent, children = await TaskService(session).create_plan(
                user_id, title=title, steps=steps, goal=goal
            )
            session.add(build_usage_record(user_id, conversation_id, response))
            await session.commit()

        lines = "\n".join(f"{t.position}. {t.title}" for t in children)
        return (
            f"Plano criado: '{parent.title}' (id {parent.id})\n"
            f"{lines}\n"
            "As etapas foram salvas como tarefas e podem ser acompanhadas "
            "com list_tasks e concluídas com complete_task."
        )
