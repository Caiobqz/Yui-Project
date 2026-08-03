"""Ferramentas de tarefas e lembretes (create_task, list_tasks, complete_task)."""
import uuid
from datetime import UTC, datetime
from typing import Any

from app.models.task import Task
from app.services.llm.base import ToolSpec
from app.services.task_service import TaskService
from app.tools.base import Tool, ToolContext


def _parse_due_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    due = datetime.fromisoformat(raw)
    # Sem fuso explícito, assume UTC (o modelo é instruído a enviar ISO 8601).
    return due if due.tzinfo is not None else due.replace(tzinfo=UTC)


def _format_task(task: Task) -> str:
    due = f" (para {task.due_at.isoformat()})" if task.due_at else ""
    marker = "✔" if task.status == "done" else "•"
    return f"{marker} {task.title}{due} [id {task.id}]"


async def _create_task(ctx: ToolContext, args: dict[str, Any]) -> str:
    try:
        due_at = _parse_due_at(args.get("due_at"))
    except ValueError:
        return "Erro: 'due_at' deve estar no formato ISO 8601 (ex.: 2026-07-13T18:00)."
    async with ctx.session_factory() as session:
        task = await TaskService(session).create(
            ctx.user_id,
            title=str(args["title"]),
            description=args.get("description"),
            due_at=due_at,
        )
        await session.commit()
        return f"Tarefa criada: {_format_task(task)}"


async def _list_tasks(ctx: ToolContext, args: dict[str, Any]) -> str:
    status = args.get("status") or "pending"
    if status == "all":
        status = None
    async with ctx.session_factory() as session:
        tasks = await TaskService(session).list_by_user(ctx.user_id, status=status)
    if not tasks:
        return "Nenhuma tarefa encontrada."
    lines = []
    for task in tasks:
        prefix = "  ↳ " if task.parent_id else ""
        lines.append(prefix + _format_task(task))
    return "Tarefas:\n" + "\n".join(lines)


async def _complete_task(ctx: ToolContext, args: dict[str, Any]) -> str:
    try:
        task_id = uuid.UUID(str(args["task_id"]))
    except ValueError:
        return "Erro: 'task_id' deve ser um UUID válido (use list_tasks para obter ids)."
    async with ctx.session_factory() as session:
        task = await TaskService(session).complete(ctx.user_id, task_id)
        if task is None:
            return "Tarefa não encontrada."
        await session.commit()
        return f"Tarefa concluída: {task.title}"


def task_tools() -> list[Tool]:
    return [
        Tool(
            spec=ToolSpec(
                name="create_task",
                description=(
                    "Cria uma tarefa ou lembrete para o usuário. Use quando o "
                    "usuário pedir para lembrá-lo de algo ou registrar algo a fazer."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Título curto da tarefa."},
                        "description": {"type": "string", "description": "Detalhes opcionais."},
                        "due_at": {
                            "type": "string",
                            "description": "Prazo em ISO 8601 (ex.: 2026-07-13T18:00). Opcional.",
                        },
                    },
                    "required": ["title"],
                },
            ),
            handler=_create_task,
        ),
        Tool(
            spec=ToolSpec(
                name="list_tasks",
                description="Lista as tarefas do usuário (pendentes por padrão).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["pending", "done", "all"],
                            "description": "Filtro de status (padrão: pending).",
                        },
                    },
                    "required": [],
                },
            ),
            handler=_list_tasks,
        ),
        Tool(
            spec=ToolSpec(
                name="complete_task",
                description="Marca uma tarefa como concluída (obtenha o id com list_tasks).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "UUID da tarefa."},
                    },
                    "required": ["task_id"],
                },
            ),
            handler=_complete_task,
        ),
    ]
