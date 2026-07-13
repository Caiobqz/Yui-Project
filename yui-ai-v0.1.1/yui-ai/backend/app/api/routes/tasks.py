"""Consulta de tarefas e planos do usuário autenticado.

Criação/conclusão acontecem pela conversa (ferramentas create_task,
create_plan, complete_task); esta rota dá visibilidade ao progresso.
"""
from fastapi import APIRouter, Query

from app.api.deps import CurrentUser, DbSession
from app.api.schemas import TaskResponse
from app.services.task_service import TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("")
async def list_tasks(
    user: CurrentUser,
    session: DbSession,
    status: str | None = Query(default=None, pattern="^(pending|done|cancelled)$"),
) -> list[TaskResponse]:
    tasks = await TaskService(session).list_by_user(user.id, status=status)
    return [TaskResponse.model_validate(t) for t in tasks]
