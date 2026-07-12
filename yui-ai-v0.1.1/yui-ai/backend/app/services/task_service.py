"""Tarefas e planos: criação, listagem, conclusão e progresso."""
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.task import Task


class TaskService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        user_id: uuid.UUID,
        title: str,
        description: str | None = None,
        due_at: datetime | None = None,
        parent_id: uuid.UUID | None = None,
        position: int = 0,
    ) -> Task:
        task = Task(
            user_id=user_id,
            title=title,
            description=description,
            due_at=due_at,
            parent_id=parent_id,
            position=position,
        )
        self._session.add(task)
        await self._session.flush()
        return task

    async def create_plan(
        self, user_id: uuid.UUID, title: str, steps: list[str], goal: str
    ) -> tuple[Task, list[Task]]:
        """Cria um plano (tarefa pai) com etapas ordenadas (tarefas filhas)."""
        parent = await self.create(
            user_id, title=title, description=f"Plano para: {goal}"
        )
        children = [
            await self.create(
                user_id, title=step, parent_id=parent.id, position=index
            )
            for index, step in enumerate(steps, start=1)
        ]
        return parent, children

    async def list_by_user(
        self, user_id: uuid.UUID, status: str | None = None
    ) -> list[Task]:
        query = select(Task).where(Task.user_id == user_id)
        if status is not None:
            query = query.where(Task.status == status)
        query = query.order_by(Task.created_at, Task.position)
        result = await self._session.execute(query)
        return list(result.scalars())

    async def list_children(self, parent_id: uuid.UUID) -> list[Task]:
        result = await self._session.execute(
            select(Task).where(Task.parent_id == parent_id).order_by(Task.position)
        )
        return list(result.scalars())

    async def complete(self, user_id: uuid.UUID, task_id: uuid.UUID) -> Task | None:
        result = await self._session.execute(
            select(Task).where(Task.id == task_id, Task.user_id == user_id)
        )
        task = result.scalar_one_or_none()
        if task is None:
            return None
        task.status = "done"
        return task

    @staticmethod
    def progress(children: list[Task]) -> tuple[int, int]:
        """(etapas concluídas, total de etapas)."""
        done = sum(1 for t in children if t.status == "done")
        return done, len(children)
