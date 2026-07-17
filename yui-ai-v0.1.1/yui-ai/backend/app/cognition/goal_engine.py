"""Goal Engine — acompanhamento determinístico de objetivos e planos.

Analisa os planos do usuário (tarefa pai + etapas) e deriva, sem LLM:
progresso, status (ativo / parado / abandonado / concluído), etapa atual e
bloqueio aparente. Detecta situações que justificam iniciativa da Yui
(abandono, oportunidade de conclusão) — que passam pelo Judgement Engine
antes de virar ação.
"""
import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cognition.attention import terms_of
from app.core.config import get_settings
from app.models.base import ensure_aware, utcnow
from app.models.task import Task

GoalStatus = Literal["active", "stalled", "abandoned", "completed"]


@dataclass(frozen=True)
class GoalState:
    plan_id: uuid.UUID
    title: str
    done: int
    total: int
    status: GoalStatus
    next_step: str | None
    idle_days: float

    @property
    def progress(self) -> float:
        return self.done / self.total if self.total else 0.0


class GoalEngine:
    async def analyze(
        self, session: AsyncSession, user_id: uuid.UUID
    ) -> list[GoalState]:
        result = await session.execute(
            select(Task)
            .where(Task.user_id == user_id, Task.parent_id.is_(None))
            .order_by(Task.created_at)
        )
        parents = list(result.scalars())
        if not parents:
            return []

        # Uma única query para todas as etapas (evita N+1: era uma query por
        # plano). Agrupa em memória preservando a ordem de posição.
        parent_ids = [p.id for p in parents]
        rows = await session.execute(
            select(Task).where(Task.parent_id.in_(parent_ids)).order_by(Task.position)
        )
        by_parent: dict[uuid.UUID, list[Task]] = {}
        for child in rows.scalars():
            if child.parent_id is not None:
                by_parent.setdefault(child.parent_id, []).append(child)

        states: list[GoalState] = []
        for parent in parents:
            children = by_parent.get(parent.id, [])
            if not children:
                continue  # tarefa avulsa, não é um plano/objetivo
            states.append(self._state(parent, children))
        return states

    @staticmethod
    def active_terms(states: list[GoalState]) -> set[str]:
        """Termos dos objetivos ativos/parados — insumo do Attention Manager."""
        terms: set[str] = set()
        for state in states:
            if state.status in ("active", "stalled"):
                terms |= terms_of(state.title)
                if state.next_step:
                    terms |= terms_of(state.next_step)
        return terms

    async def active_goal_terms(
        self, session: AsyncSession, user_id: uuid.UUID
    ) -> set[str]:
        return self.active_terms(await self.analyze(session, user_id))

    async def find_stale_plan(
        self, session: AsyncSession, user_id: uuid.UUID
    ) -> GoalState | None:
        """Plano mais antigo parado/abandonado (usado pela curiosidade)."""
        stale = [
            s
            for s in await self.analyze(session, user_id)
            if s.status in ("stalled", "abandoned")
        ]
        stale.sort(key=lambda s: s.idle_days, reverse=True)
        return stale[0] if stale else None

    def _state(self, parent: Task, children: list[Task]) -> GoalState:
        settings = get_settings()
        done = sum(1 for c in children if c.status == "done")
        total = len(children)
        pending = [c for c in children if c.status == "pending"]
        next_step = pending[0].title if pending else None
        idle_days = max(
            0.0,
            (utcnow() - ensure_aware(parent.updated_at)).total_seconds() / 86_400,
        )

        status: GoalStatus
        if not pending:
            status = "completed"
        elif idle_days >= settings.goal_abandoned_days:
            status = "abandoned"
        elif idle_days >= settings.plan_stale_days:
            status = "stalled"
        else:
            status = "active"

        return GoalState(
            plan_id=parent.id,
            title=parent.title,
            done=done,
            total=total,
            status=status,
            next_step=next_step,
            idle_days=idle_days,
        )
