"""Curiosity Engine — identifica lacunas e oportunidades de ajudar.

Determinístico e barato: examina o que a Yui SABE (memórias por categoria,
planos e seu progresso, tamanho do relacionamento) e sugere NO MÁXIMO uma
pergunta por turno, injetada no prompt como sugestão — o modelo decide se
o momento é natural para perguntar (regra da identidade).

Lacunas detectadas:
- usuário já interagiu o suficiente, mas a Yui não conhece seus objetivos;
- plano sem progresso há N dias;
- interesses conhecidos, mas nenhuma preferência de como ajudar.
"""
import uuid
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.base import utcnow
from app.models.memory import MemoryEntry
from app.models.task import Task
from app.models.user_profile import UserProfile


@dataclass(frozen=True)
class CuriosityHint:
    reason: str
    suggestion: str


class CuriosityEngine:
    async def suggest(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        profile: UserProfile,
    ) -> CuriosityHint | None:
        settings = get_settings()
        if not settings.curiosity_enabled:
            return None
        if profile.interaction_count < settings.curiosity_min_interactions:
            # Início do relacionamento: deixa a conversa fluir sem perguntas.
            return None

        # Lacuna 1: plano parado (oportunidade concreta de ajudar).
        stale_plan = await self._find_stale_plan(session, user_id)
        if stale_plan is not None:
            return CuriosityHint(
                reason=f"plano '{stale_plan.title}' sem progresso recente",
                suggestion=(
                    f"O plano '{stale_plan.title}' está sem progresso há alguns "
                    "dias. Se for natural, pergunte como está indo e se o "
                    "usuário quer ajustar as etapas."
                ),
            )

        # Lacuna 2: a Yui ainda não conhece os objetivos do usuário.
        counts = await self._memory_counts_by_category(session, user_id)
        if counts.get("objetivos", 0) == 0:
            return CuriosityHint(
                reason="nenhuma memória na categoria 'objetivos'",
                suggestion=(
                    "Você ainda não conhece os objetivos deste usuário. Se for "
                    "natural, pergunte o que ele está buscando alcançar no momento."
                ),
            )

        # Lacuna 3: conhece interesses, mas não preferências de como ajudar.
        if counts.get("interesses", 0) > 0 and counts.get("preferencias", 0) == 0:
            return CuriosityHint(
                reason="interesses conhecidos sem preferências registradas",
                suggestion=(
                    "Você conhece interesses do usuário, mas não como ele "
                    "prefere ser ajudado. Se for natural, pergunte como ele "
                    "gosta de aprender ou trabalhar."
                ),
            )

        return None

    @staticmethod
    async def _memory_counts_by_category(
        session: AsyncSession, user_id: uuid.UUID
    ) -> dict[str, int]:
        result = await session.execute(
            select(MemoryEntry.category, func.count())
            .where(MemoryEntry.user_id == user_id)
            .group_by(MemoryEntry.category)
        )
        return {category: count for category, count in result.all()}

    @staticmethod
    async def _find_stale_plan(
        session: AsyncSession, user_id: uuid.UUID
    ) -> Task | None:
        """Plano (tarefa pai) com etapas pendentes e sem atualização recente."""
        cutoff = utcnow() - timedelta(days=get_settings().plan_stale_days)
        pending_children = (
            select(Task.parent_id)
            .where(Task.user_id == user_id, Task.status == "pending")
            .where(Task.parent_id.is_not(None))
        )
        result = await session.execute(
            select(Task)
            .where(
                Task.user_id == user_id,
                Task.parent_id.is_(None),
                Task.id.in_(pending_children),
                Task.updated_at < cutoff,
            )
            .order_by(Task.updated_at)
            .limit(1)
        )
        return result.scalar_one_or_none()
