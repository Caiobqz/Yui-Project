"""Manutenção do Memory System: poda de memórias obsoletas.

Roda periodicamente (a cada N interações do usuário, disparada no
pós-turno) e remove memórias EXTRAÍDAS que envelheceram sem nunca serem
usadas e cuja relevância atual (importância × recência) caiu abaixo do
limiar. Memórias criadas explicitamente pelo usuário (`source="user"`)
NUNCA são podadas — só o próprio usuário as remove.

A consolidação (reforço de memórias reconfirmadas) acontece na escrita
(MemoryService.reinforce, chamada pelo MemoryAgent); a deduplicação
acontece na criação. Este módulo cuida apenas do esquecimento.
"""
import logging
import uuid
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.models.base import ensure_aware, utcnow
from app.models.memory import MemoryEntry
from app.services.memory_service import recency_factor

logger = logging.getLogger("yui.memory_maintenance")


class MemoryMaintenance:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def run(self, user_id: uuid.UUID) -> int:
        """Remove memórias obsoletas do usuário. Retorna quantas podou."""
        settings = get_settings()
        now = utcnow()
        min_age = timedelta(days=settings.memory_prune_min_age_days)

        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryEntry).where(
                    MemoryEntry.user_id == user_id,
                    MemoryEntry.source == "extracted",
                    MemoryEntry.usage_count == 0,
                )
            )
            pruned = 0
            for memory in result.scalars():
                if now - ensure_aware(memory.created_at) < min_age:
                    continue
                current_relevance = memory.relevance * recency_factor(memory, now)
                if current_relevance < settings.memory_prune_score_threshold:
                    await session.delete(memory)
                    pruned += 1
            await session.commit()

        if pruned:
            logger.info(
                "Manutenção de memória: %d memória(s) obsoleta(s) removida(s) "
                "para o usuário %s.",
                pruned,
                user_id,
            )
        return pruned
