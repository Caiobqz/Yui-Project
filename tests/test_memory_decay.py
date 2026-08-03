"""Testes do decaimento e da manutenção (esquecimento) de memórias."""
import uuid
from datetime import timedelta

from sqlalchemy import select

from app.models.base import utcnow
from app.models.memory import MemoryEntry
from app.models.user import User
from app.services.memory_maintenance import MemoryMaintenance
from app.services.memory_service import memory_score, recency_factor


def _memory(memory_type: str, days_old: float, relevance: float = 0.5) -> MemoryEntry:
    entry = MemoryEntry(
        category="x", content="c", relevance=relevance, memory_type=memory_type
    )
    entry.created_at = utcnow() - timedelta(days=days_old)
    entry.last_used_at = None
    return entry


def test_episodic_decays_faster_than_semantic() -> None:
    episodic = _memory("episodic", days_old=60)
    semantic = _memory("semantic", days_old=60)
    assert recency_factor(episodic) < recency_factor(semantic)


def test_recency_has_floor() -> None:
    ancient = _memory("episodic", days_old=3650)
    assert recency_factor(ancient) == 0.3  # piso: continua encontrável


def test_fresh_memory_outranks_stale_one_with_same_similarity() -> None:
    fresh = _memory("episodic", days_old=0)
    stale = _memory("episodic", days_old=90)
    assert memory_score(0.8, fresh) > memory_score(0.8, stale)


async def test_maintenance_prunes_only_obsolete_extracted_memories(
    session_factory,
) -> None:
    async with session_factory() as session:
        user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
        session.add(user)
        await session.flush()
        user_id = user.id

        old = utcnow() - timedelta(days=120)

        obsolete = MemoryEntry(
            user_id=user_id, category="x", content="irrelevante",
            relevance=0.2, source="extracted", usage_count=0, memory_type="episodic",
        )
        used = MemoryEntry(
            user_id=user_id, category="x", content="usada com frequência",
            relevance=0.2, source="extracted", usage_count=5, memory_type="episodic",
        )
        user_created = MemoryEntry(
            user_id=user_id, category="x", content="criada pelo usuário",
            relevance=0.2, source="user", usage_count=0, memory_type="semantic",
        )
        recent = MemoryEntry(
            user_id=user_id, category="x", content="extraída ontem",
            relevance=0.2, source="extracted", usage_count=0, memory_type="episodic",
        )
        session.add_all([obsolete, used, user_created, recent])
        await session.flush()
        for entry in (obsolete, used, user_created):
            entry.created_at = old
        await session.commit()

    pruned = await MemoryMaintenance(session_factory).run(user_id)
    assert pruned == 1

    async with session_factory() as session:
        remaining = {
            m.content for m in (await session.execute(select(MemoryEntry))).scalars()
        }
    # Poda só a obsoleta: usada, criada pelo usuário e recente permanecem.
    assert remaining == {"usada com frequência", "criada pelo usuário", "extraída ontem"}
