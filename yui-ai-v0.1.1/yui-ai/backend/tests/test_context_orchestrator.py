"""Testes do Context Orchestrator (montador único + orçamento por bloco)."""
import uuid

from app.agents.memory_agent import MemoryAgent
from app.cognition.emotional_context import EmotionalContext
from app.cognition.self_model import build_self_model
from app.models.memory import MemoryEntry
from app.models.task import Task
from app.models.user import User
from app.services.context_orchestrator import (
    ContextOrchestrator,
    _clip,
    _clip_lines,
    _clip_memories,
)
from app.tools.registry import build_default_registry
from tests.fakes import FakeEmbeddings, FakeLLM


def _orchestrator(session_factory) -> ContextOrchestrator:
    agent = MemoryAgent(FakeLLM(), FakeEmbeddings(), session_factory)
    return ContextOrchestrator(agent, build_self_model(build_default_registry()))


def test_clip_truncates_to_budget() -> None:
    assert _clip("abcdefghij", 5).endswith("…")
    assert _clip("abc", 10) == "abc"


def test_clip_lines_respects_budget() -> None:
    lines = ["a" * 100, "b" * 100, "c" * 100]
    assert _clip_lines(lines, 150) == ["a" * 100]


def test_clip_memories_respects_budget() -> None:
    mems = [MemoryEntry(category="x", content="y" * 100) for _ in range(5)]
    kept = _clip_memories(mems, 150)
    assert len(kept) == 1


async def _user(db_session) -> uuid.UUID:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()
    return user.id


async def test_build_includes_self_world_and_goal_blocks(
    session_factory, db_session
) -> None:
    user_id = await _user(db_session)
    parent = Task(user_id=user_id, title="Aprender IA")
    db_session.add(parent)
    await db_session.flush()
    db_session.add(Task(user_id=user_id, title="Estudar Python", parent_id=parent.id))
    await db_session.commit()

    orchestrator = _orchestrator(session_factory)
    async with session_factory() as session:
        prompt, used = await orchestrator.build(
            session=session,
            user_id=user_id,
            text="olá",
            query_embedding=None,
            summary=None,
            adaptation_notes=[],
            relationship=None,
            emotional=EmotionalContext(signals=(), guidance=()),
            curiosity=None,
        )

    # Domínios do World Model presentes e rotulados, sem mistura.
    assert "<modelo_da_yui>" in prompt
    assert "<ambiente>" in prompt
    assert "<objetivos_ativos>" in prompt
    assert "Aprender IA" in prompt
    # Identidade (estável) antes de tudo.
    assert prompt.index("Você é Yui") < prompt.index("<modelo_da_yui>")
    assert used == 0


async def test_build_includes_affect_and_initiative_blocks(
    session_factory, db_session
) -> None:
    from app.cognition.affect import AffectSnapshot

    user_id = await _user(db_session)
    await db_session.commit()

    orchestrator = _orchestrator(session_factory)
    async with session_factory() as session:
        prompt, _ = await orchestrator.build(
            session=session,
            user_id=user_id,
            text="olá",
            query_embedding=None,
            summary=None,
            adaptation_notes=[],
            relationship=None,
            emotional=EmotionalContext(signals=(), guidance=()),
            curiosity=None,
            affect=AffectSnapshot(warmth=0.5, joy=0.8, concern=0.0),
            initiative="Retomar o objetivo 'Voltar a correr'.",
        )

    # Estado afetivo no domínio SELF; iniciativa como diretiva de estratégia.
    assert "<estado_afetivo>" in prompt
    assert "alegria" in prompt
    assert "Iniciativa própria" in prompt
    assert "Voltar a correr" in prompt


async def test_neutral_affect_adds_no_block(session_factory, db_session) -> None:
    from app.cognition.affect import AffectSnapshot

    user_id = await _user(db_session)
    await db_session.commit()

    orchestrator = _orchestrator(session_factory)
    async with session_factory() as session:
        prompt, _ = await orchestrator.build(
            session=session,
            user_id=user_id,
            text="olá",
            query_embedding=None,
            summary=None,
            adaptation_notes=[],
            relationship=None,
            emotional=EmotionalContext(signals=(), guidance=()),
            curiosity=None,
            affect=AffectSnapshot(warmth=0.1, joy=0.0, concern=0.0),
        )
    assert "<estado_afetivo>" not in prompt
