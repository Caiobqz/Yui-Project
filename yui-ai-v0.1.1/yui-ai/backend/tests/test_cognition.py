"""Testes do núcleo cognitivo: emocional, curiosidade, adaptação e relacionamento."""
import uuid
from datetime import timedelta

from app.cognition.curiosity import CuriosityEngine
from app.cognition.emotional_context import analyze
from app.cognition.user_model import UserModelService
from app.models.base import utcnow
from app.models.memory import MemoryEntry
from app.models.task import Task
from app.models.user import User

# --- Emotional Context Model ---------------------------------------------------


def test_detects_frustration() -> None:
    ctx = analyze("isso não funciona de novo!!")
    assert "frustracao" in ctx.signals
    assert any("frustrado" in g for g in ctx.guidance)


def test_detects_difficulty_and_hurry() -> None:
    assert "dificuldade" in analyze("não entendo esse erro, estou travado").signals
    assert "pressa" in analyze("preciso disso urgente, sem tempo").signals


def test_detects_motivation() -> None:
    ctx = analyze("finalmente consegui, deu certo!")
    assert "motivacao" in ctx.signals


def test_neutral_text_has_no_signals() -> None:
    ctx = analyze("qual a capital da França?")
    assert ctx.is_neutral


# --- Adaptation Engine + Relationship Model -------------------------------------


async def _user_and_profile(db_session) -> tuple[uuid.UUID, object]:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()
    profile = await UserModelService(db_session).get_or_create(user.id)
    return user.id, profile


async def test_adaptation_notes_dedupe_and_cap(db_session) -> None:
    _, profile = await _user_and_profile(db_session)

    added = UserModelService.apply_adaptation(
        profile, ["Prefere exemplos práticos", "Prefere exemplos práticos e diretos"]
    )
    assert added == 1  # a segunda é ~equivalente → deduplicada

    # Teto: as notas mais antigas saem. (Tokens distintos por nota — dígitos
    # sozinhos são descartados pela normalização lexical.)
    for i in range(20):
        UserModelService.apply_adaptation(
            profile, [f"Prefere comunicacao variante{i}"]
        )
    from app.core.config import get_settings

    assert len(profile.preferences) == get_settings().adaptation_max_notes


async def test_relationship_line_evolves(db_session) -> None:
    user_id, profile = await _user_and_profile(db_session)
    line = UserModelService.relationship_line(profile)
    assert line is not None and "primeira conversa" in line

    # UPDATE atômico: recarrega o objeto para observar o efeito.
    await UserModelService.register_interaction(db_session, user_id)
    await db_session.refresh(profile)
    line = UserModelService.relationship_line(profile)
    assert line is not None and "conversa de número 2" in line
    assert profile.last_interaction_at is not None


# --- Curiosity Engine ------------------------------------------------------------


async def test_curiosity_quiet_in_early_relationship(db_session) -> None:
    user_id, profile = await _user_and_profile(db_session)
    profile.interaction_count = 1
    hint = await CuriosityEngine().suggest(db_session, user_id, profile)
    assert hint is None


async def test_curiosity_asks_about_goals_when_unknown(db_session) -> None:
    user_id, profile = await _user_and_profile(db_session)
    profile.interaction_count = 5
    hint = await CuriosityEngine().suggest(db_session, user_id, profile)
    assert hint is not None
    assert "objetivos" in hint.reason


async def test_curiosity_prioritizes_stale_plan(db_session) -> None:
    user_id, profile = await _user_and_profile(db_session)
    profile.interaction_count = 5

    old = utcnow() - timedelta(days=30)
    parent = Task(user_id=user_id, title="Aprender IA")
    db_session.add(parent)
    await db_session.flush()
    child = Task(user_id=user_id, title="Estudar Python", parent_id=parent.id)
    db_session.add(child)
    await db_session.flush()
    # Envelhece o plano diretamente (updated_at controla a detecção).
    parent.updated_at = old
    await db_session.flush()

    hint = await CuriosityEngine().suggest(db_session, user_id, profile)
    assert hint is not None
    assert "Aprender IA" in hint.reason


async def test_completing_step_resets_plan_staleness(db_session) -> None:
    """Progresso recente numa etapa impede a cobrança indevida do plano."""
    from app.services.task_service import TaskService

    user_id, profile = await _user_and_profile(db_session)
    profile.interaction_count = 5

    old = utcnow() - timedelta(days=30)
    parent = Task(user_id=user_id, title="Aprender IA")
    db_session.add(parent)
    await db_session.flush()
    step_a = Task(user_id=user_id, title="Etapa A", parent_id=parent.id)
    step_b = Task(user_id=user_id, title="Etapa B", parent_id=parent.id)
    db_session.add_all([step_a, step_b])
    await db_session.flush()
    parent.updated_at = old
    await db_session.flush()

    # Concluir uma etapa "acorda" o plano (toca o updated_at do pai)...
    await TaskService(db_session).complete(user_id, step_a.id)
    hint = await CuriosityEngine().suggest(db_session, user_id, profile)
    # ...então a curiosidade cai na próxima lacuna (objetivos), não no plano.
    assert hint is None or "Aprender IA" not in hint.reason


async def test_adaptation_rejects_secret_looking_notes(db_session) -> None:
    _, profile = await _user_and_profile(db_session)
    added = UserModelService.apply_adaptation(
        profile, ["Prefere respostas curtas", "senha: hunter2"]
    )
    assert added == 1
    assert profile.preferences == ["Prefere respostas curtas"]


async def test_curiosity_silent_when_user_is_known(db_session) -> None:
    user_id, profile = await _user_and_profile(db_session)
    profile.interaction_count = 5
    db_session.add_all(
        [
            MemoryEntry(user_id=user_id, category="objetivos", content="Quer trabalhar com IA"),
            MemoryEntry(user_id=user_id, category="interesses", content="Gosta de Python"),
            MemoryEntry(user_id=user_id, category="preferencias", content="Prefere exemplos"),
        ]
    )
    await db_session.flush()
    hint = await CuriosityEngine().suggest(db_session, user_id, profile)
    assert hint is None
