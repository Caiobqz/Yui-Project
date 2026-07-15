"""Testes do Goal Engine (progresso, status e termos de objetivos)."""
import uuid
from datetime import timedelta

from app.cognition.goal_engine import GoalEngine
from app.models.base import utcnow
from app.models.task import Task
from app.models.user import User


async def _user(db_session) -> uuid.UUID:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()
    return user.id


async def _plan(db_session, user_id, title, steps_done, steps_pending, idle_days=0.0):
    parent = Task(user_id=user_id, title=title)
    db_session.add(parent)
    await db_session.flush()
    pos = 0
    for _ in range(steps_done):
        pos += 1
        db_session.add(Task(user_id=user_id, title=f"etapa {pos}", parent_id=parent.id,
                            status="done", position=pos))
    for _ in range(steps_pending):
        pos += 1
        db_session.add(Task(user_id=user_id, title=f"etapa {pos}", parent_id=parent.id,
                            position=pos))
    await db_session.flush()
    if idle_days:
        parent.updated_at = utcnow() - timedelta(days=idle_days)
        await db_session.flush()
    return parent


async def test_status_classification(db_session) -> None:
    user_id = await _user(db_session)
    await _plan(db_session, user_id, "Ativo", 1, 2, idle_days=1)
    await _plan(db_session, user_id, "Parado", 1, 1, idle_days=10)
    await _plan(db_session, user_id, "Abandonado", 0, 3, idle_days=40)
    await _plan(db_session, user_id, "Concluido", 2, 0, idle_days=1)

    states = {s.title: s for s in await GoalEngine().analyze(db_session, user_id)}
    assert states["Ativo"].status == "active"
    assert states["Parado"].status == "stalled"
    assert states["Abandonado"].status == "abandoned"
    assert states["Concluido"].status == "completed"
    assert states["Ativo"].progress == 1 / 3
    assert states["Ativo"].next_step == "etapa 2"


async def test_active_goal_terms_only_from_active_and_stalled(db_session) -> None:
    user_id = await _user(db_session)
    await _plan(db_session, user_id, "Aprender inteligencia artificial", 0, 2, idle_days=1)
    await _plan(db_session, user_id, "Projeto concluido antigo", 2, 0, idle_days=1)

    terms = await GoalEngine().active_goal_terms(db_session, user_id)
    assert "inteligencia" in terms and "artificial" in terms
    assert "concluido" not in terms


async def test_find_stale_plan_picks_most_idle(db_session) -> None:
    user_id = await _user(db_session)
    await _plan(db_session, user_id, "Menos parado", 0, 2, idle_days=10)
    await _plan(db_session, user_id, "Mais parado", 0, 2, idle_days=40)

    stale = await GoalEngine().find_stale_plan(db_session, user_id)
    assert stale is not None and stale.title == "Mais parado"
