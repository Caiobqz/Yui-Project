"""Testes da Bússola Moral e do Judgement Engine (ações autônomas)."""
import uuid
from datetime import timedelta

import pytest

from app.cognition.judgement import JudgementEngine
from app.cognition.moral_compass import MoralCompass, ProposedAction
from app.core.config import get_settings
from app.core.metrics import METRICS
from app.models.base import utcnow
from app.models.task import Task
from app.models.user import User


def _action(**kwargs) -> ProposedAction:
    base = dict(
        kind="check_in", description="acompanhar objetivo",
        benefit=0.8, risk=0.1, reversibility=1.0, urgency=0.5,
    )
    base.update(kwargs)
    return ProposedAction(**base)  # type: ignore[arg-type]


def test_beneficial_low_risk_action_proceeds() -> None:
    ev = MoralCompass().evaluate(_action(), alignment=0.6, permitted=True)
    assert ev.decision == "proceed"
    assert ev.confidence >= 0.55


def test_high_risk_low_reversibility_is_declined() -> None:
    ev = MoralCompass().evaluate(
        _action(risk=0.85, reversibility=0.2), alignment=0.6, permitted=True
    )
    assert ev.decision == "decline"


def test_high_risk_but_reversible_asks_user() -> None:
    ev = MoralCompass().evaluate(
        _action(risk=0.75, reversibility=0.8), alignment=0.6, permitted=True
    )
    assert ev.decision == "ask_user"


def test_low_benefit_action_is_deferred() -> None:
    ev = MoralCompass().evaluate(
        _action(benefit=0.1, urgency=0.1), alignment=0.0, permitted=True
    )
    assert ev.decision == "defer"


async def _abandoned_plan_user(db_session) -> uuid.UUID:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()
    parent = Task(user_id=user.id, title="Aprender inteligencia artificial")
    db_session.add(parent)
    await db_session.flush()
    db_session.add(
        Task(user_id=user.id, title="Estudar fundamentos", parent_id=parent.id, position=1)
    )
    await db_session.flush()
    parent.updated_at = utcnow() - timedelta(days=40)  # abandonado
    await db_session.flush()
    return user.id


async def test_propose_initiatives_approves_check_in_for_abandoned_plan(
    db_session,
) -> None:
    user_id = await _abandoned_plan_user(db_session)
    initiatives = await JudgementEngine().propose_initiatives(db_session, user_id)
    assert any(i.kind == "check_in" for i in initiatives)
    assert all(i.evaluation.decision == "proceed" for i in initiatives)


async def test_autonomy_disabled_yields_no_initiatives(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await _abandoned_plan_user(db_session)
    monkeypatch.setattr(get_settings(), "autonomy_enabled", False)
    assert await JudgementEngine().propose_initiatives(db_session, user_id) == []


async def test_deliberation_records_metrics(db_session) -> None:
    METRICS.reset()
    JudgementEngine().deliberate(_action(), alignment=0.6)
    snapshot = METRICS.snapshot()
    assert snapshot["counters"].get("judgement.decision.proceed") == 1
    assert "judgement.confidence" in snapshot["samples"]
