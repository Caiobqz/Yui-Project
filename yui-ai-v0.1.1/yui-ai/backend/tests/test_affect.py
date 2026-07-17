"""Testes do Affective State Model (v0.5) — determinismo, decaimento e prompt."""
import uuid
from datetime import timedelta

from app.cognition.affect import AffectService, AffectSnapshot
from app.cognition.emotional_context import analyze
from app.models.affect import AffectiveState
from app.models.base import utcnow
from app.models.user import User


async def _user(db_session) -> uuid.UUID:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()
    return user.id


async def test_snapshot_is_neutral_for_unknown_user(db_session) -> None:
    snapshot = await AffectService(db_session).snapshot(uuid.uuid4())
    assert snapshot.joy == 0.0
    assert snapshot.concern == 0.0
    # Estado em repouso não gasta tokens: nenhum bloco no prompt.
    assert snapshot.to_prompt() is None


async def test_positive_turn_raises_joy_and_warmth(db_session) -> None:
    user_id = await _user(db_session)
    service = AffectService(db_session)
    row = await service.register_turn(user_id, analyze("consegui, deu certo!"))
    assert row.joy >= 0.20
    assert row.warmth > 0.1  # convivência + conquista


async def test_frustration_raises_concern_and_dampens_joy(db_session) -> None:
    user_id = await _user(db_session)
    service = AffectService(db_session)
    await service.register_turn(user_id, analyze("consegui, deu certo!"))
    row = await service.register_turn(user_id, analyze("isso não funciona!!"))
    assert row.concern >= 0.15
    assert row.joy < 0.20  # a frustração reduz a alegria recente


async def test_dimensions_stay_clamped(db_session) -> None:
    user_id = await _user(db_session)
    service = AffectService(db_session)
    for _ in range(20):
        row = await service.register_turn(user_id, analyze("não funciona de novo!!"))
    assert 0.0 <= row.joy <= 1.0
    assert 0.0 <= row.concern <= 1.0
    assert 0.0 <= row.warmth <= 1.0


async def test_joy_decays_fast_but_warmth_persists(db_session) -> None:
    user_id = await _user(db_session)
    service = AffectService(db_session)
    row = await service.register_turn(user_id, analyze("finalmente deu certo!"))
    row.warmth = 0.5
    row.joy = 0.8
    # Envelhece o estado: 30 dias sem conversar.
    row.updated_at = utcnow() - timedelta(days=30)
    await db_session.flush()

    snapshot = await service.snapshot(user_id)
    # Alegria contextual (meia-vida 2d) praticamente volta à linha de base...
    assert snapshot.joy < 0.05
    # ...mas o apego (meia-vida 90d) permanece: a relação não reinicia.
    assert snapshot.warmth > 0.4


def test_prompt_describes_levels_without_numbers() -> None:
    snapshot = AffectSnapshot(warmth=0.5, joy=0.8, concern=0.2)
    prompt = snapshot.to_prompt()
    assert prompt is not None
    assert "apego" in prompt and "alegria" in prompt and "preocupação" in prompt
    assert "não emoção humana" in prompt
    # Nunca expõe números do modelo no prompt.
    assert "0.5" not in prompt and "0.8" not in prompt


async def test_register_turn_creates_row_once(db_session) -> None:
    user_id = await _user(db_session)
    service = AffectService(db_session)
    await service.register_turn(user_id, analyze("olá"))
    await service.register_turn(user_id, analyze("tudo bem?"))
    rows = (
        await db_session.execute(
            AffectiveState.__table__.select().where(
                AffectiveState.user_id == user_id
            )
        )
    ).all()
    assert len(rows) == 1
