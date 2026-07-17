"""Testes das iniciativas persistentes (v0.5): raras, únicas e entregues."""
import uuid
from datetime import timedelta

import pytest

from app.core.config import get_settings
from app.models.base import utcnow
from app.models.task import Task
from app.models.user import User
from app.services.initiative_service import InitiativeService


async def _user_with_abandoned_plan(db_session, title: str = "Aprender IA") -> uuid.UUID:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()
    return await _add_abandoned_plan(db_session, user.id, title)


async def _add_abandoned_plan(db_session, user_id: uuid.UUID, title: str) -> uuid.UUID:
    parent = Task(user_id=user_id, title=title)
    db_session.add(parent)
    await db_session.flush()
    db_session.add(
        Task(user_id=user_id, title=f"Etapa de {title}", parent_id=parent.id)
    )
    await db_session.flush()
    parent.updated_at = utcnow() - timedelta(days=40)  # abandonado
    await db_session.flush()
    return user_id


async def test_generate_records_approved_initiative(db_session) -> None:
    user_id = await _user_with_abandoned_plan(db_session)
    service = InitiativeService(db_session)
    assert await service.generate(user_id) == 1

    record = await service.pending_for_turn(user_id)
    assert record is not None
    assert record.kind == "check_in"
    assert record.status == "pending"
    assert record.dedupe_key.startswith("check_in:")


async def test_same_situation_is_never_reproposed(db_session) -> None:
    user_id = await _user_with_abandoned_plan(db_session)
    service = InitiativeService(db_session)
    assert await service.generate(user_id) == 1
    # Mesma situação (mesmo plano abandonado): cooldown bloqueia a repetição,
    # inclusive depois da entrega.
    assert await service.generate(user_id) == 0
    record = await service.pending_for_turn(user_id)
    assert record is not None
    await service.mark_delivered(user_id, record.id)
    assert await service.generate(user_id) == 0


async def test_pending_cap_keeps_initiatives_rare(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await _user_with_abandoned_plan(db_session, "Aprender IA")
    await _add_abandoned_plan(db_session, user_id, "Escrever livro")
    monkeypatch.setattr(get_settings(), "initiative_max_pending", 1)
    try:
        assert await InitiativeService(db_session).generate(user_id) == 1
    finally:
        monkeypatch.undo()


async def test_mark_delivered_consumes_pending(db_session) -> None:
    user_id = await _user_with_abandoned_plan(db_session)
    service = InitiativeService(db_session)
    await service.generate(user_id)
    record = await service.pending_for_turn(user_id)
    assert record is not None

    await service.mark_delivered(user_id, record.id)
    assert await service.pending_for_turn(user_id) is None
    delivered = (await service.list_recent(user_id))[0]
    assert delivered.status == "delivered"
    assert delivered.delivered_at is not None


async def test_generate_respects_autonomy_flag(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await _user_with_abandoned_plan(db_session)
    monkeypatch.setattr(get_settings(), "autonomy_enabled", False)
    try:
        assert await InitiativeService(db_session).generate(user_id) == 0
    finally:
        monkeypatch.undo()


async def test_full_turn_delivers_pending_initiative(
    client, session_factory, fake_llm
) -> None:
    """Fluxo completo: a pendente entra no prompt do turno e é consumida."""
    from sqlalchemy import select

    from tests.conftest import register_and_login

    headers = await register_and_login(client, email="turno@example.com")
    async with session_factory() as session:
        user_id = (
            await session.execute(
                select(User.id).where(User.email == "turno@example.com")
            )
        ).scalar_one()
        await _add_abandoned_plan(session, user_id, "Voltar a correr")
        await InitiativeService(session).generate(user_id)
        await session.commit()

    resp = await client.post(
        "/api/v1/chat", json={"message": "oi, tudo bem?"}, headers=headers
    )
    assert resp.status_code == 200
    system_prompt = fake_llm.calls[-1][0]
    assert "Iniciativa própria" in system_prompt
    assert "Voltar a correr" in system_prompt
    assert "momento oportuno" in system_prompt

    # Entregue: não volta a ser oferecida no próximo turno.
    async with session_factory() as session:
        assert await InitiativeService(session).pending_for_turn(user_id) is None
    await client.post("/api/v1/chat", json={"message": "e aí?"}, headers=headers)
    assert "Iniciativa própria" not in fake_llm.calls[-1][0]


async def test_initiatives_endpoint_lists_persisted_records(
    client, session_factory
) -> None:
    from sqlalchemy import select

    from tests.conftest import register_and_login

    headers = await register_and_login(client, email="init@example.com")
    async with session_factory() as session:
        user_id = (
            await session.execute(
                select(User.id).where(User.email == "init@example.com")
            )
        ).scalar_one()
        await _add_abandoned_plan(session, user_id, "Estudar japonês")
        await InitiativeService(session).generate(user_id)
        await session.commit()

    resp = await client.get("/api/v1/initiatives", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    item = body[0]
    # Campos da v0.4 preservados + novos campos aditivos.
    assert item["kind"] == "check_in"
    assert item["decision"] == "proceed"
    assert item["status"] == "pending"
    assert {"id", "description", "confidence", "score", "rationale", "created_at"} <= set(item)
