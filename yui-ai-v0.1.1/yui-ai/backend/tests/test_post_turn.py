"""Testes do pós-turno cognitivo orquestrado pelo YuiCore."""
import json
import uuid

from sqlalchemy import select

from app.agents.yui_core import YuiCore
from app.memory.short_term import ShortTermMemory
from app.models.memory import MemoryEntry
from app.models.user import User
from app.models.user_profile import UserProfile
from app.services.llm.base import LLMResponse
from app.services.rate_limiter import RateLimiter
from app.tools.registry import build_default_registry
from tests.fakes import FakeEmbeddings, FakeLLM, FakeRedis


def _core(session_factory, chat_llm: FakeLLM, utility_llm: FakeLLM) -> YuiCore:
    redis = FakeRedis()
    return YuiCore(
        session_factory=session_factory,
        short_term=ShortTermMemory(redis),  # type: ignore[arg-type]
        llm=chat_llm,
        rate_limiter=RateLimiter(redis),  # type: ignore[arg-type]
        embeddings=FakeEmbeddings(),
        registry=build_default_registry(),
        utility_llm=utility_llm,
    )


async def _create_user(session_factory) -> uuid.UUID:
    async with session_factory() as session:
        user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
        session.add(user)
        await session.commit()
        return user.id


async def test_post_turn_stores_memories_and_adaptation(
    session_factory, monkeypatch
) -> None:
    monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
    from app.core.config import get_settings

    get_settings.cache_clear()

    try:
        user_id = await _create_user(session_factory)
        utility = FakeLLM(
            script=[
                LLMResponse(
                    content=json.dumps(
                        {
                            "memories": [
                                {
                                    "content": "Quer trabalhar com IA",
                                    "category": "objetivos",
                                    "type": "semantic",
                                    "importance": 0.9,
                                    "confidence": 0.9,
                                }
                            ],
                            "adaptation": ["Prefere exemplos práticos"],
                        }
                    ),
                    model="fake-utility",
                    input_tokens=5,
                    output_tokens=5,
                )
            ]
        )
        core = _core(session_factory, FakeLLM(), utility)

        await core.run_post_turn(
            user_id, uuid.uuid4(), "quero trabalhar com IA", "Ótimo objetivo!"
        )

        async with session_factory() as session:
            memory = (await session.execute(select(MemoryEntry))).scalar_one()
            assert memory.content == "Quer trabalhar com IA"
            profile = (await session.execute(select(UserProfile))).scalar_one()
            assert profile.preferences == ["Prefere exemplos práticos"]

        # A análise rodou no modelo utilitário, não no principal.
        assert len(utility.calls) == 1
        assert core._llm.calls == []  # type: ignore[attr-defined]
    finally:
        get_settings.cache_clear()


async def test_interaction_count_increments_per_turn(
    session_factory,
) -> None:
    user_id = await _create_user(session_factory)
    core = _core(session_factory, FakeLLM(), FakeLLM())

    await core.process_message(user_id, "free", "olá")
    await core.process_message(user_id, "free", "tudo bem?")

    async with session_factory() as session:
        profile = (await session.execute(select(UserProfile))).scalar_one()
        assert profile.interaction_count == 2
        assert profile.last_interaction_at is not None
