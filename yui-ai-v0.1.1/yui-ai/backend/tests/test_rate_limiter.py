"""Testes do rate limiter (janela por minuto e orçamento diário de tokens)."""
import uuid

import pytest

import app.services.rate_limiter as rl_module
from app.core.exceptions import RateLimitExceededError
from app.services.rate_limiter import PlanLimits, RateLimiter
from tests.fakes import FakeRedis


@pytest.fixture
def tight_limits(monkeypatch: pytest.MonkeyPatch) -> PlanLimits:
    limits = PlanLimits(chat_per_minute=2, tokens_per_day=100)
    monkeypatch.setattr(rl_module, "get_plan_limits", lambda plan: limits)
    return limits


async def test_enforce_blocks_after_minute_limit(tight_limits: PlanLimits) -> None:
    limiter = RateLimiter(FakeRedis())  # type: ignore[arg-type]
    user_id = uuid.uuid4()

    await limiter.enforce(user_id, "free")
    await limiter.enforce(user_id, "free")
    with pytest.raises(RateLimitExceededError) as exc:
        await limiter.enforce(user_id, "free")
    assert exc.value.retry_after_seconds == 60


async def test_enforce_blocks_after_daily_token_budget(tight_limits: PlanLimits) -> None:
    redis = FakeRedis()
    limiter = RateLimiter(redis)  # type: ignore[arg-type]
    user_id = uuid.uuid4()

    await limiter.register_tokens(user_id, 100)  # atinge o orçamento
    with pytest.raises(RateLimitExceededError):
        await limiter.enforce(user_id, "free")


async def test_limits_are_per_user(tight_limits: PlanLimits) -> None:
    limiter = RateLimiter(FakeRedis())  # type: ignore[arg-type]
    user_a, user_b = uuid.uuid4(), uuid.uuid4()

    await limiter.enforce(user_a, "free")
    await limiter.enforce(user_a, "free")
    # A estourou; B continua liberado.
    with pytest.raises(RateLimitExceededError):
        await limiter.enforce(user_a, "free")
    await limiter.enforce(user_b, "free")


async def test_redis_failure_fails_open(tight_limits: PlanLimits) -> None:
    class BrokenRedis:
        async def incr(self, key: str) -> int:
            raise ConnectionError("redis fora do ar")

    limiter = RateLimiter(BrokenRedis())  # type: ignore[arg-type]
    # Decisão documentada: indisponibilidade do Redis não bloqueia o chat.
    await limiter.enforce(uuid.uuid4(), "free")
