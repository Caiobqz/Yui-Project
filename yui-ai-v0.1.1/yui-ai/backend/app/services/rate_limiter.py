"""Rate limiting e orçamento diário de tokens, por usuário, sobre Redis.

Duas proteções complementares:
- Janela fixa de requisições por minuto (protege contra loops/abuso).
- Contador diário de tokens (protege o custo com o provedor de IA).

Decisão consciente de disponibilidade: se o Redis estiver indisponível, o
limitador loga um aviso e PERMITE a requisição (fail-open) — a Yui é uma
assistente pessoal e indisponibilidade do cache não deve derrubar o chat.
Para um SaaS multiusuário, reavaliar para fail-closed.
"""
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import date

from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.exceptions import RateLimitExceededError

logger = logging.getLogger("yui.rate_limiter")


@dataclass(frozen=True)
class PlanLimits:
    chat_per_minute: int
    tokens_per_day: int


def get_plan_limits(plan: str) -> PlanLimits:
    """Limites por plano. Novos planos entram aqui (ex.: 'pro', 'unlimited')."""
    settings = get_settings()
    plans = {
        "free": PlanLimits(
            chat_per_minute=settings.rate_limit_chat_per_minute,
            tokens_per_day=settings.daily_token_limit,
        ),
    }
    return plans.get(plan, plans["free"])


class RateLimiter:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    @staticmethod
    def _minute_key(user_id: uuid.UUID) -> str:
        return f"yui:rl:chat:{user_id}:{int(time.time() // 60)}"

    @staticmethod
    def _tokens_key(user_id: uuid.UUID) -> str:
        return f"yui:rl:tokens:{user_id}:{date.today().isoformat()}"

    async def enforce(self, user_id: uuid.UUID, plan: str) -> None:
        """Levanta RateLimitExceededError se algum limite do plano foi atingido."""
        limits = get_plan_limits(plan)
        try:
            key = self._minute_key(user_id)
            count = await self._redis.incr(key)
            if count == 1:
                # 90s > janela de 60s: cobre relógios levemente defasados.
                await self._redis.expire(key, 90)
            if count > limits.chat_per_minute:
                raise RateLimitExceededError(
                    "Limite de mensagens por minuto atingido.",
                    retry_after_seconds=60,
                )

            used = await self._redis.get(self._tokens_key(user_id))
            if used is not None and int(used) >= limits.tokens_per_day:
                raise RateLimitExceededError(
                    "Limite diário de tokens atingido. Tente novamente amanhã."
                )
        except RateLimitExceededError:
            raise
        except Exception:  # noqa: BLE001 — fail-open documentado no módulo
            logger.warning(
                "Rate limiter indisponível (Redis?); permitindo requisição.",
                exc_info=True,
            )

    async def register_tokens(self, user_id: uuid.UUID, tokens: int) -> None:
        """Acumula tokens consumidos no dia (após a resposta do modelo)."""
        if tokens <= 0:
            return
        try:
            key = self._tokens_key(user_id)
            await self._redis.incrby(key, tokens)
            # 2 dias: sobrevive à virada de fuso sem acumular lixo.
            await self._redis.expire(key, 172_800)
        except Exception:  # noqa: BLE001
            logger.warning("Falha ao registrar tokens no Redis.", exc_info=True)
