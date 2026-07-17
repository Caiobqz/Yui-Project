"""Affective State Model — estados afetivos computacionais persistentes (v0.5).

Realismo (regra da identidade): a Yui NÃO sente emoções humanas. Estas
dimensões são estados computacionais que persistem entre conversas, decaem
com o tempo e influenciam decisões futuras — o tom da companhia (bloco do
prompt) e o julgamento de iniciativas (Bússola Moral, dimensão `concern`).

Determinístico de ponta a ponta (princípio v0.4): eventos do turno — os
sinais do Emotional Context Model — aplicam deltas fixos; cada dimensão
decai exponencialmente para sua linha de base com meia-vida própria:

- warmth   apego pela convivência: cresce um pouco a cada conversa e decai
           devagar — a relação não "reinicia" com a ausência;
- joy      alegria contextual: sobe com conquistas do usuário, decai rápido;
- concern  preocupação protetora: sobe com frustração/dificuldade, decai em
           dias — preocupação antiga não vira ansiedade permanente.
"""
import math
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cognition.emotional_context import EmotionalContext
from app.core.metrics import METRICS
from app.models.affect import AffectiveState
from app.models.base import ensure_aware, utcnow

# Linha de base de cada dimensão (estado "em repouso").
_BASELINES = {"warmth": 0.1, "joy": 0.0, "concern": 0.0}
# Meia-vida do decaimento em direção à linha de base, em dias.
_HALF_LIFE_DAYS = {"warmth": 90.0, "joy": 2.0, "concern": 7.0}

# Deltas por evento do turno (sinais do Emotional Context Model).
_TURN_WARMTH_DELTA = 0.01  # convivência: todo turno aproxima um pouco
_SIGNAL_DELTAS: dict[str, dict[str, float]] = {
    "motivacao": {"joy": 0.20, "warmth": 0.02},
    "frustracao": {"concern": 0.15, "joy": -0.05},
    "dificuldade": {"concern": 0.10},
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _decayed(value: float, dimension: str, elapsed_days: float) -> float:
    baseline = _BASELINES[dimension]
    half_life = _HALF_LIFE_DAYS[dimension]
    return baseline + (value - baseline) * math.pow(0.5, elapsed_days / half_life)


def _level(value: float) -> str | None:
    """Discretiza uma dimensão em linguagem natural (nunca números no prompt)."""
    if value < 0.15:
        return None
    if value < 0.40:
        return "leve"
    if value < 0.70:
        return "presente"
    return "forte"


@dataclass(frozen=True)
class AffectSnapshot:
    """Estado afetivo já decaído para o instante da leitura."""

    warmth: float
    joy: float
    concern: float

    def to_prompt(self) -> str | None:
        """Bloco do prompt; None quando tudo está na linha de base (economia)."""
        parts: list[str] = []
        if (level := _level(self.warmth)) is not None:
            parts.append(f"apego pela convivência {level}")
        if (level := _level(self.joy)) is not None:
            parts.append(f"alegria recente {level}")
        if (level := _level(self.concern)) is not None:
            parts.append(f"preocupação com o usuário {level}")
        if not parts:
            return None
        return (
            "Seu estado afetivo atual (modelo computacional persistente, não "
            "emoção humana): " + "; ".join(parts) + ". Deixe-o transparecer "
            "com naturalidade e sobriedade — nunca o descreva em números nem "
            "finja sentimentos humanos."
        )


_NEUTRAL = AffectSnapshot(
    warmth=_BASELINES["warmth"], joy=_BASELINES["joy"], concern=_BASELINES["concern"]
)


class AffectService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def snapshot(self, user_id: uuid.UUID) -> AffectSnapshot:
        """Estado atual (com decaimento aplicado). Leitura pura: não escreve."""
        row = await self._session.scalar(
            select(AffectiveState).where(AffectiveState.user_id == user_id)
        )
        if row is None:
            return _NEUTRAL
        elapsed = self._elapsed_days(row)
        return AffectSnapshot(
            warmth=_clamp(_decayed(row.warmth, "warmth", elapsed)),
            joy=_clamp(_decayed(row.joy, "joy", elapsed)),
            concern=_clamp(_decayed(row.concern, "concern", elapsed)),
        )

    async def register_turn(
        self, user_id: uuid.UUID, emotional: EmotionalContext
    ) -> AffectiveState:
        """Aplica os eventos do turno ao estado persistido (decai + deltas).

        Roda na fase 3 do turno (conexão curta, mesmo commit da persistência).
        O lock de linha serializa turnos concorrentes do mesmo usuário no
        PostgreSQL (no SQLite dos testes é no-op — single-writer).
        """
        row = await self._session.scalar(
            select(AffectiveState)
            .where(AffectiveState.user_id == user_id)
            .with_for_update()
        )
        if row is None:
            row = AffectiveState(user_id=user_id)
            self._session.add(row)
            await self._session.flush()

        elapsed = self._elapsed_days(row)
        values = {
            dim: _decayed(getattr(row, dim), dim, elapsed)
            for dim in ("warmth", "joy", "concern")
        }
        values["warmth"] += _TURN_WARMTH_DELTA
        for signal in emotional.signals:
            for dimension, delta in _SIGNAL_DELTAS.get(signal, {}).items():
                values[dimension] += delta

        row.warmth = _clamp(values["warmth"])
        row.joy = _clamp(values["joy"])
        row.concern = _clamp(values["concern"])
        row.updated_at = utcnow()
        METRICS.observe("affect.warmth", row.warmth)
        METRICS.observe("affect.concern", row.concern)
        return row

    @staticmethod
    def _elapsed_days(row: AffectiveState) -> float:
        reference = row.updated_at or row.created_at
        if reference is None:  # objeto transiente (testes de unidade)
            return 0.0
        return max(0.0, (utcnow() - ensure_aware(reference)).total_seconds() / 86_400)
