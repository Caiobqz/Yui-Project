"""Judgement Engine — o fluxo de julgamento de toda ação autônoma da Yui.

    Percepção → Compreensão → Contexto → Memórias → Objetivos →
    Relationship Model → Bússola Moral → Avaliação de Risco → Confiança →
    Decisão → Execução → Aprendizado

Este módulo implementa a espinha determinística: dado o contexto do usuário
(objetivos ativos e estado do relacionamento) e ações candidatas geradas pelo
Goal Engine, delibera com a Bússola Moral e decide o que — se algo — a Yui
deve iniciar. A entrega efetiva (mensagem proativa) depende de um canal que
chega em versões futuras (voz/push); aqui a decisão e o registro já existem.
"""
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.cognition.attention import terms_of
from app.cognition.goal_engine import GoalEngine, GoalState
from app.cognition.moral_compass import (
    MoralCompass,
    MoralEvaluation,
    ProposedAction,
    permitted_by_default,
)
from app.core.config import get_settings
from app.core.metrics import METRICS


@dataclass(frozen=True)
class Initiative:
    """Uma ação autônoma que passou no julgamento."""

    kind: str
    description: str
    evaluation: MoralEvaluation


class JudgementEngine:
    def __init__(
        self,
        compass: MoralCompass | None = None,
        goal_engine: GoalEngine | None = None,
    ) -> None:
        self._compass = compass or MoralCompass()
        self._goals = goal_engine or GoalEngine()

    def deliberate(
        self, action: ProposedAction, *, alignment: float
    ) -> MoralEvaluation:
        """Delibera sobre UMA ação candidata e registra a decisão."""
        evaluation = self._compass.evaluate(
            action,
            alignment=alignment,
            permitted=permitted_by_default(action.tool_name),
        )
        METRICS.incr(f"judgement.decision.{evaluation.decision}")
        METRICS.observe("judgement.confidence", evaluation.confidence)
        if evaluation.decision != "proceed":
            METRICS.incr("moral.intervention")
        return evaluation

    async def propose_initiatives(
        self, session: AsyncSession, user_id: uuid.UUID
    ) -> list[Initiative]:
        """Gera ações candidatas a partir dos objetivos e julga cada uma.

        Retorna apenas as que a Bússola Moral aprovou para ação autônoma
        (decision == "proceed"). Determinístico de ponta a ponta.
        """
        settings = get_settings()
        if not settings.autonomy_enabled:
            return []

        states = await self._goals.analyze(session, user_id)
        goal_terms = _all_goal_terms(states)
        approved: list[Initiative] = []
        for action in _candidate_actions(states):
            alignment = _alignment(action, states, goal_terms)
            evaluation = self.deliberate(action, alignment=alignment)
            if evaluation.decision == "proceed":
                approved.append(
                    Initiative(
                        kind=action.kind,
                        description=action.description,
                        evaluation=evaluation,
                    )
                )
        if approved:
            METRICS.incr("autonomous.initiatives", by=len(approved))
        return approved


def _candidate_actions(states: list[GoalState]) -> list[ProposedAction]:
    """Traduz situações de objetivo em ações candidatas (comunicativas)."""
    actions: list[ProposedAction] = []
    for state in states:
        if state.status == "abandoned":
            actions.append(
                ProposedAction(
                    kind="check_in",
                    description=(
                        f"Retomar o objetivo '{state.title}', parado há "
                        f"{int(state.idle_days)} dias na etapa '{state.next_step}'."
                    ),
                    benefit=0.8,
                    risk=0.1,
                    reversibility=1.0,
                    urgency=0.5,
                )
            )
        elif state.status == "active" and state.total and state.done == state.total - 1:
            # Oportunidade: falta uma etapa para concluir.
            actions.append(
                ProposedAction(
                    kind="encouragement",
                    description=(
                        f"Incentivar a conclusão de '{state.title}': falta apenas "
                        f"a etapa '{state.next_step}'."
                    ),
                    benefit=0.7,
                    risk=0.05,
                    reversibility=1.0,
                    urgency=0.4,
                )
            )
    return actions


def _all_goal_terms(states: list[GoalState]) -> set[str]:
    terms: set[str] = set()
    for state in states:
        terms |= terms_of(state.title)
    return terms


def _alignment(
    action: ProposedAction, states: list[GoalState], goal_terms: set[str]
) -> float:
    """Alinhamento: quanto a ação toca objetivos ativos do usuário."""
    action_terms = terms_of(action.description)
    if not goal_terms:
        return 0.0
    overlap = action_terms & goal_terms
    return min(1.0, len(overlap) / 3.0) if overlap else 0.2
