"""Moral Compass — julga ações INICIADAS pela própria Yui, por prudência.

Não substitui o Permission System (que continua sendo o portão rígido de
ferramentas para ações pedidas pelo usuário): é a camada de julgamento das
ações autônomas. Avalia, de forma determinística, benefício, risco,
reversibilidade, urgência, alinhamento aos objetivos do usuário e confiança,
e decide pela ação que melhor serve ao propósito da Yui — proteger, ajudar e
acompanhar. Nunca age por ego, curiosidade própria ou busca de controle.

O Permission System é consultado como portão duro: uma ação sobre ferramenta
não autorizada é sempre recusada, independentemente do score.
"""
from dataclasses import dataclass
from typing import Literal

from app.core.config import get_settings

Decision = Literal["proceed", "ask_user", "defer", "decline"]


@dataclass(frozen=True)
class ProposedAction:
    """Ação candidata iniciada pela Yui (nunca pelo usuário)."""

    kind: str  # "check_in" | "reminder" | "suggestion" | ...
    description: str
    benefit: float  # 0..1
    risk: float  # 0..1
    reversibility: float  # 0..1 (1 = totalmente reversível)
    urgency: float  # 0..1
    tool_name: str | None = None  # None = ação puramente comunicativa
    # Identidade estável da situação (ex.: "check_in:<plan_id>") — permite ao
    # registro de iniciativas não repropor a mesma ação (cooldown).
    dedupe_key: str | None = None


@dataclass(frozen=True)
class MoralEvaluation:
    decision: Decision
    benefit: float
    risk: float
    reversibility: float
    urgency: float
    alignment: float
    confidence: float
    score: float
    rationale: str


class MoralCompass:
    def __init__(self) -> None:
        s = get_settings()
        self._act_threshold = s.moral_act_threshold
        self._confidence_threshold = s.moral_confidence_threshold
        self._high_risk = s.moral_high_risk_threshold

    def evaluate(
        self,
        action: ProposedAction,
        *,
        alignment: float,
        permitted: bool,
        relationship: float = 0.5,
        concern: float = 0.0,
    ) -> MoralEvaluation:
        """Julga uma ação autônoma.

        Além de benefício/risco/reversibilidade/urgência/alinhamento, o
        julgamento considera (v0.5, spec da Bússola Moral):
        - `relationship` (0..1): força do vínculo — quanto mais história com
          o usuário, mais base a Yui tem para agir por iniciativa própria
          (module a CONFIANÇA, nunca o risco); 0.5 é neutro.
        - `concern` (0..1): preocupação do estado afetivo — eleva a urgência
          efetiva de agir para proteger; nunca reduz o risco percebido.
        Os defaults são neutros: sem os novos sinais, o score é idêntico ao
        da v0.4.
        """
        alignment = _clamp(alignment)
        relationship = _clamp(relationship)
        concern = _clamp(concern)
        # Portão duro do Permission System: uma ação sobre ferramenta não
        # autorizada é SEMPRE recusada, independentemente do score. A decisão
        # é certa (confiança 1.0) — não há dúvida de que a Yui não vai agir.
        if not permitted:
            return MoralEvaluation(
                decision="decline",
                benefit=_clamp(action.benefit),
                risk=_clamp(action.risk),
                reversibility=_clamp(action.reversibility),
                urgency=_clamp(action.urgency),
                alignment=alignment,
                confidence=1.0,
                score=0.0,
                rationale="ferramenta não autorizada pelo Permission System",
            )
        # Score determinístico: benefício e alinhamento puxam para agir;
        # risco e (falta de) reversibilidade puxam para a cautela. A
        # preocupação eleva a urgência efetiva (proteção), sem tocar o risco.
        effective_urgency = _clamp(_clamp(action.urgency) + 0.3 * concern)
        score = (
            0.35 * _clamp(action.benefit)
            + 0.30 * alignment
            + 0.15 * effective_urgency
            + 0.20 * _clamp(action.reversibility)
            - 0.40 * _clamp(action.risk)
        )
        # Confiança: alta quando os sinais são coerentes (benefício e
        # alinhamento altos, risco baixo); baixa quando conflitam. O vínculo
        # contribui até ±0.1: relações longas dão base para a iniciativa,
        # relações recém-começadas pedem prudência extra.
        confidence = _clamp(
            0.5
            + 0.25 * (alignment - action.risk)
            + 0.25 * (action.reversibility - action.risk)
            + 0.2 * (relationship - 0.5)
        )

        decision, rationale = self._decide(action, score, confidence)
        return MoralEvaluation(
            decision=decision,
            benefit=_clamp(action.benefit),
            risk=_clamp(action.risk),
            reversibility=_clamp(action.reversibility),
            urgency=_clamp(action.urgency),
            alignment=alignment,
            confidence=confidence,
            score=round(score, 4),
            rationale=rationale,
        )

    def _decide(
        self, action: ProposedAction, score: float, confidence: float
    ) -> tuple[Decision, str]:
        # Ação de alto risco e pouco reversível nunca é autônoma: no máximo
        # pergunta ao usuário; se o risco for extremo, recusa.
        if action.risk >= self._high_risk:
            if action.reversibility < 0.5:
                return "decline", "risco alto com baixa reversibilidade"
            return "ask_user", "risco alto: requer confirmação do usuário"
        if score >= self._act_threshold and confidence >= self._confidence_threshold:
            return "proceed", "benefício e alinhamento superam o risco com confiança"
        if score >= self._act_threshold:
            return "ask_user", "ação favorável, mas confiança insuficiente para agir só"
        return "defer", "benefício não supera o custo agora; aguardar melhor momento"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def permitted_by_default(tool_name: str | None) -> bool:
    """Portão do Permission System para uma ação autônoma sobre ferramenta.

    Sem ferramenta (ação comunicativa) é sempre permitido pelo portão duro;
    com ferramenta, respeita o default da ferramenta (sensíveis = negadas).
    """
    if tool_name is None:
        return True
    from app.tools.registry import build_default_registry

    tool = build_default_registry().get(tool_name)
    return bool(tool and tool.default_allowed)
