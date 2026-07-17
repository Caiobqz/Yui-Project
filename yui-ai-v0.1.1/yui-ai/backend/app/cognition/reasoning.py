"""Reasoning Engine — o estado cognitivo do turno.

Formaliza o fluxo:

    Entrada → contexto emocional (heurística)
            → atenção sobre memórias (relevância, objetivos, redundância)
            → objetivos ativos (Goal Engine)
            → modelo do usuário (adaptação + relacionamento)
            → self / world model
            → curiosidade (lacunas)
            → estratégia (diretivas de resposta)
            → resposta ou ação (tool calling)

O `CognitiveState` reúne tudo que o turno precisa; o Context Orchestrator o
monta e o `context_service` (prompt builder de baixo nível) o transforma em
system prompt. O entendimento de intenção continua delegado ao tool calling.
"""
from dataclasses import dataclass, field

from app.cognition.curiosity import CuriosityHint
from app.cognition.emotional_context import EmotionalContext
from app.models.memory import MemoryEntry


@dataclass(frozen=True)
class CognitiveState:
    """Tudo que o Reasoning Engine reuniu para um turno."""

    memories: list[MemoryEntry] = field(default_factory=list)
    summary: str | None = None
    adaptation_notes: list[str] = field(default_factory=list)
    relationship: str | None = None
    emotional: EmotionalContext = field(
        default_factory=lambda: EmotionalContext(signals=(), guidance=())
    )
    curiosity: CuriosityHint | None = None
    # v0.4 — domínios do World Model e objetivos ativos.
    self_prompt: str | None = None
    environment_prompt: str | None = None
    general_boundary: str | None = None
    goals: list[str] = field(default_factory=list)
    # v0.5 — estado afetivo persistente e iniciativa própria pendente.
    affect_prompt: str | None = None
    initiative: str | None = None

    def strategy_directives(self) -> list[str]:
        """Diretivas de estratégia derivadas dos sinais do turno."""
        directives = list(self.emotional.guidance)
        if self.curiosity is not None:
            directives.append(self.curiosity.suggestion)
        if self.initiative:
            directives.append(
                "Iniciativa própria (já aprovada pelo seu julgamento): "
                + self.initiative
                + " Traga o assunto com naturalidade SOMENTE se houver um "
                "momento oportuno nesta conversa; nunca insista nem "
                "atropele o que o usuário precisa agora."
            )
        return directives
