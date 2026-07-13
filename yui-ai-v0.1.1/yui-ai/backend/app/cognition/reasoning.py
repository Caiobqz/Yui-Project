"""Reasoning Engine — o estado cognitivo do turno.

Formaliza o fluxo:

    Entrada → contexto emocional (heurística)
            → memória relevante (semântica/lexical, com decaimento)
            → modelo do usuário (adaptação + relacionamento)
            → curiosidade (lacunas)
            → estratégia (diretivas de resposta)
            → resposta ou ação (tool calling)

O `CognitiveState` reúne tudo que o turno precisa; `context_service`
transforma o estado em system prompt. O entendimento de intenção continua
delegado ao tool calling do modelo — as ferramentas SÃO as intenções
acionáveis; o estado cognitivo modula COMO responder.
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

    def strategy_directives(self) -> list[str]:
        """Diretivas de estratégia derivadas dos sinais do turno."""
        directives = list(self.emotional.guidance)
        if self.curiosity is not None:
            directives.append(self.curiosity.suggestion)
        return directives
