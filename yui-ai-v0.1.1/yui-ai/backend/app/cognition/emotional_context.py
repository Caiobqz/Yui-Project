"""Emotional Context Model — análise heurística do contexto emocional.

NÃO cria emoções: detecta sinais no texto do usuário (frustração,
dificuldade, pressa, motivação) por padrões determinísticos — barato,
previsível e testável — e produz diretivas que modulam tom, complexidade e
forma de explicação da resposta. É um modelo computacional de interação.
"""
import re
from dataclasses import dataclass

_PATTERNS: dict[str, tuple[list[re.Pattern[str]], str]] = {
    "frustracao": (
        [
            re.compile(r"(?i)\bn[ãa]o (funciona|est[áa] funcionando|consigo)\b"),
            re.compile(r"(?i)\b(de novo|dando erro|que saco|droga|odeio|desisto)\b"),
            re.compile(r"!{2,}"),
        ],
        "O usuário parece frustrado: valide a dificuldade em uma frase, seja "
        "direto na solução e evite humor e rodeios.",
    ),
    "dificuldade": (
        [
            re.compile(r"(?i)\bn[ãa]o (entendo|entendi|sei como|fa[çc]o ideia)\b"),
            re.compile(r"(?i)\b(confuso|confusa|perdido|perdida|travado|travada|complicado demais)\b"),
        ],
        "O usuário demonstra dificuldade: explique passo a passo, com um "
        "exemplo prático, e evite jargões.",
    ),
    "pressa": (
        [
            re.compile(r"(?i)\b(urgente|r[áa]pido|rapidinho|agora|j[áa]|sem tempo|correndo)\b"),
        ],
        "O usuário está com pressa: responda o essencial primeiro, em poucas "
        "linhas, e ofereça detalhes só se ele pedir.",
    ),
    "motivacao": (
        [
            re.compile(r"(?i)\b(consegui|finalmente|deu certo|funcionou|animado|animada|empolgado|empolgada)\b"),
        ],
        "O usuário está motivado: reconheça o progresso brevemente e sugira "
        "um próximo passo concreto.",
    ),
}


@dataclass(frozen=True)
class EmotionalContext:
    signals: tuple[str, ...]
    guidance: tuple[str, ...]

    @property
    def is_neutral(self) -> bool:
        return not self.signals


def analyze(text: str) -> EmotionalContext:
    signals: list[str] = []
    guidance: list[str] = []
    for signal, (patterns, directive) in _PATTERNS.items():
        if any(p.search(text) for p in patterns):
            signals.append(signal)
            guidance.append(directive)
    return EmotionalContext(signals=tuple(signals), guidance=tuple(guidance))
