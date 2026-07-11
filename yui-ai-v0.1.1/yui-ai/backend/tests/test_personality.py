"""Testes do sistema de personalidade."""
from app.core.personality import Personality


def _sample() -> Personality:
    return Personality(
        name="Yui",
        role="assistente pessoal de IA",
        objective="Ajudar o usuário.",
        communication_style=["Tom amigável."],
        behavior_rules=["Não inventar informações."],
        limitations=["Não possui consciência real."],
    )


def test_system_prompt_contains_all_sections() -> None:
    prompt = _sample().to_system_prompt()
    assert "Você é Yui" in prompt
    assert "Objetivo:" in prompt
    assert "Estilo de comunicação:" in prompt
    assert "Regras de comportamento:" in prompt
    assert "Limitações:" in prompt
    assert "- Tom amigável." in prompt
