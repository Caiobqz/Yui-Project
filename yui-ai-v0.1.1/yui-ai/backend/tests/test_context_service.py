"""Testes da montagem de contexto."""
from app.core.personality import Personality
from app.models.memory import MemoryEntry
from app.services.context_service import build_system_prompt


def _personality() -> Personality:
    return Personality(
        name="Yui",
        role="assistente",
        objective="Ajudar.",
        communication_style=[],
        behavior_rules=[],
        limitations=[],
    )


def test_prompt_without_memories_warns_model() -> None:
    prompt = build_system_prompt(_personality(), [])
    assert "Ainda não há memórias" in prompt


def test_prompt_includes_memories() -> None:
    memory = MemoryEntry(
        user_id="u1", category="estudos", content="Está aprendendo Python", relevance=0.8
    )
    prompt = build_system_prompt(_personality(), [memory])
    assert "[estudos] Está aprendendo Python" in prompt
