"""Montagem do contexto enviado ao modelo: personalidade + memórias."""
from app.core.personality import Personality
from app.models.memory import MemoryEntry


def build_system_prompt(
    personality: Personality, memories: list[MemoryEntry]
) -> str:
    prompt = personality.to_system_prompt()

    if memories:
        lines = "\n".join(
            f"- [{m.category}] {m.content}" for m in memories
        )
        prompt += (
            "\n\nMemórias sobre o usuário (informações autorizadas, "
            "use quando forem relevantes para a conversa):\n" + lines
        )
    else:
        prompt += (
            "\n\nAinda não há memórias salvas sobre este usuário. "
            "Não presuma informações pessoais."
        )
    return prompt
