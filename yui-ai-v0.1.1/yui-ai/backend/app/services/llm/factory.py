"""Seleção do provedor de IA a partir da configuração (LLM_PROVIDER).

Dois provedores por processo:
- principal (`get_llm_provider`) — conversa com o usuário;
- utilitário (`get_utility_llm_provider`) — trabalho cognitivo de bastidor
  (extração de memórias, adaptação, resumo, revisão de planos) num modelo
  mais barato, para o núcleo cognitivo não dobrar o custo por turno.
"""
from functools import lru_cache

from app.core.config import get_settings
from app.services.llm.base import LLMError, LLMProvider


def _build_provider(model_override: str | None) -> LLMProvider:
    settings = get_settings()
    provider = settings.llm_provider.lower()

    # Imports tardios intencionais: evitam carregar o SDK de um provedor
    # que não está em uso.
    if provider == "claude":
        from app.services.llm.claude_provider import ClaudeProvider

        return ClaudeProvider(settings, model=model_override)
    if provider == "openai":
        from app.services.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings, model=model_override)

    raise LLMError(
        f"Provedor de IA desconhecido: '{settings.llm_provider}'. "
        "Valores aceitos: 'claude', 'openai'."
    )


@lru_cache
def get_llm_provider() -> LLMProvider:
    return _build_provider(None)


@lru_cache
def get_utility_llm_provider() -> LLMProvider:
    settings = get_settings()
    model = (
        settings.anthropic_utility_model
        if settings.llm_provider.lower() == "claude"
        else settings.openai_utility_model
    )
    return _build_provider(model)
