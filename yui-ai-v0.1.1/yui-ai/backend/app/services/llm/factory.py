"""Seleção do provedor de IA a partir da configuração (LLM_PROVIDER)."""
from functools import lru_cache

from app.core.config import get_settings
from app.services.llm.base import LLMError, LLMProvider


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    provider = settings.llm_provider.lower()

    # Imports tardios intencionais: evitam carregar o SDK de um provedor
    # que não está em uso (e permitem instalar apenas o necessário no futuro).
    if provider == "claude":
        from app.services.llm.claude_provider import ClaudeProvider

        return ClaudeProvider(settings)
    if provider == "openai":
        from app.services.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)

    raise LLMError(
        f"Provedor de IA desconhecido: '{settings.llm_provider}'. "
        "Valores aceitos: 'claude', 'openai'."
    )
