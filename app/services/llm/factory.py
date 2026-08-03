"""Seleção do provedor de IA a partir da configuração (LLM_PROVIDER).

Dois provedores por processo:
- principal (`get_llm_provider`) — conversa com o usuário;
- utilitário (`get_utility_llm_provider`) — trabalho cognitivo de bastidor
  (extração de memórias, adaptação, resumo, revisão de planos) num modelo
  mais barato, para o núcleo cognitivo não dobrar o custo por turno.

Backends comerciais (Claude, OpenAI) sempre usam modelos DIFERENTES para
principal/utilitário (o utilitário é mais barato) — nenhuma reutilização de
instância faz sentido aí. Backends locais (llama.cpp, Ollama) podem apontar
para o MESMO arquivo/modelo quando o operador não configura um utilitário
separado; nesse caso a mesma instância é reutilizada, evitando carregar o
mesmo modelo pesado duas vezes na RAM/VRAM.
"""
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.llm.base import LLMError, LLMProvider


def _build_main_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower()

    # Imports tardios intencionais: evitam carregar o SDK/lib de um backend
    # que não está em uso (SDKs comerciais nunca são obrigatórios; llama.cpp
    # não é importado quando o backend selecionado é outro).
    if provider == "claude":
        from app.services.llm.claude_provider import ClaudeProvider

        return ClaudeProvider(settings, model=None)
    if provider == "openai":
        from app.services.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings, model=None)
    if provider == "llama_cpp":
        from app.services.llm.llama_cpp_provider import LlamaCppProvider

        return LlamaCppProvider(settings, model_path=settings.resolved_llm_model_path)
    if provider == "ollama":
        from app.services.llm.ollama_provider import OllamaProvider

        return OllamaProvider(settings, model=settings.ollama_model)

    raise LLMError(
        f"Provedor de IA desconhecido: '{settings.llm_provider}'. "
        "Valores aceitos: 'claude', 'openai', 'llama_cpp', 'ollama'."
    )


def _build_utility_provider(settings: Settings) -> LLMProvider:
    provider = settings.effective_utility_llm_provider

    if provider == "claude":
        from app.services.llm.claude_provider import ClaudeProvider

        return ClaudeProvider(settings, model=settings.anthropic_utility_model)
    if provider == "openai":
        from app.services.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings, model=settings.openai_utility_model)
    if provider == "llama_cpp":
        from app.services.llm.llama_cpp_provider import LlamaCppProvider

        model_path = settings.resolved_utility_llm_model_path or settings.resolved_llm_model_path
        return LlamaCppProvider(settings, model_path=model_path)
    if provider == "ollama":
        from app.services.llm.ollama_provider import OllamaProvider

        model = settings.ollama_utility_model or settings.ollama_model
        return OllamaProvider(settings, model=model)

    raise LLMError(
        f"Provedor de IA utilitário desconhecido: '{provider}'. "
        "Valores aceitos: 'claude', 'openai', 'llama_cpp', 'ollama'."
    )


def _utility_shares_main_config(settings: Settings) -> bool:
    """True quando o utilitário aponta para exatamente o mesmo backend E
    modelo/arquivo que o principal — caso em que a mesma instância deve ser
    reutilizada em vez de reconstruída.

    Só relevante para llama.cpp e Ollama (backends locais e pesados).
    Claude/OpenAI sempre usam `*_utility_model` diferente do principal por
    design (modelo mais barato), então nunca compartilham instância.
    """
    main = settings.llm_provider.lower()
    utility = settings.effective_utility_llm_provider
    if main != utility:
        return False
    if main == "llama_cpp":
        utility_path = settings.resolved_utility_llm_model_path or settings.resolved_llm_model_path
        return utility_path == settings.resolved_llm_model_path
    if main == "ollama":
        utility_model = settings.ollama_utility_model or settings.ollama_model
        return utility_model == settings.ollama_model
    return False


@lru_cache
def get_llm_provider() -> LLMProvider:
    return _build_main_provider(get_settings())


@lru_cache
def get_utility_llm_provider() -> LLMProvider:
    settings = get_settings()
    if _utility_shares_main_config(settings):
        # Mesmo backend e mesmo modelo/arquivo do principal: reutiliza a
        # instância já carregada em vez de duplicar o consumo de RAM/VRAM
        # (crítico para llama.cpp) ou abrir uma segunda conexão idêntica
        # (Ollama). Preserva o comportamento anterior quando o operador não
        # configura um utilitário separado.
        return get_llm_provider()
    return _build_utility_provider(settings)


def clear_llm_provider_cache() -> None:
    """Limpa o cache das factories de LLM (principal e utilitário).

    Uso exclusivo em testes — permite reconstruir providers com uma nova
    configuração dentro do mesmo processo (`get_settings.cache_clear()`
    também deve ser chamado quando a configuração mudou).
    """
    get_llm_provider.cache_clear()
    get_utility_llm_provider.cache_clear()


async def validate_llm_providers() -> None:
    """Valida disponibilidade dos provedores de LLM configurados (boot).

    Chamado no lifespan após a construção síncrona dos providers (que já
    falha cedo para config/arquivo GGUF ausente). Cobre validações que só
    podem ser assíncronas (ex.: conectividade com um servidor Ollama).
    Evita validar duas vezes quando principal e utilitário compartilham a
    mesma instância.
    """
    seen: set[int] = set()
    for provider in (get_llm_provider(), get_utility_llm_provider()):
        if id(provider) in seen:
            continue
        seen.add(id(provider))
        await provider.validate()


async def close_llm_providers() -> None:
    """Fecha os provedores de LLM (principal e utilitário) no shutdown.

    Evita fechar duas vezes quando a mesma instância é compartilhada entre
    os dois papéis (ver `get_utility_llm_provider`).
    """
    seen: set[int] = set()
    for provider in (get_llm_provider(), get_utility_llm_provider()):
        if id(provider) in seen:
            continue
        seen.add(id(provider))
        await provider.aclose()
