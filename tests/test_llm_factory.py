"""Testes da factory de LLM (app.services.llm.factory).

Cobre: seleção de backend, preservação da separação principal/utilitário,
reuso de instância quando a configuração é idêntica (evita carregar o
mesmo modelo pesado duas vezes), limpeza de cache para testes, e a
deduplicação de validate()/aclose() quando principal e utilitário
compartilham a mesma instância.

Ollama é usado nos testes de reuso porque `OllamaProvider.__init__` nunca
faz chamada de rede (só monta o cliente httpx) — permite exercitar a
factory real sem precisar de um servidor Ollama disponível. llama.cpp usa
o módulo fake de sempre (sem compilação real).
"""
import sys
import types
from typing import Any

import pytest

import app.services.llm.factory as factory_module
from app.core.config import get_settings
from app.services.llm.base import LLMError, LLMProvider
from app.services.llm.factory import (
    clear_llm_provider_cache,
    close_llm_providers,
    get_llm_provider,
    get_utility_llm_provider,
    validate_llm_providers,
)


def _safe_clear_llm_provider_cache() -> None:
    """Como `clear_llm_provider_cache`, mas tolera testes que substituem
    `get_llm_provider`/`get_utility_llm_provider` por funções sem
    `.cache_clear()` (os testes de dedup de validate/close fazem isso
    deliberadamente, via monkeypatch, para isolar `validate_llm_providers`/
    `close_llm_providers` do cache real). O monkeypatch já reverte essas
    substituições em seu próprio teardown; este helper só evita que o
    teardown DESTE fixture rode antes disso e quebre por atributo ausente.
    """
    for fn in (factory_module.get_llm_provider, factory_module.get_utility_llm_provider):
        cache_clear = getattr(fn, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch: pytest.MonkeyPatch):
    """Cada teste começa com caches limpos e termina sem vazar configuração
    (env vars) nem instâncias cacheadas para o próximo teste."""
    _safe_clear_llm_provider_cache()
    get_settings.cache_clear()
    yield
    _safe_clear_llm_provider_cache()
    get_settings.cache_clear()


def _set_env(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key.upper(), value)
    get_settings.cache_clear()
    clear_llm_provider_cache()


@pytest.fixture
def fake_llama_cpp(monkeypatch: pytest.MonkeyPatch):
    class _FakeLlama:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def create_chat_completion(self, **kwargs: Any) -> Any:
            return {"choices": [{"message": {"content": ""}}], "model": "fake"}

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)


# --- Seleção de backend --------------------------------------------------------


def test_selects_llama_cpp(monkeypatch, fake_llama_cpp, tmp_path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    _set_env(monkeypatch, llm_provider="llama_cpp", llm_model_path=str(gguf))
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    assert isinstance(get_llm_provider(), LlamaCppProvider)


async def test_selects_ollama(monkeypatch) -> None:
    _set_env(monkeypatch, llm_provider="ollama", ollama_base_url="http://fake", ollama_model="qwen3:8b")
    from app.services.llm.ollama_provider import OllamaProvider

    provider = get_llm_provider()
    assert isinstance(provider, OllamaProvider)
    await provider.aclose()


def test_selects_claude(monkeypatch) -> None:
    _set_env(monkeypatch, llm_provider="claude", anthropic_api_key="sk-fake")
    from app.services.llm.claude_provider import ClaudeProvider

    assert isinstance(get_llm_provider(), ClaudeProvider)


def test_selects_openai(monkeypatch) -> None:
    _set_env(monkeypatch, llm_provider="openai", openai_api_key="sk-fake")
    from app.services.llm.openai_provider import OpenAIProvider

    assert isinstance(get_llm_provider(), OpenAIProvider)


def test_unknown_backend_raises(monkeypatch) -> None:
    # Contorna a validação de Settings para forçar um valor inválido direto
    # na factory (settings.llm_provider já teria sido barrado por
    # validate_runtime em uso normal — aqui testamos a factory isolada).
    from app.core.config import Settings
    from app.services.llm.factory import _build_main_provider

    settings = Settings.model_construct(llm_provider="cohere")
    with pytest.raises(LLMError, match="desconhecido"):
        _build_main_provider(settings)


# --- Separação principal/utilitário (preservada) -------------------------------


def test_claude_utility_uses_different_cheaper_model(monkeypatch) -> None:
    _set_env(monkeypatch, llm_provider="claude", anthropic_api_key="sk-fake")
    main = get_llm_provider()
    utility = get_utility_llm_provider()
    assert main is not utility
    assert main._model != utility._model  # type: ignore[attr-defined]
    assert utility._model == get_settings().anthropic_utility_model  # type: ignore[attr-defined]


async def test_ollama_utility_different_model_gives_different_instance(monkeypatch) -> None:
    _set_env(
        monkeypatch,
        llm_provider="ollama",
        ollama_base_url="http://fake",
        ollama_model="qwen3:8b",
        ollama_utility_model="qwen3:4b",
    )
    main = get_llm_provider()
    utility = get_utility_llm_provider()
    assert main is not utility
    assert main._model != utility._model  # type: ignore[attr-defined]
    await main.aclose()
    await utility.aclose()


# --- Reuso de instância (mesma config) -----------------------------------------


async def test_ollama_utility_reuses_main_instance_when_config_identical(monkeypatch) -> None:
    """Sem OLLAMA_UTILITY_MODEL configurado, o utilitário herda o modelo do
    principal — mesma config, mesma instância (evita abrir duas conexões
    idênticas)."""
    _set_env(monkeypatch, llm_provider="ollama", ollama_base_url="http://fake", ollama_model="qwen3:8b")
    main = get_llm_provider()
    utility = get_utility_llm_provider()
    assert main is utility
    await main.aclose()

def test_llama_cpp_utility_reuses_main_instance_when_no_separate_path(
    monkeypatch, fake_llama_cpp, tmp_path
) -> None:
    """Sem UTILITY_LLM_MODEL_PATH configurado, o utilitário aponta para o
    MESMO arquivo GGUF do principal — deve reutilizar a instância já
    carregada em vez de carregar o mesmo modelo pesado duas vezes."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    _set_env(monkeypatch, llm_provider="llama_cpp", llm_model_path=str(gguf))
    main = get_llm_provider()
    utility = get_utility_llm_provider()
    assert main is utility


def test_llama_cpp_utility_different_path_gives_different_instance(
    monkeypatch, fake_llama_cpp, tmp_path
) -> None:
    gguf_main = tmp_path / "main.gguf"
    gguf_main.write_bytes(b"x")
    gguf_utility = tmp_path / "utility.gguf"
    gguf_utility.write_bytes(b"y")
    _set_env(
        monkeypatch,
        llm_provider="llama_cpp",
        llm_model_path=str(gguf_main),
        utility_llm_model_path=str(gguf_utility),
    )
    main = get_llm_provider()
    utility = get_utility_llm_provider()
    assert main is not utility


async def test_mixed_backends_main_llama_cpp_utility_ollama(monkeypatch, fake_llama_cpp, tmp_path) -> None:
    """Backends diferentes entre principal e utilitário: nunca compartilha
    instância (tipos incompatíveis), sem exigir configuração cruzada."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    _set_env(
        monkeypatch,
        llm_provider="llama_cpp",
        llm_model_path=str(gguf),
        utility_llm_provider="ollama",
        ollama_base_url="http://fake",
        ollama_model="qwen3:4b",
    )
    from app.services.llm.llama_cpp_provider import LlamaCppProvider
    from app.services.llm.ollama_provider import OllamaProvider

    main = get_llm_provider()
    utility = get_utility_llm_provider()
    assert isinstance(main, LlamaCppProvider)
    assert isinstance(utility, OllamaProvider)
    assert main is not utility
    await utility.aclose()


# --- clear_llm_provider_cache() ------------------------------------------------


def test_clear_cache_forces_new_instance(monkeypatch) -> None:
    _set_env(monkeypatch, llm_provider="claude", anthropic_api_key="sk-fake")
    first = get_llm_provider()
    clear_llm_provider_cache()
    second = get_llm_provider()
    assert first is not second


def test_without_clear_cache_same_instance_returned(monkeypatch) -> None:
    _set_env(monkeypatch, llm_provider="claude", anthropic_api_key="sk-fake")
    assert get_llm_provider() is get_llm_provider()


# --- validate_llm_providers() / close_llm_providers(): dedup -------------------


class _CountingProvider(LLMProvider):
    def __init__(self) -> None:
        self.validate_calls = 0
        self.close_calls = 0

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def validate(self) -> None:
        self.validate_calls += 1

    async def aclose(self) -> None:
        self.close_calls += 1


async def test_validate_and_close_dedup_when_instance_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    shared = _CountingProvider()
    monkeypatch.setattr(factory_module, "get_llm_provider", lambda: shared)
    monkeypatch.setattr(factory_module, "get_utility_llm_provider", lambda: shared)

    await validate_llm_providers()
    assert shared.validate_calls == 1  # não 2, apesar de ser consultado duas vezes

    await close_llm_providers()
    assert shared.close_calls == 1


async def test_validate_and_close_called_independently_when_different(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = _CountingProvider()
    utility = _CountingProvider()
    monkeypatch.setattr(factory_module, "get_llm_provider", lambda: main)
    monkeypatch.setattr(factory_module, "get_utility_llm_provider", lambda: utility)

    await validate_llm_providers()
    assert main.validate_calls == 1
    assert utility.validate_calls == 1

    await close_llm_providers()
    assert main.close_calls == 1
    assert utility.close_calls == 1


async def test_validate_propagates_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingProvider(_CountingProvider):
        async def validate(self) -> None:
            raise LLMError("servidor indisponível (simulado)")

    failing = _FailingProvider()
    monkeypatch.setattr(factory_module, "get_llm_provider", lambda: failing)
    monkeypatch.setattr(factory_module, "get_utility_llm_provider", lambda: failing)

    with pytest.raises(LLMError, match="indisponível"):
        await validate_llm_providers()
