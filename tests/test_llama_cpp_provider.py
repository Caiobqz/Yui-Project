"""Testes do LlamaCppProvider.

Usa um módulo `llama_cpp` FALSO (via monkeypatch em sys.modules) — a
biblioteca real exige compilação nativa (CMake + C++), então a suíte
padrão nunca a instala nem a importa de verdade. Isso testa exaustivamente
a INTEGRAÇÃO (conversão de formatos, concorrência, streaming, detecção de
capacidades, tratamento de erros) sem validar a biblioteca em si — a
inferência real com um modelo GGUF requer validação manual (ver README).
"""
import sys
import types
from typing import Any

import pytest

from app.core.config import Settings
from app.services.llm.base import ChatMessage, LLMError, ToolSpec


class _FakeLlama:
    """Dublê de `llama_cpp.Llama` com respostas roteirizadas."""

    def __init__(self, script: list[dict[str, Any]] | None = None, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self._script = script or []
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self._script)
        return self._script[0] if self._script else {
            "choices": [{"message": {"content": ""}}],
            "model": "fake",
        }


@pytest.fixture
def fake_llama_cpp_module(monkeypatch: pytest.MonkeyPatch):
    """Injeta um módulo `llama_cpp` falso em sys.modules; devolve a classe
    fake para que o teste configure o roteiro de respostas."""
    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    return _FakeLlama


@pytest.fixture
def gguf_path(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"conteudo falso de gguf")
    return path


def _settings(gguf_path, **overrides) -> Settings:
    return Settings(
        llm_provider="llama_cpp",
        llm_model_path=str(gguf_path),
        **overrides,
    )


# --- Construção e validação de configuração ---------------------------------


def test_missing_model_path_raises(fake_llama_cpp_module) -> None:
    settings = Settings(llm_provider="llama_cpp", llm_model_path=None)
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    with pytest.raises(LLMError, match="LLM_MODEL_PATH"):
        LlamaCppProvider(settings)


def test_missing_gguf_file_raises(fake_llama_cpp_module, tmp_path) -> None:
    settings = Settings(llm_provider="llama_cpp", llm_model_path=str(tmp_path / "nao_existe.gguf"))
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    with pytest.raises(LLMError, match="não encontrado"):
        LlamaCppProvider(settings)


def test_missing_library_raises_clear_error(monkeypatch: pytest.MonkeyPatch, gguf_path) -> None:
    """Sem o pacote llama-cpp-python instalado, o erro deve ser acionável."""
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # simula ImportError
    settings = _settings(gguf_path)
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    with pytest.raises(LLMError, match="llama-cpp-python"):
        LlamaCppProvider(settings)


def test_load_failure_raises_llm_error(monkeypatch: pytest.MonkeyPatch, gguf_path) -> None:
    class _BrokenLlama:
        def __init__(self, **kwargs: Any) -> None:
            raise ValueError("arquivo corrompido")

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _BrokenLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    settings = _settings(gguf_path)
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    with pytest.raises(LLMError, match="Falha ao carregar"):
        LlamaCppProvider(settings)


# --- Capabilities (tool calling) --------------------------------------------


def test_capabilities_plain_chatml_does_not_enable_tools(fake_llama_cpp_module, gguf_path) -> None:
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    settings = _settings(gguf_path, llm_chat_format="chatml")
    provider = LlamaCppProvider(settings)
    assert provider.capabilities().tool_calling is False


def test_capabilities_function_calling_format_enables_tools(fake_llama_cpp_module, gguf_path) -> None:
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    settings = _settings(gguf_path, llm_chat_format="chatml-function-calling")
    provider = LlamaCppProvider(settings)
    assert provider.capabilities().tool_calling is True


def test_capabilities_no_chat_format_defaults_to_no_tools(fake_llama_cpp_module, gguf_path) -> None:
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    provider = LlamaCppProvider(_settings(gguf_path))
    assert provider.capabilities().tool_calling is False


def test_capabilities_explicit_override_wins(fake_llama_cpp_module, gguf_path) -> None:
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    settings = _settings(gguf_path, llm_chat_format="chatml", llm_supports_tool_calling=True)
    provider = LlamaCppProvider(settings)
    assert provider.capabilities().tool_calling is True


async def test_tools_rejected_when_unsupported(fake_llama_cpp_module, gguf_path) -> None:
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    provider = LlamaCppProvider(_settings(gguf_path))
    tool = ToolSpec(name="x", description="x", input_schema={"type": "object", "properties": {}})
    with pytest.raises(LLMError, match="tool calling"):
        await provider.generate("sys", [ChatMessage(role="user", content="oi")], tools=[tool])


# --- Concorrência (correção de segurança) -----------------------------------


def test_concurrency_always_clamped_to_one(fake_llama_cpp_module, gguf_path, caplog) -> None:
    """Requisito de segurança: nunca > 1, mesmo se configurado maior —
    esta implementação não isola contexto por requisição."""
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    with caplog.at_level("WARNING"):
        provider = LlamaCppProvider(_settings(gguf_path, llm_max_concurrent_requests=8))
    assert provider._inference_semaphore._value == 1
    assert any("forçando 1" in r.message for r in caplog.records)


def test_concurrency_default_is_one_without_warning(fake_llama_cpp_module, gguf_path, caplog) -> None:
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    with caplog.at_level("WARNING"):
        provider = LlamaCppProvider(_settings(gguf_path, llm_max_concurrent_requests=1))
    assert provider._inference_semaphore._value == 1
    assert not any("forçando 1" in r.message for r in caplog.records)


# --- generate() --------------------------------------------------------------


async def test_generate_returns_parsed_response(fake_llama_cpp_module, gguf_path, monkeypatch) -> None:
    from app.services.llm import llama_cpp_provider as mod

    settings = _settings(gguf_path)

    # Configura o script de resposta na própria classe fake antes de construir.
    script = [
        {
            "choices": [{"message": {"content": "Olá!"}}],
            "model": "qwen-test",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
    ]

    class _ScriptedLlama(_FakeLlama):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(script=script, **kwargs)

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _ScriptedLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    provider = mod.LlamaCppProvider(settings)
    response = await provider.generate("sys", [ChatMessage(role="user", content="oi")])
    assert response.content == "Olá!"
    assert response.model == "qwen-test"
    assert response.input_tokens == 5
    assert response.output_tokens == 3


# --- Streaming ----------------------------------------------------------------


async def test_streaming_accumulates_deltas_and_tool_calls(monkeypatch, gguf_path) -> None:
    from app.services.llm import llama_cpp_provider as mod

    chunks = [
        {"choices": [{"delta": {"content": "Vou"}}], "model": "qwen-test"},
        {"choices": [{"delta": {"content": " verificar"}}], "model": "qwen-test"},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "list_tasks", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ],
            "model": "qwen-test",
        },
    ]

    class _StreamingLlama(_FakeLlama):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(script=chunks, **kwargs)

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _StreamingLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    settings = _settings(gguf_path, llm_chat_format="chatml-function-calling")
    provider = mod.LlamaCppProvider(settings)

    deltas = []
    final = None
    tool = ToolSpec(name="list_tasks", description="lista", input_schema={"type": "object", "properties": {}})
    async for chunk in provider.generate_stream(
        "sys", [ChatMessage(role="user", content="liste")], tools=[tool]
    ):
        if chunk.delta:
            deltas.append(chunk.delta)
        if chunk.response:
            final = chunk.response

    assert "".join(deltas) == "Vou verificar"
    assert final is not None
    assert len(final.tool_calls) == 1
    assert final.tool_calls[0].name == "list_tasks"
    assert final.tool_calls[0].arguments == {}


async def test_streaming_error_raises_llm_error(monkeypatch, gguf_path) -> None:
    from app.services.llm import llama_cpp_provider as mod

    class _BrokenStreamLlama(_FakeLlama):
        def create_chat_completion(self, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                raise RuntimeError("falha simulada de inferência")
            return super().create_chat_completion(**kwargs)

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = _BrokenStreamLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    provider = mod.LlamaCppProvider(_settings(gguf_path))
    with pytest.raises(LLMError, match="streaming"):
        async for _ in provider.generate_stream("sys", [ChatMessage(role="user", content="oi")]):
            pass
