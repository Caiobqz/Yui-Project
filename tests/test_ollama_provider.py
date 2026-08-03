"""Testes do OllamaProvider.

Usa `httpx.MockTransport` — nenhum servidor Ollama real é necessário. A
suíte padrão nunca depende de uma instância real em execução; testes que
exigem isso ficariam marcados `requires_ollama` (não incluídos aqui por
não haver ambiente disponível para validá-los de verdade nesta sessão).
"""
import json

import httpx
import pytest

from app.core.config import Settings
from app.services.llm.base import ChatMessage, LLMError, ToolSpec
from app.services.llm.ollama_provider import OllamaProvider


def _settings(**overrides) -> Settings:
    return Settings(
        llm_provider="ollama",
        ollama_base_url="http://fake",
        ollama_model="qwen3:8b",
        **overrides,
    )


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://fake")


# --- Construção --------------------------------------------------------------


def test_missing_base_url_raises() -> None:
    settings = Settings(llm_provider="ollama", ollama_base_url="", ollama_model="qwen3:8b")
    with pytest.raises(LLMError, match="OLLAMA_BASE_URL"):
        OllamaProvider(settings)


def test_missing_model_raises() -> None:
    settings = Settings(llm_provider="ollama", ollama_base_url="http://fake", ollama_model="")
    with pytest.raises(LLMError, match="OLLAMA_MODEL"):
        OllamaProvider(settings)


# --- Capabilities (segurança: default conservador) --------------------------


async def test_tool_calling_disabled_by_default() -> None:
    provider = OllamaProvider(_settings(), client=_client(lambda r: httpx.Response(200)))
    assert provider.capabilities().tool_calling is False
    await provider.aclose()


async def test_tool_calling_enabled_only_with_explicit_flag() -> None:
    provider = OllamaProvider(
        _settings(ollama_supports_tool_calling=True),
        client=_client(lambda r: httpx.Response(200)),
    )
    assert provider.capabilities().tool_calling is True
    await provider.aclose()


async def test_tools_rejected_when_capability_disabled() -> None:
    provider = OllamaProvider(_settings(), client=_client(lambda r: httpx.Response(200)))
    tool = ToolSpec(name="x", description="x", input_schema={"type": "object", "properties": {}})
    with pytest.raises(LLMError, match="tool calling"):
        await provider.generate("sys", [ChatMessage(role="user", content="oi")], tools=[tool])
    await provider.aclose()


# --- generate() ---------------------------------------------------------------


async def test_generate_parses_response_and_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "qwen3:8b"
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "oi!"},
                "done": True,
                "prompt_eval_count": 42,
                "eval_count": 13,
            },
        )

    provider = OllamaProvider(_settings(), client=_client(handler))
    response = await provider.generate("sys", [ChatMessage(role="user", content="oi")])
    assert response.content == "oi!"
    assert response.input_tokens == 42
    assert response.output_tokens == 13
    await provider.aclose()


async def test_generate_timeout_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout simulado")

    provider = OllamaProvider(_settings(), client=_client(handler))
    with pytest.raises(LLMError, match="Timeout"):
        await provider.generate("sys", [ChatMessage(role="user", content="oi")])
    await provider.aclose()


async def test_generate_connection_error_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conexão recusada")

    provider = OllamaProvider(_settings(), client=_client(handler))
    with pytest.raises(LLMError, match="conexão"):
        await provider.generate("sys", [ChatMessage(role="user", content="oi")])
    await provider.aclose()


async def test_generate_http_error_status_raises_with_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "modelo travou"})

    provider = OllamaProvider(_settings(), client=_client(handler))
    with pytest.raises(LLMError, match="modelo travou"):
        await provider.generate("sys", [ChatMessage(role="user", content="oi")])
    await provider.aclose()


# --- Streaming -----------------------------------------------------------------


async def test_streaming_accumulates_deltas() -> None:
    lines = [
        {"model": "qwen3:8b", "message": {"content": "Vou"}, "done": False},
        {"model": "qwen3:8b", "message": {"content": " ajudar"}, "done": False},
        {"model": "qwen3:8b", "message": {"content": ""}, "done": True,
         "prompt_eval_count": 10, "eval_count": 4},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        ndjson = "\n".join(json.dumps(line) for line in lines)
        return httpx.Response(200, content=ndjson)

    provider = OllamaProvider(_settings(), client=_client(handler))
    deltas = []
    final = None
    async for chunk in provider.generate_stream("sys", [ChatMessage(role="user", content="oi")]):
        if chunk.delta:
            deltas.append(chunk.delta)
        if chunk.response:
            final = chunk.response
    assert "".join(deltas) == "Vou ajudar"
    assert final is not None
    assert final.input_tokens == 10
    assert final.output_tokens == 4
    await provider.aclose()


async def test_streaming_tool_calls_only_in_final_chunk() -> None:
    """Comportamento real documentado do Ollama: tool_calls não fragmentados,
    só aparecem completos no(s) chunk(s) com conteúdo, incluindo o final."""
    lines = [
        {"model": "qwen3:8b", "message": {"content": ""}, "done": False},
        {
            "model": "qwen3:8b",
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "list_tasks", "arguments": {}}}],
            },
            "done": True,
            "prompt_eval_count": 55,
            "eval_count": 20,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tools" in body
        ndjson = "\n".join(json.dumps(line) for line in lines)
        return httpx.Response(200, content=ndjson)

    provider = OllamaProvider(
        _settings(ollama_supports_tool_calling=True), client=_client(handler)
    )
    tool = ToolSpec(name="list_tasks", description="lista", input_schema={"type": "object", "properties": {}})
    final = None
    async for chunk in provider.generate_stream(
        "sys", [ChatMessage(role="user", content="liste")], tools=[tool]
    ):
        if chunk.response:
            final = chunk.response
    assert final is not None
    assert len(final.tool_calls) == 1
    assert final.tool_calls[0].name == "list_tasks"
    await provider.aclose()


async def test_streaming_mid_stream_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        ndjson = "\n".join([
            json.dumps({"model": "qwen3:8b", "message": {"content": "ol"}, "done": False}),
            json.dumps({"error": "modelo travou"}),
        ])
        return httpx.Response(200, content=ndjson)

    provider = OllamaProvider(_settings(), client=_client(handler))
    with pytest.raises(LLMError, match="modelo travou"):
        async for _ in provider.generate_stream("sys", [ChatMessage(role="user", content="oi")]):
            pass
    await provider.aclose()


async def test_streaming_cancellation_closes_cleanly() -> None:
    """Desconexão do cliente SSE deve encerrar a conexão HTTP sem exceção
    não tratada — o generator assíncrono recebe GeneratorExit em aclose()."""

    async def slow_ndjson():
        for line in [
            {"model": "qwen3:8b", "message": {"content": "um"}, "done": False},
            {"model": "qwen3:8b", "message": {"content": " dois"}, "done": False},
        ]:
            yield (json.dumps(line) + "\n").encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=slow_ndjson())

    provider = OllamaProvider(_settings(), client=_client(handler))
    gen = provider.generate_stream("sys", [ChatMessage(role="user", content="oi")])
    async for chunk in gen:
        if chunk.delta:
            break
    await gen.aclose()  # não deve levantar
    await provider.aclose()


# --- check_availability() / validate() ---------------------------------------


async def test_check_availability_server_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("recusado")

    provider = OllamaProvider(_settings(), client=_client(handler))
    with pytest.raises(LLMError, match="conectar"):
        await provider.check_availability()
    await provider.aclose()


async def test_check_availability_model_missing_lists_alternatives() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})

    provider = OllamaProvider(_settings(), client=_client(handler))
    with pytest.raises(LLMError, match="llama3.2:latest"):
        await provider.check_availability()
    await provider.aclose()


async def test_check_availability_model_present_succeeds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    provider = OllamaProvider(_settings(), client=_client(handler))
    await provider.check_availability()  # não deve levantar
    await provider.aclose()


async def test_validate_delegates_to_check_availability() -> None:
    """`validate()` (contrato base) deve produzir o mesmo resultado que
    `check_availability()` — é apenas a implementação do hook do LLMProvider."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    provider = OllamaProvider(_settings(), client=_client(handler))
    await provider.validate()  # não deve levantar
    await provider.aclose()


# --- aclose() ------------------------------------------------------------------


async def test_aclose_closes_owned_client_only() -> None:
    """Um client injetado pelo chamador não deve ser fechado pelo provider
    (o dono do client é quem o criou)."""
    external_client = _client(lambda r: httpx.Response(200))
    provider = OllamaProvider(_settings(), client=external_client)
    await provider.aclose()
    # O client externo continua utilizável (não foi fechado internamente).
    assert not external_client.is_closed
    await external_client.aclose()
