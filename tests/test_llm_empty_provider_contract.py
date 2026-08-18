"""Providers normalizam vazio; YuiCore decide retry/fallback centralmente."""
from types import SimpleNamespace

from app.services.llm.claude_provider import ClaudeProvider
from app.services.llm.llama_cpp_provider import _parse_response as parse_llama
from app.services.llm.ollama_provider import _parse_response as parse_ollama
from app.services.llm.openai_provider import OpenAIProvider


def test_ollama_exposes_empty_final_to_central_validator() -> None:
    response = parse_ollama({"message": {"content": None}, "model": "ollama-fake"})
    assert response.content == ""
    assert response.tool_calls == ()


def test_llama_cpp_exposes_empty_final_to_central_validator() -> None:
    response = parse_llama(
        {"choices": [{"message": {"content": None}}], "model": "llama-fake"}
    )
    assert response.content == ""
    assert response.tool_calls == ()


def test_claude_exposes_empty_final_to_central_validator() -> None:
    raw = SimpleNamespace(
        content=[],
        model="claude-fake",
        usage=SimpleNamespace(input_tokens=1, output_tokens=0),
    )
    response = ClaudeProvider._parse(raw)
    assert response.content == ""
    assert response.tool_calls == ()


async def test_openai_exposes_empty_final_to_central_validator() -> None:
    class _Completions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=None)
                    )
                ],
                model="openai-fake",
                usage=None,
            )

    provider = object.__new__(OpenAIProvider)
    provider._client = SimpleNamespace(  # type: ignore[attr-defined]
        chat=SimpleNamespace(completions=_Completions())
    )
    provider._model = "openai-fake"  # type: ignore[attr-defined]
    provider._max_tokens = 32  # type: ignore[attr-defined]
    provider._temperature = 0.0  # type: ignore[attr-defined]

    response = await provider.generate("system", [])

    assert response.content == ""
    assert response.tool_calls == ()
