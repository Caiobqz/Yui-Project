"""Implementação do LLMProvider usando a API da Anthropic (Claude)."""
from collections.abc import AsyncIterator
from typing import Any

from anthropic import APIError, AsyncAnthropic

from app.core.config import Settings
from app.services.llm.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    LLMResponse,
    StreamChunk,
    ToolCall,
    ToolSpec,
)


def _to_anthropic_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Converte o formato neutro para o formato da Anthropic.

    Resultados de ferramenta consecutivos são agrupados numa única mensagem
    'user' com blocos tool_result — exigência da API.
    """
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "tool":
            blocks: list[dict[str, Any]] = []
            while i < len(messages) and messages[i].role == "tool":
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": messages[i].tool_call_id,
                        "content": messages[i].content,
                    }
                )
                i += 1
            out.append({"role": "user", "content": blocks})
            continue
        if m.role == "assistant" and m.tool_calls:
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            for call in m.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            out.append({"role": "assistant", "content": content})
        else:
            out.append({"role": m.role, "content": m.content})
        i += 1
    return out


class ClaudeProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY não configurada.")
        # Timeout explícito: sem ele o SDK espera até 10 minutos.
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self._model = settings.anthropic_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    def _request_kwargs(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "system": system_prompt,
            "messages": _to_anthropic_messages(messages),
        }
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        return kwargs

    @staticmethod
    def _parse(response: Any) -> LLMResponse:
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        tool_calls = tuple(
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        )
        return LLMResponse(
            content=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            tool_calls=tool_calls,
        )

    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        try:
            response = await self._client.messages.create(
                **self._request_kwargs(system_prompt, messages, tools)
            )
        except APIError as exc:
            raise LLMError(f"Erro na API Anthropic: {exc}") from exc
        return self._parse(response)

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        try:
            async with self._client.messages.stream(
                **self._request_kwargs(system_prompt, messages, tools)
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(delta=text)
                final = await stream.get_final_message()
        except APIError as exc:
            raise LLMError(f"Erro na API Anthropic (stream): {exc}") from exc
        yield StreamChunk(response=self._parse(final))
