"""Implementação do LLMProvider usando a API da OpenAI."""
import json
from collections.abc import AsyncGenerator
from typing import Any

from openai import APIError, AsyncOpenAI

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


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _to_openai_messages(
    system_prompt: str, messages: list[ChatMessage]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in m.tool_calls
                    ],
                }
            )
        elif m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


class OpenAIProvider(LLMProvider):
    def __init__(self, settings: Settings, model: str | None = None) -> None:
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY não configurada.")
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self._model = model or settings.openai_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    def _request_kwargs(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None,
    ) -> dict[str, Any]:
        # Nota: modelos de raciocínio recentes da OpenAI usam
        # `max_completion_tokens`; ao migrar de modelo, revisar aqui.
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": _to_openai_messages(system_prompt, messages),
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
        return kwargs

    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        try:
            response = await self._client.chat.completions.create(
                **self._request_kwargs(system_prompt, messages, tools)
            )
        except APIError as exc:
            raise LLMError(f"Erro na API OpenAI: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        tool_calls = tuple(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=_parse_arguments(tc.function.arguments),
            )
            for tc in (choice.message.tool_calls or [])
            # Ignora variantes não-função (custom tool calls da OpenAI).
            if tc.type == "function"
        )
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            tool_calls=tool_calls,
        )

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        try:
            stream = await self._client.chat.completions.create(
                stream=True,
                stream_options={"include_usage": True},
                **self._request_kwargs(system_prompt, messages, tools),
            )
            content_parts: list[str] = []
            tool_acc: dict[int, dict[str, str]] = {}
            model = self._model
            input_tokens: int | None = None
            output_tokens: int | None = None
            async for chunk in stream:
                if chunk.usage is not None:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
                if chunk.model:
                    model = chunk.model
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                    yield StreamChunk(delta=delta.content)
                for tc in delta.tool_calls or []:
                    acc = tool_acc.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function and tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        acc["arguments"] += tc.function.arguments
        except APIError as exc:
            raise LLMError(f"Erro na API OpenAI (stream): {exc}") from exc

        tool_calls = tuple(
            ToolCall(
                id=acc["id"],
                name=acc["name"],
                arguments=_parse_arguments(acc["arguments"]),
            )
            for _, acc in sorted(tool_acc.items())
        )
        yield StreamChunk(
            response=LLMResponse(
                content="".join(content_parts),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
            )
        )
