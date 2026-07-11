"""Implementação do LLMProvider usando a API da Anthropic (Claude)."""
from anthropic import APIError, AsyncAnthropic

from app.core.config import Settings
from app.services.llm.base import ChatMessage, LLMError, LLMProvider, LLMResponse


class ClaudeProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY não configurada.")
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    async def generate(
        self, system_prompt: str, messages: list[ChatMessage]
    ) -> LLMResponse:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
        except APIError as exc:
            raise LLMError(f"Erro na API Anthropic: {exc}") from exc

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return LLMResponse(
            content=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
