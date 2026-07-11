"""Implementação do LLMProvider usando a API da OpenAI."""
from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.core.config import Settings
from app.services.llm.base import ChatMessage, LLMError, LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY não configurada.")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    async def generate(
        self, system_prompt: str, messages: list[ChatMessage]
    ) -> LLMResponse:
        chat_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        for m in messages:
            if m.role == "user":
                chat_messages.append({"role": "user", "content": m.content})
            else:
                chat_messages.append({"role": "assistant", "content": m.content})

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=chat_messages,
            )
        except APIError as exc:
            raise LLMError(f"Erro na API OpenAI: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
