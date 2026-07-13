"""Provedor de embeddings via API da OpenAI (text-embedding-3-*)."""
from openai import APIError, AsyncOpenAI

from app.core.config import Settings
from app.services.embeddings.base import EmbeddingError, EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=openai exige OPENAI_API_KEY configurada."
            )
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self._model = settings.embedding_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=texts
            )
        except APIError as exc:
            raise EmbeddingError(f"Erro na API de embeddings: {exc}") from exc
        # A API preserva a ordem dos inputs; o sort por index é defensivo.
        items = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in items]
