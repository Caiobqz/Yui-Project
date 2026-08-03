"""Provedor de embeddings via API da OpenAI (text-embedding-3-*)."""
from openai import APIError, AsyncOpenAI

from app.core.config import Settings
from app.services.embeddings.base import (
    EmbeddingError,
    EmbeddingProvider,
    validate_embedding_dimension,
)

# Dimensão nativa dos modelos de embedding da OpenAI atualmente suportados.
# Não inferida do provedor em runtime (a API não declara a dimensão de
# forma estruturada antes da primeira chamada) — mantida explícita e
# revisada manualmente quando novos modelos forem adicionados.
_KNOWN_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


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
        if self._model not in _KNOWN_DIMENSIONS:
            raise EmbeddingError(
                f"EMBEDDING_MODEL='{self._model}' não é um modelo OpenAI "
                "conhecido por este provedor. Modelos suportados: "
                + ", ".join(sorted(_KNOWN_DIMENSIONS))
                + ". Isso evita persistir vetores com dimensão presumida "
                "incorretamente."
            )
        self._dimension = _KNOWN_DIMENSIONS[self._model]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

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
        vectors = [list(item.embedding) for item in items]
        for vector in vectors:
            validate_embedding_dimension(
                vector, self._dimension, context=f"OpenAIEmbeddingProvider/{self._model}"
            )
        return vectors
