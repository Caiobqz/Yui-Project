"""Seleção do provedor de embeddings a partir da configuração.

`EMBEDDING_PROVIDER=disabled` retorna None e a memória cai automaticamente
para a busca lexical — a Yui continua funcional sem chave da OpenAI.
"""
from functools import lru_cache

from app.core.config import get_settings
from app.services.embeddings.base import EmbeddingError, EmbeddingProvider


@lru_cache
def get_embedding_provider() -> EmbeddingProvider | None:
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "disabled":
        return None
    if provider == "openai":
        from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(settings)

    raise EmbeddingError(
        f"Provedor de embeddings desconhecido: '{settings.embedding_provider}'. "
        "Valores aceitos: 'openai', 'disabled'."
    )
