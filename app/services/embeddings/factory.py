"""Seleção do provedor de embeddings a partir da configuração.

`EMBEDDING_PROVIDER=disabled` retorna None e a memória cai automaticamente
para a busca lexical — a Yui continua funcional sem nenhum provedor de
embeddings, local ou comercial.
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
    if provider == "sentence_transformers":
        from app.services.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )

        return SentenceTransformersProvider(settings)

    raise EmbeddingError(
        f"Provedor de embeddings desconhecido: '{settings.embedding_provider}'. "
        "Valores aceitos: 'openai', 'sentence_transformers', 'disabled'."
    )


def clear_embedding_provider_cache() -> None:
    """Limpa o cache da factory de embeddings.

    Uso exclusivo em testes — permite reconstruir o provider com uma nova
    configuração dentro do mesmo processo (`get_settings.cache_clear()`
    também deve ser chamado quando a configuração mudou).
    """
    get_embedding_provider.cache_clear()
