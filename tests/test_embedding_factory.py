"""Testes da factory de embeddings (app.services.embeddings.factory)."""
import sys
import types

import pytest

from app.core.config import Settings, get_settings
from app.services.embeddings.base import EmbeddingError
from app.services.embeddings.factory import (
    clear_embedding_provider_cache,
    get_embedding_provider,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_embedding_provider_cache()
    get_settings.cache_clear()
    yield
    clear_embedding_provider_cache()
    get_settings.cache_clear()


def _set_env(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key.upper(), value)
    get_settings.cache_clear()
    clear_embedding_provider_cache()


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch):
    class _FakeModel:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            self.model_name = model_name
            self.device = device

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 384 for _ in texts]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


def test_disabled_returns_none(monkeypatch) -> None:
    _set_env(monkeypatch, embedding_provider="disabled")
    assert get_embedding_provider() is None


def test_openai_selected(monkeypatch) -> None:
    _set_env(monkeypatch, embedding_provider="openai", openai_api_key="sk-fake")
    from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider

    provider = get_embedding_provider()
    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_sentence_transformers_selected(monkeypatch, fake_sentence_transformers) -> None:
    _set_env(monkeypatch, embedding_provider="sentence_transformers", embedding_model="BAAI/bge-small-en-v1.5")
    from app.services.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )

    provider = get_embedding_provider()
    assert isinstance(provider, SentenceTransformersProvider)
    assert provider.dimension == 384


def test_unknown_backend_raises(monkeypatch) -> None:
    settings = Settings.model_construct(embedding_provider="pinecone")
    monkeypatch.setattr("app.services.embeddings.factory.get_settings", lambda: settings)
    get_embedding_provider.cache_clear()
    with pytest.raises(EmbeddingError, match="desconhecido"):
        get_embedding_provider()
    get_embedding_provider.cache_clear()


def test_clear_cache_forces_new_instance(monkeypatch, fake_sentence_transformers) -> None:
    _set_env(monkeypatch, embedding_provider="sentence_transformers", embedding_model="BAAI/bge-small-en-v1.5")
    first = get_embedding_provider()
    clear_embedding_provider_cache()
    second = get_embedding_provider()
    assert first is not second


def test_without_clear_cache_same_instance(monkeypatch, fake_sentence_transformers) -> None:
    _set_env(monkeypatch, embedding_provider="sentence_transformers", embedding_model="BAAI/bge-small-en-v1.5")
    assert get_embedding_provider() is get_embedding_provider()
