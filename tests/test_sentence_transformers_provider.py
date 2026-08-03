"""Testes do SentenceTransformersProvider.

Usa um módulo `sentence_transformers` FALSO (via monkeypatch em
sys.modules) — a biblioteca real traz PyTorch como dependência transitiva
e baixaria um modelo do Hugging Face Hub no primeiro uso (domínio não
disponível neste ambiente de teste). Isso testa a INTEGRAÇÃO (medição de
dimensão, validação, device, tratamento de erros) sem validar a biblioteca
real — um modelo real requer validação manual (ver README).
"""
import asyncio
import sys
import types
from typing import Any

import pytest

from app.core.config import Settings
from app.services.embeddings.base import EmbeddingDimensionMismatchError, EmbeddingError
from app.services.embeddings.sentence_transformers_provider import (
    SentenceTransformersProvider,
)


def _settings(**overrides: Any) -> Settings:
    params: dict[str, Any] = {
        "embedding_provider": "sentence_transformers",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_device": "cpu",
    }
    params.update(overrides)
    return Settings(**params)


@pytest.fixture
def fake_module(monkeypatch: pytest.MonkeyPatch):
    """Devolve uma função que injeta um módulo fake com dimensão configurável."""

    def _install(dimension: int = 384, raise_on_device: str | None = None):
        class _FakeModel:
            def __init__(self, model_name: str, device: str | None = None) -> None:
                if raise_on_device is not None and device == raise_on_device:
                    raise RuntimeError(f"dispositivo '{device}' indisponível (simulado)")
                self.model_name = model_name
                self.device = device

            def encode(self, texts: list[str]) -> list[list[float]]:
                return [[0.1] * dimension for _ in texts]

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        return _FakeModel

    return _install


# --- Construção e medição de dimensão ------------------------------------------


def test_dimension_measured_via_probe(fake_module) -> None:
    fake_module(dimension=384)
    provider = SentenceTransformersProvider(_settings())
    assert provider.dimension == 384


def test_provenance_properties(fake_module) -> None:
    fake_module(dimension=384)
    provider = SentenceTransformersProvider(_settings())
    assert provider.provider_name == "sentence_transformers"
    assert provider.model_name == "BAAI/bge-small-en-v1.5"


def test_missing_library_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbeddingError, match="sentence-transformers"):
        SentenceTransformersProvider(_settings())


def test_device_error_not_silently_downgraded(fake_module) -> None:
    """Requisito de segurança: device configurado indisponível deve falhar
    claramente, nunca cair para CPU sem aviso."""
    fake_module(raise_on_device="cuda")
    with pytest.raises(EmbeddingError, match="cuda"):
        SentenceTransformersProvider(_settings(embedding_device="cuda"))


# --- embed() e validação de dimensão --------------------------------------------


async def test_embed_returns_vectors_with_correct_dimension(fake_module) -> None:
    fake_module(dimension=384)
    provider = SentenceTransformersProvider(_settings())
    vectors = await provider.embed(["ola", "mundo"])
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)


async def test_embed_empty_list_returns_empty(fake_module) -> None:
    fake_module(dimension=384)
    provider = SentenceTransformersProvider(_settings())
    assert await provider.embed([]) == []


async def test_embed_dimension_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sondagem inicial mede 384, mas encode() posterior (simulando um bug
    da biblioteca ou modelo inconsistente) devolve dimensão diferente —
    deve ser detectado antes de devolver o vetor ao chamador."""

    call_count = {"n": 0}

    class _InconsistentModel:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            pass

        def encode(self, texts: list[str]) -> list[list[float]]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [[0.1] * 384]  # sondagem inicial no __init__
            return [[0.1] * 100 for _ in texts]  # chamada real, dimensão errada

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _InconsistentModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    provider = SentenceTransformersProvider(_settings())
    assert provider.dimension == 384
    with pytest.raises(EmbeddingDimensionMismatchError):
        await provider.embed(["texto"])


async def test_embed_runs_off_event_loop(fake_module) -> None:
    """encode() é síncrono/bloqueante — embed() deve rodar via
    asyncio.to_thread, permitindo outras tarefas prosseguirem em paralelo."""
    fake_module(dimension=8)
    provider = SentenceTransformersProvider(_settings())

    other_task_progressed = False

    async def other_task() -> None:
        nonlocal other_task_progressed
        await asyncio.sleep(0)
        other_task_progressed = True

    results = await asyncio.gather(provider.embed(["texto"]), other_task())
    assert results[0][0] is not None
    assert other_task_progressed
