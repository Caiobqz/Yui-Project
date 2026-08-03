"""Testes de validação de dimensão e registro de proveniência de embeddings.

Cobre o requisito de segurança "valide a dimensão antes da persistência e
da consulta" e o registro de qual provider/modelo gerou cada embedding.

O teste do fallback pgvector (`test_retrieve_semantic_pgvector_dimension_
mismatch_falls_back`) usa uma sessão FALSA que simula o dialeto
'postgresql' e o erro de dimensão do operador `<=>` — não há Postgres real
disponível neste ambiente, então este teste valida apenas o CONTROLE DE
FLUXO (try/except/rollback/fallback) do código, não o comportamento real
do pgvector. Isso NÃO é um teste de integração real; é um teste unitário
com mock do caminho condicional por dialeto.
"""
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.agents.memory_agent import MemoryAgent
from app.models.base import Base
from app.models.user import User
from app.services.embeddings.base import EmbeddingProvider
from app.services.memory_service import MemoryService
from tests.fakes import FakeEmbeddings, FakeLLM


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def _create_user(factory: async_sessionmaker) -> uuid.UUID:
    async with factory() as session:
        user = User(email=f"{uuid.uuid4()}@x.com", hashed_password="x", name="Teste")
        session.add(user)
        await session.commit()
        return user.id


# --- MemoryAgent.embed_texts(): validação de dimensão --------------------------


class _WrongDimensionEmbeddings(EmbeddingProvider):
    """Declara dimensão 384 mas devolve vetores de dimensão 100 — simula um
    provedor com bug ou uma incompatibilidade não detectada em outro lugar."""

    @property
    def dimension(self) -> int:
        return 384

    @property
    def provider_name(self) -> str:
        return "wrong"

    @property
    def model_name(self) -> str:
        return "wrong-model"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 100 for _ in texts]


async def test_embed_texts_dimension_mismatch_degrades_to_none(session_factory) -> None:
    agent = MemoryAgent(
        llm=FakeLLM(), embeddings=_WrongDimensionEmbeddings(), session_factory=session_factory
    )
    result = await agent.embed_texts(["texto de teste"])
    assert result is None


async def test_embed_texts_correct_dimension_passes_through(session_factory) -> None:
    agent = MemoryAgent(
        llm=FakeLLM(), embeddings=FakeEmbeddings(dimension=8), session_factory=session_factory
    )
    result = await agent.embed_texts(["texto de teste"])
    assert result is not None
    assert len(result[0]) == 8


async def test_embed_texts_none_when_embeddings_disabled(session_factory) -> None:
    agent = MemoryAgent(llm=FakeLLM(), embeddings=None, session_factory=session_factory)
    assert await agent.embed_texts(["texto"]) is None


# --- Proveniência: remember() populates embedding_provider/model ---------------


async def test_remember_populates_provenance_when_embedding_succeeds(session_factory) -> None:
    user_id = await _create_user(session_factory)
    agent = MemoryAgent(
        llm=FakeLLM(), embeddings=FakeEmbeddings(dimension=8), session_factory=session_factory
    )
    entry = await agent.remember(user_id, "gosto de programação em Python", category="interesses")
    assert entry is not None
    assert entry.embedding is not None
    assert entry.embedding_provider == "fake"
    assert entry.embedding_model == "fake-embeddings"


async def test_remember_leaves_provenance_null_when_embeddings_disabled(session_factory) -> None:
    user_id = await _create_user(session_factory)
    agent = MemoryAgent(llm=FakeLLM(), embeddings=None, session_factory=session_factory)
    entry = await agent.remember(user_id, "gosto de programação em Python", category="interesses")
    assert entry is not None
    assert entry.embedding is None
    assert entry.embedding_provider is None
    assert entry.embedding_model is None


async def test_remember_leaves_provenance_null_when_embedding_fails(session_factory) -> None:
    user_id = await _create_user(session_factory)
    agent = MemoryAgent(
        llm=FakeLLM(), embeddings=_WrongDimensionEmbeddings(), session_factory=session_factory
    )
    entry = await agent.remember(user_id, "gosto de programação em Python", category="interesses")
    assert entry is not None
    assert entry.embedding is None
    assert entry.embedding_provider is None
    assert entry.embedding_model is None


# --- MemoryService.retrieve_semantic(): memórias com dimensão stale ------------


async def test_retrieve_semantic_skips_stale_dimension_without_crash(session_factory) -> None:
    user_id = await _create_user(session_factory)
    async with session_factory() as session:
        service = MemoryService(session)
        await service.create(
            user_id, "geral", "memória antiga (modelo trocado)",
            embedding=[0.1] * 384, embedding_provider="sentence_transformers", embedding_model="bge-small",
        )
        await service.create(
            user_id, "geral", "memória atual",
            embedding=[0.2] * 1536, embedding_provider="openai", embedding_model="text-embedding-3-small",
        )
        await session.commit()

        results = await service.retrieve_semantic(user_id, [0.2] * 1536)
        contents = [m.content for m in results]
        assert "memória atual" in contents
        assert "memória antiga (modelo trocado)" not in contents


async def test_retrieve_semantic_pgvector_dimension_mismatch_falls_back() -> None:
    """Simula (via mock) o operador pgvector `<=>` falhando por mistura de
    dimensões, e confirma que o código faz rollback + fallback para busca
    completa em Python, em vez de propagar a exceção.

    NÃO é um teste de integração real com Postgres/pgvector — é um teste
    unitário do controle de fluxo condicional por dialeto, usando uma
    sessão e resultados totalmente mockados.
    """
    fake_memory = MagicMock()
    fake_memory.id = uuid.uuid4()
    fake_memory.embedding = [0.2] * 1536
    fake_memory.content = "memória via fallback"
    fake_memory.relevance = 0.8
    fake_memory.confidence = 0.9
    fake_memory.access_count = 0
    fake_memory.created_at = None
    fake_memory.last_accessed_at = None
    fake_memory.last_used_at = None

    ordered_result = MagicMock()
    ordered_result.scalars.side_effect = DBAPIError("stmt", {}, Exception("dimension mismatch"))

    fallback_result = MagicMock()
    fallback_result.scalars.return_value = [fake_memory]

    session = MagicMock()
    session.get_bind.return_value.dialect.name = "postgresql"
    session.rollback = AsyncMock()

    call_count = {"n": 0}

    async def _execute(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Primeira chamada (ORDER BY <=>): simula o erro do pgvector.
            raise DBAPIError("stmt", {}, Exception("dimension mismatch"))
        # Segunda chamada (fallback, busca completa): retorna candidatos.
        return fallback_result

    session.execute = AsyncMock(side_effect=_execute)

    service = MemoryService(session)
    results = await service.retrieve_semantic(uuid.uuid4(), [0.2] * 1536)

    assert session.rollback.await_count == 1
    assert call_count["n"] == 2
    assert len(results) == 1
    assert results[0].content == "memória via fallback"
