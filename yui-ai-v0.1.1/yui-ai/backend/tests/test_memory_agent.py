"""Testes do MemoryAgent: extração automática, dedupe, triagem e recuperação."""
import json
import uuid

from sqlalchemy import select

from app.agents.memory_agent import MemoryAgent
from app.models.memory import MemoryEntry
from app.models.usage import UsageRecord
from app.models.user import User
from app.services.llm.base import LLMResponse
from tests.fakes import FakeEmbeddings, FakeLLM


def _extraction_response(memories: list[dict]) -> LLMResponse:
    return LLMResponse(
        content=json.dumps({"memories": memories}),
        model="fake-model",
        input_tokens=20,
        output_tokens=10,
    )


async def _create_user(session_factory) -> uuid.UUID:
    async with session_factory() as session:
        user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
        session.add(user)
        await session.commit()
        return user.id


async def test_extraction_stores_memory_with_metadata(session_factory) -> None:
    user_id = await _create_user(session_factory)
    llm = FakeLLM(
        script=[
            _extraction_response(
                [
                    {
                        "content": "Quer trabalhar com inteligência artificial",
                        "category": "objetivos",
                        "importance": 0.9,
                        "confidence": 0.85,
                    }
                ]
            )
        ]
    )
    agent = MemoryAgent(llm, FakeEmbeddings(), session_factory)

    saved = await agent.extract_from_turn(
        user_id, uuid.uuid4(), "quero muito trabalhar com IA", "Que ótimo objetivo!"
    )
    assert saved == 1

    async with session_factory() as session:
        entry = (await session.execute(select(MemoryEntry))).scalar_one()
        assert entry.source == "extracted"
        assert entry.category == "objetivos"
        assert entry.relevance == 0.9
        assert entry.confidence == 0.85
        assert entry.embedding is not None
        # A chamada de extração foi contabilizada.
        usage = (await session.execute(select(UsageRecord))).scalar_one()
        assert usage.input_tokens == 20


async def test_extraction_skips_duplicates_via_embedding(session_factory) -> None:
    user_id = await _create_user(session_factory)
    same_vector = [1.0, 0.0, 0.0, 0.0]
    embeddings = FakeEmbeddings(
        mapping={
            "Gosta de estudar à noite": same_vector,
            "Prefere estudar no período noturno": same_vector,
        },
        dimension=4,
    )
    extraction = [
        _extraction_response(
            [{"content": "Gosta de estudar à noite", "category": "habitos",
              "importance": 0.6, "confidence": 0.9}]
        ),
        _extraction_response(
            [{"content": "Prefere estudar no período noturno", "category": "habitos",
              "importance": 0.6, "confidence": 0.9}]
        ),
    ]
    agent = MemoryAgent(FakeLLM(script=extraction), embeddings, session_factory)

    assert await agent.extract_from_turn(user_id, uuid.uuid4(), "a", "b") == 1
    # Semanticamente idêntica (mesmo vetor) → deduplicada.
    assert await agent.extract_from_turn(user_id, uuid.uuid4(), "c", "d") == 0


async def test_extraction_rejects_secrets_and_low_confidence(session_factory) -> None:
    user_id = await _create_user(session_factory)
    extraction = [
        _extraction_response(
            [
                {"content": "senha do email: hunter2", "category": "x",
                 "importance": 0.9, "confidence": 0.9},
                {"content": "Talvez goste de café", "category": "preferencias",
                 "importance": 0.3, "confidence": 0.2},
            ]
        )
    ]
    agent = MemoryAgent(FakeLLM(script=extraction), FakeEmbeddings(), session_factory)

    assert await agent.extract_from_turn(user_id, uuid.uuid4(), "a", "b") == 0


async def test_extraction_tolerates_non_json_response(session_factory) -> None:
    user_id = await _create_user(session_factory)
    llm = FakeLLM(
        script=[LLMResponse(content="claro! aqui vai...", model="fake-model")]
    )
    agent = MemoryAgent(llm, None, session_factory)
    assert await agent.extract_from_turn(user_id, uuid.uuid4(), "a", "b") == 0


async def test_retrieve_semantic_without_shared_keywords(
    session_factory, db_session
) -> None:
    """O exemplo do requisito: memória sobre IA recuperada por pergunta
    sobre estudos, sem palavras em comum."""
    user_id = await _create_user(session_factory)
    memory_vec = [1.0, 0.0, 0.0, 0.0]
    query_vec = [0.95, 0.05, 0.0, 0.0]  # próximo, mas não idêntico
    embeddings = FakeEmbeddings(
        mapping={
            "Usuário quer trabalhar com inteligência artificial": memory_vec,
            "Qual área devo estudar?": query_vec,
        },
        dimension=4,
    )
    agent = MemoryAgent(FakeLLM(), embeddings, session_factory)
    entry = await agent.remember(
        user_id,
        content="Usuário quer trabalhar com inteligência artificial",
        category="objetivos",
        importance=0.9,
    )
    assert entry is not None

    query_embedding = await agent.embed_query("Qual área devo estudar?")
    results = await agent.retrieve(
        db_session, user_id, "Qual área devo estudar?", query_embedding
    )
    assert [m.content for m in results] == [
        "Usuário quer trabalhar com inteligência artificial"
    ]
    # Recuperação marca last_used_at.
    assert results[0].last_used_at is not None
