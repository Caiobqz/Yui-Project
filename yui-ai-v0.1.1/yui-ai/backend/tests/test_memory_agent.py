"""Testes do Memory System: análise pós-turno, consolidação e recuperação."""
import json
import uuid

from sqlalchemy import select

from app.agents.memory_agent import MemoryAgent
from app.cognition.analyzer import TurnAnalyzer, parse_analysis
from app.models.memory import MemoryEntry
from app.models.user import User
from app.services.llm.base import LLMResponse
from tests.fakes import FakeEmbeddings, FakeLLM


def _analysis_response(memories: list[dict], adaptation: list[str] | None = None) -> LLMResponse:
    return LLMResponse(
        content=json.dumps({"memories": memories, "adaptation": adaptation or []}),
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


async def test_analysis_produces_typed_memories_and_adaptation(session_factory) -> None:
    llm = FakeLLM(
        script=[
            _analysis_response(
                [
                    {
                        "content": "Terminou o projeto do curso",
                        "category": "conquistas",
                        "type": "episodic",
                        "importance": 0.8,
                        "confidence": 0.9,
                    }
                ],
                adaptation=["Prefere exemplos práticos"],
            )
        ]
    )
    analysis = await TurnAnalyzer(llm).analyze("terminei o projeto!", "Parabéns!")
    assert len(analysis.memories) == 1
    assert analysis.memories[0].memory_type == "episodic"
    assert analysis.adaptation == ["Prefere exemplos práticos"]

    user_id = await _create_user(session_factory)
    agent = MemoryAgent(llm, FakeEmbeddings(), session_factory)
    created = await agent.store_candidates(user_id, analysis.memories)
    assert created == 1

    async with session_factory() as session:
        entry = (await session.execute(select(MemoryEntry))).scalar_one()
        assert entry.memory_type == "episodic"
        assert entry.source == "extracted"
        assert entry.relevance == 0.8
        assert entry.confidence == 0.9
        assert entry.embedding is not None


async def test_duplicate_candidate_reinforces_existing_memory(session_factory) -> None:
    """Consolidação: informação reconfirmada fortalece a memória, não duplica."""
    user_id = await _create_user(session_factory)
    same_vector = [1.0, 0.0, 0.0, 0.0]
    embeddings = FakeEmbeddings(
        mapping={
            "Gosta de estudar à noite": same_vector,
            "Prefere estudar no período noturno": same_vector,
        },
        dimension=4,
    )
    agent = MemoryAgent(FakeLLM(), embeddings, session_factory)

    first = await agent.remember(
        user_id, "Gosta de estudar à noite", category="habitos", confidence=0.7
    )
    assert first is not None

    analysis_memories, _ = parse_analysis(
        json.dumps(
            {
                "memories": [
                    {
                        "content": "Prefere estudar no período noturno",
                        "category": "habitos",
                        "type": "procedural",
                        "importance": 0.6,
                        "confidence": 0.95,
                    }
                ]
            }
        )
    )
    created = await agent.store_candidates(user_id, analysis_memories)
    assert created == 0  # não criou nova...

    async with session_factory() as session:
        entry = (await session.execute(select(MemoryEntry))).scalar_one()
        assert entry.usage_count == 1  # ...mas reforçou a existente
        assert entry.confidence == 0.95
        assert entry.last_used_at is not None


async def test_candidates_rejected_by_guardian_or_confidence(session_factory) -> None:
    user_id = await _create_user(session_factory)
    memories, _ = parse_analysis(
        json.dumps(
            {
                "memories": [
                    {"content": "senha do email: hunter2", "category": "x",
                     "type": "semantic", "importance": 0.9, "confidence": 0.9},
                    {"content": "Talvez goste de café", "category": "preferencias",
                     "type": "semantic", "importance": 0.3, "confidence": 0.2},
                ]
            }
        )
    )
    agent = MemoryAgent(FakeLLM(), FakeEmbeddings(), session_factory)
    assert await agent.store_candidates(user_id, memories) == 0


def test_parse_analysis_tolerates_garbage() -> None:
    assert parse_analysis("claro! aqui vai...") == ([], [])
    assert parse_analysis('{"memories": "não é lista"}') == ([], [])
    memories, notes = parse_analysis(
        '```json\n{"memories": [], "adaptation": ["Nota"]}\n```'
    )
    assert memories == [] and notes == ["Nota"]


async def test_retrieve_semantic_without_shared_keywords(
    session_factory, db_session
) -> None:
    """Memória sobre IA recuperada por pergunta sobre estudos, sem palavras
    em comum — e com recência/frequência atualizadas."""
    user_id = await _create_user(session_factory)
    memory_vec = [1.0, 0.0, 0.0, 0.0]
    query_vec = [0.95, 0.05, 0.0, 0.0]
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
    assert results[0].last_used_at is not None
    assert results[0].usage_count == 1
