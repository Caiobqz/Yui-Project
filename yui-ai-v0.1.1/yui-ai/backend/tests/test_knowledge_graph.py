"""Testes do Knowledge Graph derivado."""
import uuid

from app.cognition.knowledge_graph import build_knowledge_graph
from app.models.memory import MemoryEntry
from app.models.task import Task
from app.models.user import User


async def test_graph_links_user_goals_steps_and_categories(db_session) -> None:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()

    parent = Task(user_id=user.id, title="Aprender python")
    db_session.add(parent)
    await db_session.flush()
    db_session.add(Task(user_id=user.id, title="Instalar python", parent_id=parent.id))
    db_session.add(
        MemoryEntry(user_id=user.id, category="python", content="usa python no trabalho")
    )
    await db_session.flush()

    graph = await build_knowledge_graph(db_session, user.id)
    kinds = {n.kind for n in graph.nodes.values()}
    assert {"user", "goal", "step", "category", "memory"} <= kinds

    # A categoria 'python' compartilha termo com o objetivo 'Aprender python'.
    terms = graph.related_terms({"Aprender python"})
    assert "python" in terms


async def test_empty_user_graph_has_only_user_node(db_session) -> None:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()

    graph = await build_knowledge_graph(db_session, user.id)
    assert list(graph.nodes.values())[0].kind == "user"
    assert graph.edges == []
