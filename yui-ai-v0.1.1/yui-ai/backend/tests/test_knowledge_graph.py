"""Testes do Knowledge Graph derivado."""
import uuid

from app.cognition.knowledge_graph import (
    build_knowledge_graph,
    related_category_terms,
)
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


async def test_light_path_equals_full_graph_related_terms(db_session) -> None:
    """O caminho leve do turno equivale ao grafo completo (sem materializá-lo)."""
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()

    parent = Task(user_id=user.id, title="Aprender python")
    db_session.add(parent)
    await db_session.flush()
    db_session.add(Task(user_id=user.id, title="Instalar python", parent_id=parent.id))
    db_session.add_all(
        [
            MemoryEntry(user_id=user.id, category="python", content="usa no trabalho"),
            MemoryEntry(user_id=user.id, category="culinária", content="gosta de cozinhar"),
        ]
    )
    await db_session.flush()

    labels = {"Aprender python"}
    graph = await build_knowledge_graph(db_session, user.id)
    light = await related_category_terms(db_session, user.id, labels)
    assert light == graph.related_terms(labels)
    assert "python" in light and "culinaria" not in light

    assert await related_category_terms(db_session, user.id, set()) == set()


async def test_empty_user_graph_has_only_user_node(db_session) -> None:
    user = User(email=f"{uuid.uuid4()}@example.com", hashed_password="h")
    db_session.add(user)
    await db_session.flush()

    graph = await build_knowledge_graph(db_session, user.id)
    assert list(graph.nodes.values())[0].kind == "user"
    assert graph.edges == []
