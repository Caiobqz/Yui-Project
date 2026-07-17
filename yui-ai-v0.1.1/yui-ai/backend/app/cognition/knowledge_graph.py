"""Knowledge Graph — grafo de entidades DERIVADO dos dados existentes.

Liga usuário → objetivos → etapas e usuário → categorias de memória
(interesses, preferências, hábitos). É derivado (construído sob demanda a
partir de user_profiles, tasks e memory_entries), não uma nova fonte de
verdade: mantém a evolução incremental e evita extração de entidades por LLM.

Uso: recuperação relacional — dado um conjunto de objetivos ativos, retorna
os termos das memórias conectadas por categoria, reforçando a atenção sobre
informações relacionadas mesmo sem similaridade direta com a pergunta.

Versão futura: persistir o grafo e extrair entidades/relações com o modelo
utilitário para capturar ligações que não estão na estrutura atual.
"""
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cognition.attention import terms_of
from app.models.memory import MemoryEntry
from app.models.task import Task


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: str  # user | goal | step | category | memory
    label: str


@dataclass
class KnowledgeGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)  # (src, rel, dst)

    def add_node(self, node: GraphNode) -> None:
        self.nodes.setdefault(node.id, node)

    def add_edge(self, src: str, rel: str, dst: str) -> None:
        self.edges.append((src, rel, dst))

    def neighbors(self, node_id: str) -> list[GraphNode]:
        out: list[GraphNode] = []
        for src, _rel, dst in self.edges:
            if src == node_id and dst in self.nodes:
                out.append(self.nodes[dst])
            elif dst == node_id and src in self.nodes:
                out.append(self.nodes[src])
        return out

    def related_terms(self, goal_labels: set[str]) -> set[str]:
        """Termos de categorias de memória conectadas aos objetivos dados."""
        terms: set[str] = set()
        goal_ids = [
            n.id
            for n in self.nodes.values()
            if n.kind == "goal" and n.label in goal_labels
        ]
        for goal_id in goal_ids:
            for neighbor in self.neighbors(goal_id):
                if neighbor.kind == "category":
                    terms |= terms_of(neighbor.label)
        return terms

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [
                {"id": n.id, "kind": n.kind, "label": n.label}
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": s, "relation": r, "target": d} for s, r, d in self.edges
            ],
        }


async def related_category_terms(
    session: AsyncSession, user_id: uuid.UUID, goal_labels: set[str]
) -> set[str]:
    """Termos das categorias de memória relacionadas aos objetivos dados.

    Caminho LEVE para o turno: equivale a
    `build_knowledge_graph(...).related_terms(goal_labels)` sem materializar
    o grafo — que carregaria TODAS as memórias (conteúdo incluso) a cada
    turno com objetivos ativos. Aqui basta um SELECT DISTINCT de categorias.
    O grafo completo continua disponível para a rota /knowledge-graph.
    """
    if not goal_labels:
        return set()
    result = await session.execute(
        select(MemoryEntry.category)
        .where(MemoryEntry.user_id == user_id)
        .distinct()
    )
    goal_term_sets = [terms_of(label) for label in goal_labels]
    terms: set[str] = set()
    for (category,) in result:
        category_terms = terms_of(category)
        if any(category_terms & goal_terms for goal_terms in goal_term_sets):
            terms |= category_terms
    return terms


async def build_knowledge_graph(
    session: AsyncSession, user_id: uuid.UUID
) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    user_node = GraphNode(id=f"user:{user_id}", kind="user", label="usuário")
    graph.add_node(user_node)

    # Objetivos (planos) e suas etapas.
    plans = await session.execute(
        select(Task).where(Task.user_id == user_id, Task.parent_id.is_(None))
    )
    parent_ids: list[uuid.UUID] = []
    for parent in plans.scalars():
        parent_ids.append(parent.id)
        goal_id = f"goal:{parent.id}"
        graph.add_node(GraphNode(id=goal_id, kind="goal", label=parent.title))
        graph.add_edge(user_node.id, "persegue", goal_id)

    if parent_ids:
        steps = await session.execute(
            select(Task).where(Task.parent_id.in_(parent_ids))
        )
        for step in steps.scalars():
            step_id = f"step:{step.id}"
            graph.add_node(GraphNode(id=step_id, kind="step", label=step.title))
            graph.add_edge(f"goal:{step.parent_id}", "tem_etapa", step_id)

    # Categorias de memória como entidades de interesse/hábito/preferência.
    memories = await session.execute(
        select(MemoryEntry).where(MemoryEntry.user_id == user_id)
    )
    for memory in memories.scalars():
        category_id = f"category:{memory.category}"
        graph.add_node(
            GraphNode(id=category_id, kind="category", label=memory.category)
        )
        graph.add_edge(user_node.id, "conhece", category_id)
        memory_id = f"memory:{memory.id}"
        graph.add_node(
            GraphNode(id=memory_id, kind="memory", label=memory.content[:80])
        )
        graph.add_edge(category_id, "contém", memory_id)

    # Liga objetivos a categorias que compartilham termos (relação derivada).
    goal_nodes = [n for n in graph.nodes.values() if n.kind == "goal"]
    category_nodes = [n for n in graph.nodes.values() if n.kind == "category"]
    for goal in goal_nodes:
        goal_terms = terms_of(goal.label)
        for category in category_nodes:
            if goal_terms & terms_of(category.label):
                graph.add_edge(goal.id, "relaciona", category.id)

    return graph
