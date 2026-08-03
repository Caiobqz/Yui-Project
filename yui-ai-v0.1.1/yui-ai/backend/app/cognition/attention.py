"""Attention Manager — seleciona quais memórias realmente importam.

Determinístico (não usa LLM). Cada candidata recebe um Attention Score:

    score = similaridade × importância × recência (relevância base)
          + alinhamento a objetivos ativos
          + vínculo de relacionamento
          + preferência do usuário (memória criada por ele)
          + reforço do grafo (termos conectados aos objetivos)
          − penalidade por redundância (MMR contra o que já foi selecionado)

Somente as memórias de maior score entram no prompt. A penalidade de
redundância evita encher o contexto com N variações da mesma informação.
"""
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.models.memory import MemoryEntry
from app.services.memory_service import lexical_overlap, memory_score


@dataclass(frozen=True)
class AttentionContext:
    """Sinais do turno que modulam a atenção."""

    goal_terms: set[str] = field(default_factory=set)
    graph_terms: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ScoredMemory:
    memory: MemoryEntry
    score: float
    base_relevance: float


class AttentionManager:
    def __init__(self) -> None:
        settings = get_settings()
        self._limit = settings.memory_retrieval_limit
        self._w_goal = settings.attention_goal_weight
        self._w_relationship = settings.attention_relationship_weight
        self._w_preference = settings.attention_preference_weight
        self._w_graph = settings.attention_graph_weight
        self._w_redundancy = settings.attention_redundancy_penalty

    def _static_boost(self, memory: MemoryEntry, ctx: AttentionContext) -> float:
        boost = 0.0
        terms = _terms(memory)
        if ctx.goal_terms & terms:
            boost += self._w_goal
        if ctx.graph_terms & terms:
            boost += self._w_graph
        if memory.memory_type == "relationship":
            boost += self._w_relationship
        if memory.source == "user":
            boost += self._w_preference
        return boost

    def select(
        self,
        candidates: list[tuple[MemoryEntry, float]],
        ctx: AttentionContext | None = None,
    ) -> list[ScoredMemory]:
        """Seleção gulosa com penalidade de redundância (estilo MMR)."""
        ctx = ctx or AttentionContext()
        pool = [
            ScoredMemory(
                memory=memory,
                score=memory_score(similarity, memory) + self._static_boost(memory, ctx),
                base_relevance=memory_score(similarity, memory),
            )
            for memory, similarity in candidates
        ]

        selected: list[ScoredMemory] = []
        selected_terms: list[set[str]] = []
        while pool and len(selected) < self._limit:
            best_index = -1
            best_adjusted = float("-inf")
            for i, scored in enumerate(pool):
                terms = _terms(scored.memory)
                redundancy = max(
                    (_jaccard(terms, prev) for prev in selected_terms), default=0.0
                )
                adjusted = scored.score - self._w_redundancy * redundancy
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_index = i
            chosen = pool.pop(best_index)
            selected.append(chosen)
            selected_terms.append(_terms(chosen.memory))
        return selected


def _terms(memory: MemoryEntry) -> set[str]:
    # Reaproveita a normalização lexical do serviço de memória.
    from app.services.memory_service import _normalize

    return _normalize(memory.content) | _normalize(memory.category)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def terms_of(text: str) -> set[str]:
    """Termos normalizados de um texto arbitrário (objetivos, etc.)."""
    from app.services.memory_service import _normalize

    return _normalize(text)


__all__ = ["AttentionManager", "AttentionContext", "ScoredMemory", "terms_of", "lexical_overlap"]
