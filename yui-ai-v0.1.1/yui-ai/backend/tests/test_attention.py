"""Testes do Attention Manager (seleção determinística de memórias)."""
from app.cognition.attention import AttentionContext, AttentionManager
from app.models.memory import MemoryEntry


def _mem(content: str, category: str = "geral", **kwargs) -> MemoryEntry:
    return MemoryEntry(content=content, category=category, relevance=0.5, **kwargs)


def test_limit_is_respected() -> None:
    manager = AttentionManager()
    candidates = [(_mem(f"fato numero {i} distinto"), 0.9) for i in range(20)]
    selected = manager.select(candidates)
    from app.core.config import get_settings

    assert len(selected) == get_settings().memory_retrieval_limit


def test_redundancy_penalty_prefers_diverse_memories() -> None:
    manager = AttentionManager()
    # Três quase-duplicatas + uma memória diversa, todas com mesma similaridade.
    candidates = [
        (_mem("gosta muito de programar em python"), 0.9),
        (_mem("adora programar usando python"), 0.9),
        (_mem("programacao em python e sua paixao"), 0.9),
        (_mem("mora na cidade de salvador bahia"), 0.9),
    ]
    selected = manager.select(candidates)
    contents = [s.memory.content for s in selected]
    # A memória diversa entra antes da terceira variação redundante.
    assert "salvador" in contents[1]


def test_goal_terms_boost_relevant_memory() -> None:
    manager = AttentionManager()
    candidates = [
        (_mem("interesse casual por culinaria italiana"), 0.5),
        (_mem("quer dominar inteligencia artificial"), 0.5),
    ]
    ctx = AttentionContext(goal_terms={"inteligencia", "artificial"})
    selected = manager.select(candidates, ctx)
    assert "inteligencia" in selected[0].memory.content


def test_user_source_preference_boost() -> None:
    manager = AttentionManager()
    candidates = [
        (_mem("fato extraido automaticamente", source="extracted"), 0.6),
        (_mem("fato dito pelo proprio usuario", source="user"), 0.6),
    ]
    selected = manager.select(candidates)
    assert selected[0].memory.source == "user"
