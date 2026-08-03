"""Testes do ranking lexical de memórias (v0.1)."""
from app.services.memory_service import _normalize


def test_normalize_removes_accents_and_stopwords() -> None:
    terms = _normalize("Como está a minha evolução nos estudos de Python?")
    assert "evolucao" in terms
    assert "estudos" in terms
    assert "python" in terms
    assert "como" not in terms  # stopword
    assert "a" not in terms


def test_normalize_ignores_short_tokens() -> None:
    assert "ai" not in _normalize("ai de mim")
