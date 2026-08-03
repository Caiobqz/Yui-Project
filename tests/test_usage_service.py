"""Testes da contabilidade de uso e custo estimado."""
from decimal import Decimal

from app.services.usage_service import estimate_cost_usd


def test_known_model_cost() -> None:
    # 1M de input + 1M de output no claude-sonnet-4-6 = 3 + 15 USD.
    cost = estimate_cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == Decimal("18.000000")


def test_versioned_model_id_matches_by_prefix() -> None:
    assert estimate_cost_usd("gpt-4o-2024-11-20", 1_000_000, 0) == Decimal("2.500000")


def test_unknown_model_returns_none() -> None:
    assert estimate_cost_usd("modelo-misterioso", 1000, 1000) is None


def test_none_token_counts_are_treated_as_zero() -> None:
    assert estimate_cost_usd("claude-sonnet-4-6", None, None) == Decimal("0.000000")
