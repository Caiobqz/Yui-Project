"""Contabilidade de uso dos modelos de IA: tokens e custo estimado.

Os preços são por milhão de tokens (input, output) em USD. O match é por
prefixo do model id, tolerando sufixos de versão retornados pelos provedores.
Modelos desconhecidos geram registro com custo nulo (nunca bloqueiam o fluxo).
"""
import uuid
from decimal import Decimal

from app.models.usage import UsageRecord
from app.services.llm.base import LLMResponse

# (input USD/MTok, output USD/MTok)
PRICES_PER_MTOK: dict[str, tuple[Decimal, Decimal]] = {
    "claude-sonnet-4-6": (Decimal("3.00"), Decimal("15.00")),
    "claude-sonnet-5": (Decimal("3.00"), Decimal("15.00")),
    "claude-haiku-4-5": (Decimal("1.00"), Decimal("5.00")),
    "claude-opus-4-8": (Decimal("5.00"), Decimal("25.00")),
    "gpt-4o": (Decimal("2.50"), Decimal("10.00")),
}

_MTOK = Decimal(1_000_000)


def estimate_cost_usd(
    model: str, input_tokens: int | None, output_tokens: int | None
) -> Decimal | None:
    for prefix, (price_in, price_out) in PRICES_PER_MTOK.items():
        if model.startswith(prefix):
            cost = (
                Decimal(input_tokens or 0) * price_in
                + Decimal(output_tokens or 0) * price_out
            ) / _MTOK
            return cost.quantize(Decimal("0.000001"))
    return None


def build_usage_record(
    user_id: uuid.UUID,
    conversation_id: uuid.UUID | None,
    response: LLMResponse,
) -> UsageRecord:
    return UsageRecord(
        user_id=user_id,
        conversation_id=conversation_id,
        model=response.model,
        input_tokens=response.input_tokens or 0,
        output_tokens=response.output_tokens or 0,
        estimated_cost_usd=estimate_cost_usd(
            response.model, response.input_tokens, response.output_tokens
        ),
    )
