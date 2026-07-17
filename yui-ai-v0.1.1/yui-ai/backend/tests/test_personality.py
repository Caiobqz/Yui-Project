"""Testes do Identity System e do Personality Engine."""
import dataclasses

import pytest

from app.cognition.identity import YUI_IDENTITY, identity_prompt
from app.core.personality import Personality


def test_identity_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        YUI_IDENTITY.name = "Outra"  # type: ignore[misc]


def test_identity_prompt_contains_all_sections() -> None:
    prompt = identity_prompt()
    assert "Você é Yui" in prompt
    # v0.5: companheira (presença), nunca ferramenta/assistente.
    assert "companheira" in prompt
    assert "assistente" not in prompt
    assert "Propósito:" in prompt
    assert "Acompanhar o usuário" in prompt
    assert "Valores:" in prompt
    assert "Privacidade do usuário" in prompt
    assert "Limitações:" in prompt
    assert "Regras invioláveis:" in prompt
    # Realismo exigido: nada de consciência/emoções humanas reais.
    assert "estados computacionais" in prompt
    assert "não possui consciência" in prompt.lower()


def test_style_prompt_contains_traits_and_style() -> None:
    personality = Personality(
        traits={"curiosidade": "Demonstra interesse.", "honestidade": "Direta."},
        communication_style=["Tom amigável."],
    )
    prompt = personality.to_style_prompt()
    assert "Traços de personalidade:" in prompt
    assert "- Curiosidade: Demonstra interesse." in prompt
    assert "Estilo de comunicação:" in prompt
    assert "- Tom amigável." in prompt
