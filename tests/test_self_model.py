"""Testes do Self Model (read-only) e do World Model."""
import dataclasses

import pytest

from app.cognition.self_model import build_self_model, get_self_model
from app.cognition.world_model import build_world_model
from app.tools.registry import build_default_registry


def test_self_model_is_read_only() -> None:
    model = get_self_model()
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.version = "9.9.9"  # type: ignore[misc]


def test_self_model_reflects_registry_and_identity() -> None:
    registry = build_default_registry()
    model = build_self_model(registry)
    assert model.name == "Yui"
    assert model.version == "0.5.0"
    assert "proteger" in model.purpose
    assert set(model.tools_available) == {s.name for s in registry.specs()}
    assert model.health == "operational"
    assert "moral_compass" in model.modules_loaded
    # Prompt do domínio self não vaza segredos nem promete o que não tem.
    assert "Yui" in model.to_prompt()


def test_world_model_separates_environment_and_general() -> None:
    world = build_world_model(get_self_model())
    env = world.environment_prompt()
    assert "Data e hora" in env and "pt-BR" in env
    assert "treinamento" in world.general_boundary()
