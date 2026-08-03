"""Personality Engine — traços e estilo de conversa, carregados de YAML.

Separação do núcleo cognitivo:
- QUEM a Yui é (identidade imutável) → app/cognition/identity.py (código).
- COMO a Yui conversa (traços/estilo, configurável) → este módulo (YAML).
- Como conversar com CADA usuário (aprendido) → Adaptation Engine
  (app/cognition/user_model.py).
"""
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.core.config import get_settings


class Personality(BaseModel):
    """Traços de personalidade e estilo de comunicação configuráveis."""

    traits: dict[str, str] = Field(default_factory=dict)
    communication_style: list[str] = Field(default_factory=list)

    def to_style_prompt(self) -> str:
        """Bloco de personalidade/estilo do system prompt."""
        parts: list[str] = []
        if self.traits:
            lines = "\n".join(
                f"- {name.capitalize()}: {description.strip()}"
                for name, description in self.traits.items()
            )
            parts.append(f"Traços de personalidade:\n{lines}")
        if self.communication_style:
            lines = "\n".join(f"- {item}" for item in self.communication_style)
            parts.append(f"Estilo de comunicação:\n{lines}")
        return "\n\n".join(parts)


class PersonalityLoadError(RuntimeError):
    pass


@lru_cache
def get_personality() -> Personality:
    """Carrega a personalidade do arquivo configurado (cacheada por processo).

    O caminho é resolvido relativo à raiz do backend (ver Settings), então o
    boot funciona independentemente do diretório de onde o processo partiu.
    """
    path: Path = get_settings().personality_file
    if not path.exists():
        raise PersonalityLoadError(f"Arquivo de personalidade não encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Personality.model_validate(data)
