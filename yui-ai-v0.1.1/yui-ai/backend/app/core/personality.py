"""Carregamento e validação da personalidade da Yui a partir de YAML."""
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.core.config import get_settings


class Personality(BaseModel):
    """Representação validada da personalidade configurável."""

    name: str
    role: str
    objective: str
    communication_style: list[str] = Field(default_factory=list)
    behavior_rules: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    def to_system_prompt(self) -> str:
        """Converte a personalidade em um bloco de instruções para o LLM."""

        def bullets(items: list[str]) -> str:
            return "\n".join(f"- {item}" for item in items)

        return (
            f"Você é {self.name}, {self.role}.\n\n"
            f"Objetivo:\n{self.objective.strip()}\n\n"
            f"Estilo de comunicação:\n{bullets(self.communication_style)}\n\n"
            f"Regras de comportamento:\n{bullets(self.behavior_rules)}\n\n"
            f"Limitações:\n{bullets(self.limitations)}"
        )


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
