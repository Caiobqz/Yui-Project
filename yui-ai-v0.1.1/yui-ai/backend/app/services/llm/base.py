"""Contrato abstrato para provedores de modelos de linguagem.

Nenhum outro módulo da Yui deve importar SDKs de provedores diretamente.
Isso mantém o sistema independente de fornecedor (Claude, OpenAI, locais).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMError(RuntimeError):
    """Erro genérico de comunicação com o provedor de IA."""


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
    ) -> LLMResponse:
        """Gera uma resposta a partir do system prompt e do histórico."""
        raise NotImplementedError
