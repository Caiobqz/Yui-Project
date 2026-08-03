"""Contrato abstrato para provedores de modelos de linguagem.

Nenhum outro módulo da Yui deve importar SDKs de provedores diretamente.
Isso mantém o sistema independente de fornecedor (Claude, OpenAI, locais).

O contrato cobre três capacidades:
- geração simples (`generate`);
- tool calling — o modelo solicita ferramentas via `ToolCall` e recebe os
  resultados como mensagens de papel "tool";
- streaming (`generate_stream`) — deltas de texto seguidos da resposta final.
"""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolSpec:
    """Descrição de uma ferramenta oferecida ao modelo (formato neutro)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """Solicitação de execução de ferramenta feita pelo modelo."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    # Preenchido em mensagens 'assistant' que solicitam ferramentas.
    tool_calls: tuple[ToolCall, ...] = ()
    # Preenchido em mensagens 'tool' (resultado devolvido ao modelo).
    tool_call_id: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class StreamChunk:
    """Evento de streaming: delta de texto OU a resposta final consolidada."""

    delta: str | None = None
    response: LLMResponse | None = None


class LLMError(RuntimeError):
    """Erro genérico de comunicação com o provedor de IA."""


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        """Gera uma resposta a partir do system prompt e do histórico."""
        raise NotImplementedError

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming de resposta.

        Implementação default: delega para `generate` e emite o texto inteiro
        como um único delta — qualquer provedor funciona no endpoint de
        streaming; provedores com suporte nativo sobrescrevem.
        """
        response = await self.generate(system_prompt, messages, tools)
        if response.content:
            yield StreamChunk(delta=response.content)
        yield StreamChunk(response=response)
