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
from collections.abc import AsyncGenerator
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


@dataclass(frozen=True)
class LLMCapabilities:
    """Capacidades declaradas por um provedor (formato neutro).

    Modelos locais variam muito em suporte a tool calling e em contagem
    exata de tokens — ao contrário das APIs comerciais, que suportam ambos
    incondicionalmente. Este contrato existe para que o restante do sistema
    detecte capacidades reais em vez de presumi-las por nome de provedor ou
    de modelo. O default (`_DEFAULT_CAPABILITIES`) preserva o comportamento
    anterior: streaming, tool calling e métricas de uso sempre disponíveis
    — correto para Claude e OpenAI, que já garantem os três.
    """

    streaming: bool = True
    tool_calling: bool = True
    usage_metrics: bool = True


_DEFAULT_CAPABILITIES = LLMCapabilities()


class LLMError(RuntimeError):
    """Erro genérico de comunicação com o provedor de IA."""


class LLMProvider(ABC):
    def capabilities(self) -> LLMCapabilities:
        """Capacidades deste provedor. Default: as três sempre disponíveis.

        Providers locais (llama.cpp, Ollama) sobrescrevem para refletir o
        que o backend/modelo configurado realmente suporta. Nenhum chamador
        deve inferir capacidades por `isinstance` ou por nome de modelo —
        sempre por este método.
        """
        return _DEFAULT_CAPABILITIES

    async def validate(self) -> None:
        """Validação assíncrona de disponibilidade, chamada uma vez no boot.

        Default: no-op — Claude/OpenAI validam a API key de forma síncrona
        no `__init__`; llama.cpp valida o arquivo GGUF também no `__init__`.
        Providers que dependem de um servidor externo (Ollama) sobrescrevem
        para checar disponibilidade real (conexão, modelo presente) antes
        do primeiro request de um usuário. Levanta `LLMError` em caso de
        falha, com mensagem acionável.
        """
        return None

    async def aclose(self) -> None:
        """Libera recursos do provedor (conexões HTTP, etc.) no shutdown.

        Default: no-op. Providers com recursos que exigem encerramento
        explícito (ex.: cliente HTTP do Ollama) sobrescrevem.
        """
        return None

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
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming de resposta.

        Implementação default: delega para `generate` e emite o texto inteiro
        como um único delta — qualquer provedor funciona no endpoint de
        streaming; provedores com suporte nativo sobrescrevem.
        """
        response = await self.generate(system_prompt, messages, tools)
        if response.content:
            yield StreamChunk(delta=response.content)
        yield StreamChunk(response=response)
