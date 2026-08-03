"""Implementação do LLMProvider usando uma instância local do Ollama.

Usa `httpx` (já é dependência do projeto) em vez de um SDK dedicado do
Ollama — o protocolo é HTTP simples e não justifica uma dependência a mais.

Streaming de tool calls no Ollama é reconhecidamente inconsistente entre
versões (chunks intermediários às vezes não carregam `tool_calls`, que só
aparecem completos no chunk final com `done: true`) — diferente do formato
OpenAI/llama.cpp, que fragmenta argumentos como string JSON incremental.
Este provider não presume fragmentação: acumula `tool_calls` como já vêm
(dicts completos) sempre que aparecem em qualquer chunk.
"""
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.core.config import Settings
from app.services.llm.base import (
    ChatMessage,
    LLMCapabilities,
    LLMError,
    LLMProvider,
    LLMResponse,
    StreamChunk,
    ToolCall,
    ToolSpec,
)

logger = logging.getLogger("yui.llm.ollama")


def _to_ollama_messages(
    system_prompt: str, messages: list[ChatMessage]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                        }
                        for call in m.tool_calls
                    ],
                }
            )
        elif m.role == "tool":
            # O Ollama não usa tool_call_id — correlaciona pela ordem/nome.
            # Preservamos o conteúdo; a ausência de ID é uma limitação
            # conhecida do formato nativo do Ollama, não deste provider.
            out.append({"role": "tool", "content": m.content})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _to_ollama_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _parse_tool_calls(raw_calls: list[dict[str, Any]] | None) -> tuple[ToolCall, ...]:
    if not raw_calls:
        return ()
    calls = []
    for i, tc in enumerate(raw_calls):
        func = tc.get("function", {})
        arguments = func.get("arguments", {})
        # O Ollama já entrega argumentos como objeto JSON; alguns modelos
        # (via templates customizados) podem devolver string — toleramos
        # ambos sem quebrar o loop de ferramentas.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Argumentos de tool call com JSON inválido; ignorando.")
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        # O Ollama não gera um id de tool call — sintetizamos um estável
        # dentro da resposta para que o loop de ferramentas (que exige
        # tool_call_id em ChatMessage) continue funcionando.
        calls.append(ToolCall(id=f"ollama_call_{i}", name=func.get("name", ""), arguments=arguments))
    return tuple(calls)


def _parse_response(raw: dict[str, Any]) -> LLMResponse:
    message = raw.get("message", {})
    return LLMResponse(
        content=message.get("content") or "",
        model=raw.get("model", ""),
        # prompt_eval_count/eval_count são contagens exatas do tokenizer do
        # Ollama, equivalentes semanticamente a input/output tokens — não
        # são estimativas.
        input_tokens=raw.get("prompt_eval_count"),
        output_tokens=raw.get("eval_count"),
        tool_calls=_parse_tool_calls(message.get("tool_calls")),
    )


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not settings.ollama_base_url:
            raise LLMError("LLM_PROVIDER=ollama exige OLLAMA_BASE_URL configurada.")
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = model or settings.ollama_model
        if not self._model:
            raise LLMError("LLM_PROVIDER=ollama exige OLLAMA_MODEL configurado.")
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        # Cliente injetável para testes; em produção cria o próprio,
        # fechado explicitamente em aclose().
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url, timeout=settings.ollama_timeout_seconds
        )
        self._owns_client = client is None

        if settings.ollama_supports_tool_calling is not None:
            tool_calling = settings.ollama_supports_tool_calling
        else:
            # Default CONSERVADOR: False. Não há forma confiável de detectar
            # suporte a tool calling a partir do servidor Ollama (não é
            # exposto de forma estruturada por /api/show em todas as
            # versões e varia por modelo/template). Declarar suporte por
            # padrão para "qualquer modelo" seria uma promessa falsa que
            # poderia levar o loop de ferramentas a tentar usá-lo sem
            # garantia real. O operador que validou manualmente que o
            # modelo escolhido suporta tools deve habilitar explicitamente
            # via OLLAMA_SUPPORTS_TOOL_CALLING=true.
            tool_calling = False
        self._capabilities = LLMCapabilities(
            streaming=True, tool_calling=tool_calling, usage_metrics=True
        )

    def capabilities(self) -> LLMCapabilities:
        return self._capabilities

    async def aclose(self) -> None:
        """Fecha o cliente HTTP, se esta instância o criou."""
        if self._owns_client:
            await self._client.aclose()

    async def validate(self) -> None:
        """Override do contrato base: Ollama depende de um servidor externo,
        então a validação de disponibilidade é real (não no-op)."""
        await self.check_availability()

    async def check_availability(self) -> None:
        """Valida que o servidor responde e que o modelo configurado existe.

        Chamado no boot (fail-fast, via `validate()`) — não substitui o
        health check operacional (/health/ready), que continua sem
        depender de LLM.
        """
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Não foi possível conectar ao Ollama em '{self._base_url}': {exc}. "
                "Verifique se o servidor está em execução (`ollama serve`)."
            ) from exc

        data = response.json()
        available = {m.get("name") or m.get("model") for m in data.get("models", [])}
        if self._model not in available:
            raise LLMError(
                f"Modelo '{self._model}' não encontrado no Ollama em "
                f"'{self._base_url}'. Modelos disponíveis: "
                f"{', '.join(sorted(filter(None, available))) or '(nenhum)'}. "
                f"Baixe com: ollama pull {self._model}"
            )

    def _payload(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _to_ollama_messages(system_prompt, messages),
            "stream": stream,
            # Modelos com reasoning, como Qwen3, podem gastar todo o limite
            # em `thinking` e terminar sem produzir `message.content`.
            # A Yui precisa da resposta final, não do raciocínio interno.
            "think": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        if tools:
            if not self._capabilities.tool_calling:
                raise LLMError(
                    f"O modelo '{self._model}' (backend Ollama) não tem tool "
                    "calling habilitado. Por padrão, nenhum modelo Ollama é "
                    "considerado capaz de tool calling automaticamente — "
                    "defina OLLAMA_SUPPORTS_TOOL_CALLING=true somente após "
                    "validar manualmente que o modelo escolhido suporta."
                )
            payload["tools"] = _to_ollama_tools(tools)
        return payload

    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        payload = self._payload(system_prompt, messages, tools, stream=False)

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError(f"Timeout ao comunicar com o Ollama: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            detail = _extract_error_detail(exc.response)
            raise LLMError(f"Erro na API do Ollama ({exc.response.status_code}): {detail}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Erro de conexão com o Ollama: {exc}") from exc
        return _parse_response(response.json())

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        payload = self._payload(system_prompt, messages, tools, stream=True)
        content_parts: list[str] = []
        model_name = ""
        final_tool_calls: tuple[ToolCall, ...] = ()
        input_tokens: int | None = None
        output_tokens: int | None = None

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    # Corpo de erro em requisição streaming: lido inteiro
                    # antes de levantar, para a mensagem ficar acionável.
                    body = await response.aread()
                    detail = _extract_error_detail_bytes(body)
                    raise LLMError(
                        f"Erro na API do Ollama ({response.status_code}): {detail}"
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        # Erro no meio do stream: o Ollama não muda o status
                        # HTTP, só inclui "error" no objeto NDJSON.
                        raise LLMError(f"Erro do Ollama durante o streaming: {chunk['error']}")
                    if chunk.get("model"):
                        model_name = chunk["model"]
                    message = chunk.get("message", {})
                    if message.get("content"):
                        content_parts.append(message["content"])
                        yield StreamChunk(delta=message["content"])
                    if message.get("tool_calls"):
                        # Não fragmentado: cada chunk com tool_calls já traz
                        # o conjunto completo até aquele ponto.
                        final_tool_calls = _parse_tool_calls(message["tool_calls"])
                    if chunk.get("done"):
                        input_tokens = chunk.get("prompt_eval_count")
                        output_tokens = chunk.get("eval_count")
        except httpx.TimeoutException as exc:
            raise LLMError(f"Timeout ao comunicar com o Ollama (stream): {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Erro de conexão com o Ollama (stream): {exc}") from exc

        yield StreamChunk(
            response=LLMResponse(
                content="".join(content_parts),
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=final_tool_calls,
            )
        )


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except (json.JSONDecodeError, ValueError):
        return response.text


def _extract_error_detail_bytes(body: bytes) -> str:
    try:
        return str(json.loads(body).get("error", body.decode(errors="replace")))
    except (json.JSONDecodeError, ValueError):
        return body.decode(errors="replace")
