"""Implementação do LLMProvider usando llama.cpp local (llama-cpp-python).

Carrega um único arquivo GGUF por instância (e portanto por processo, já
que a factory cacheia com `lru_cache`) e reutiliza o modelo entre chamadas.
A biblioteca é síncrona e não garante segurança de uma única instância sob
concorrência: toda geração roda em thread dedicada (`asyncio.to_thread`) e
é limitada a UMA geração por vez (semáforo travado em 1). Esta versão não
implementa isolamento de contexto por requisição (múltiplos `Llama()`
independentes ou contexts isolados), então não há forma comprovadamente
segura de permitir concorrência real — `LLM_MAX_CONCURRENT_REQUESTS` acima
de 1 é ignorado, com aviso registrado no log.

Tool calling depende do `chat_format` configurado: só os formatos com
sufixo "-function-calling" (ou "functionary") têm suporte confiável na
biblioteca — "chatml" simples NUNCA é tratado como prova de suporte.
Nenhuma lógica aqui decide por nome de MODELO — apenas pelo `chat_format`
explicitamente configurado ou por `LLM_SUPPORTS_TOOL_CALLING` quando o
operador quiser sobrescrever a detecção.
"""
import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

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

logger = logging.getLogger("yui.llm.llama_cpp")

# Chat formats com suporte a tool calling documentado pela biblioteca.
# Mantido como lista pequena e explícita — não por nome de MODELO, por
# FORMATO (que é o que a biblioteca realmente usa para decidir como
# interpretar `tools`). IMPORTANTE: "chatml" simples (sem o sufixo
# "-function-calling") NÃO está nesta lista e NUNCA deve ser tratado como
# prova de suporte a ferramentas — o chat template puro não impõe a
# gramática GBNF que a lib usa para garantir chamadas de função válidas.
_TOOL_CALLING_CHAT_FORMATS = {"chatml-function-calling", "functionary", "functionary-v2"}


def _to_llama_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Converte o formato neutro para o formato OpenAI-compatível esperado
    por `create_chat_completion` (dicts, não objetos)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in m.tool_calls
                    ],
                }
            )
        elif m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                }
            )
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _to_llama_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
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


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Argumentos de tool call com JSON inválido; ignorando.")
        return {}
    return data if isinstance(data, dict) else {}


def _parse_response(raw: dict[str, Any]) -> LLMResponse:
    choice = raw["choices"][0]
    message = choice.get("message", {})
    usage = raw.get("usage") or {}
    raw_tool_calls = message.get("tool_calls") or []
    tool_calls = tuple(
        ToolCall(
            id=tc["id"],
            name=tc["function"]["name"],
            arguments=_parse_arguments(tc["function"].get("arguments")),
        )
        for tc in raw_tool_calls
    )
    return LLMResponse(
        content=message.get("content") or "",
        model=raw.get("model", ""),
        # llama.cpp calcula tokens via o tokenizer do próprio modelo —
        # contagem exata, não estimativa, quando presente.
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        tool_calls=tool_calls,
    )


class LlamaCppProvider(LLMProvider):
    def __init__(self, settings: Settings, model_path: Path | None = None) -> None:
        path = model_path or settings.resolved_llm_model_path
        if path is None:
            raise LLMError(
                "LLM_PROVIDER=llama_cpp exige LLM_MODEL_PATH apontando para um "
                "arquivo .gguf existente."
            )
        if not path.is_file():
            raise LLMError(
                f"Modelo GGUF não encontrado em '{path}'. Verifique LLM_MODEL_PATH "
                "(caminhos relativos são resolvidos a partir da raiz do backend)."
            )

        # Import tardio: evita exigir a dependência quando outro backend é usado.
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise LLMError(
                "LLM_PROVIDER=llama_cpp exige o pacote 'llama-cpp-python' instalado "
                "(extra opcional: pip install -e '.[local-llama]')."
            ) from exc

        chat_format = settings.llm_chat_format or None
        self._chat_format = chat_format
        try:
            self._model = Llama(
                model_path=str(path),
                n_ctx=settings.llm_context_size,
                n_gpu_layers=settings.llm_gpu_layers,
                n_threads=settings.llm_threads,
                n_batch=settings.llm_batch_size,
                chat_format=chat_format,
                verbose=False,
            )
        except Exception as exc:  # a lib levanta tipos variados (ValueError, OSError, RuntimeError)
            raise LLMError(f"Falha ao carregar o modelo GGUF em '{path}': {exc}") from exc

        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature
        # Concorrência TRAVADA EM 1 (não configurável para cima): esta
        # implementação usa uma única instância `Llama()` sem isolamento de
        # contexto por requisição (sem múltiplos contexts, sem pool de
        # instâncias) — não há garantia de segurança da biblioteca sob
        # concorrência real. Permitir N>1 aqui exigiria uma implementação
        # comprovadamente segura (ex.: múltiplos `Llama()` independentes ou
        # contexts isolados por requisição), que não existe nesta versão.
        # Se o operador configurar um valor maior, ele é ignorado e um
        # aviso é registrado — nunca falha silenciosamente para 1 sem
        # visibilidade.
        configured = settings.llm_max_concurrent_requests
        if configured > 1:
            logger.warning(
                "LLM_MAX_CONCURRENT_REQUESTS=%d configurado, mas esta implementação "
                "não possui isolamento de contexto por requisição — forçando 1. "
                "Múltiplas gerações simultâneas contra a mesma instância Llama() "
                "não são seguras sem essa implementação.",
                configured,
            )
        self._inference_semaphore = asyncio.Semaphore(1)

        if settings.llm_supports_tool_calling is not None:
            tool_calling = settings.llm_supports_tool_calling
        else:
            tool_calling = chat_format in _TOOL_CALLING_CHAT_FORMATS
        self._capabilities = LLMCapabilities(
            streaming=True,
            tool_calling=tool_calling,
            # A contagem de tokens é real (tokenizer do modelo), mas só
            # aparece quando a biblioteca a inclui na resposta — não há
            # garantia incondicional como nas APIs comerciais.
            usage_metrics=True,
        )

    def capabilities(self) -> LLMCapabilities:
        return self._capabilities

    def _check_tools(self, tools: list[ToolSpec] | None) -> None:
        if tools and not self._capabilities.tool_calling:
            raise LLMError(
                "O modelo/chat_format configurado (LLM_CHAT_FORMAT="
                f"'{self._chat_format}') não declara suporte a tool calling. "
                "Configure LLM_CHAT_FORMAT='chatml-function-calling' com um "
                "modelo compatível, ou LLM_SUPPORTS_TOOL_CALLING=true se você "
                "validou manualmente que o modelo suporta."
            )

    def _request_kwargs(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                *_to_llama_messages(messages),
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = _to_llama_tools(tools)
        return kwargs

    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        self._check_tools(tools)
        kwargs = self._request_kwargs(system_prompt, messages, tools)

        async with self._inference_semaphore:
            try:
                raw = await asyncio.to_thread(
                    self._model.create_chat_completion, **kwargs
                )
            except Exception as exc:  # a lib não define uma exceção única própria
                raise LLMError(f"Erro na inferência llama.cpp: {exc}") from exc
        return _parse_response(raw)

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        self._check_tools(tools)
        kwargs = self._request_kwargs(system_prompt, messages, tools)

        content_parts: list[str] = []
        tool_acc: dict[int, dict[str, str]] = {}
        model_name = ""

        # Abertura E consumo do stream ficam sob o mesmo semáforo que limita
        # `generate`: abrir o gerador já dispara avaliação do prompt contra o
        # modelo, então é trabalho de inferência tanto quanto consumi-lo.
        async with self._inference_semaphore:
            # create_chat_completion(stream=True) devolve um gerador SÍNCRONO.
            # Consumi-lo inteiro dentro de um único asyncio.to_thread perderia
            # o streaming incremental; em vez disso avançamos um passo por
            # vez em thread, preservando deltas reais e permitindo
            # cancelamento entre passos (o cliente SSE pode desconectar a
            # qualquer momento).
            try:
                stream_iter = await asyncio.to_thread(
                    self._model.create_chat_completion, stream=True, **kwargs
                )
            except Exception as exc:
                raise LLMError(f"Erro ao iniciar streaming llama.cpp: {exc}") from exc

            _STREAM_DONE = object()
            while True:
                try:
                    next_chunk = await asyncio.to_thread(next, stream_iter, _STREAM_DONE)
                except Exception as exc:
                    raise LLMError(f"Erro durante streaming llama.cpp: {exc}") from exc
                if next_chunk is _STREAM_DONE:
                    break
                chunk = cast(dict[str, Any], next_chunk)
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                if chunk.get("model"):
                    model_name = chunk["model"]
                if delta.get("content"):
                    content_parts.append(delta["content"])
                    yield StreamChunk(delta=delta["content"])
                for tc in delta.get("tool_calls") or []:
                    index = tc.get("index", 0)
                    acc = tool_acc.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        acc["id"] = tc["id"]
                    func = tc.get("function") or {}
                    if func.get("name"):
                        acc["name"] = func["name"]
                    if func.get("arguments"):
                        acc["arguments"] += func["arguments"]

        tool_calls = tuple(
            ToolCall(id=acc["id"], name=acc["name"], arguments=_parse_arguments(acc["arguments"]))
            for _, acc in sorted(tool_acc.items())
        )
        # llama.cpp não reporta usage no modo streaming de forma consistente
        # entre versões/formatos — None é honesto aqui (usage_service trata
        # None como 0 tokens registrados, nunca bloqueia o fluxo).
        yield StreamChunk(
            response=LLMResponse(
                content="".join(content_parts),
                model=model_name,
                input_tokens=None,
                output_tokens=None,
                tool_calls=tool_calls,
            )
        )
