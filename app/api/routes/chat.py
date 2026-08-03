"""Rotas de conversa com a Yui: resposta completa e streaming (SSE).

O usuário vem exclusivamente do token (CurrentUser). Na rota síncrona, erros
de domínio são convertidos em HTTP pelos handlers globais; na rota SSE os
erros viram eventos `error` (o status HTTP já foi enviado ao abrir o stream).
"""
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, Yui
from app.api.schemas import ChatRequest, ChatResponse
from app.core.exceptions import ConversationNotFoundError, RateLimitExceededError

logger = logging.getLogger("yui.chat")

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(payload: ChatRequest, user: CurrentUser, yui: Yui) -> ChatResponse:
    reply = await yui.process_message(
        user_id=user.id,
        plan=user.plan,
        text=payload.message,
        conversation_id=payload.conversation_id,
    )
    return ChatResponse(
        conversation_id=reply.conversation_id,
        reply=reply.content,
        model=reply.model,
        memories_used=reply.memories_used,
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest, user: CurrentUser, yui: Yui
) -> StreamingResponse:
    """Resposta em tempo real via Server-Sent Events.

    Eventos: delta (texto incremental), tool (ferramentas em execução),
    done (metadados finais) e error (mensagem genérica).
    """

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in yui.stream_message(
                user_id=user.id,
                plan=user.plan,
                text=payload.message,
                conversation_id=payload.conversation_id,
            ):
                yield _sse(event)
        except RateLimitExceededError as exc:
            yield _sse({"type": "error", "detail": str(exc)})
        except ConversationNotFoundError:
            yield _sse({"type": "error", "detail": "Conversa não encontrada."})
        except Exception:  # noqa: BLE001 — detalhe no log, genérico no cliente
            logger.exception("Falha no streaming do chat.")
            yield _sse(
                {
                    "type": "error",
                    "detail": "Falha ao gerar a resposta. Tente novamente.",
                }
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
