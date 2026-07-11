"""Rotas de conversa com a Yui.

Erros de LLM (LLMError) são convertidos em HTTP 502 por um exception
handler global registrado em app/main.py — inclusive quando ocorrem na
resolução de dependências (ex.: API key ausente), fora do corpo da rota.
"""
from fastapi import APIRouter

from app.api.deps import Yui
from app.api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(payload: ChatRequest, yui: Yui) -> ChatResponse:
    reply = await yui.process_message(
        user_id=payload.user_id,
        text=payload.message,
        conversation_id=payload.conversation_id,
    )
    return ChatResponse(
        conversation_id=reply.conversation_id,
        reply=reply.content,
        model=reply.model,
        memories_used=reply.memories_used,
    )
