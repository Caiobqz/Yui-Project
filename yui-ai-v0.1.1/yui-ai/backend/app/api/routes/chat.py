"""Rotas de conversa com a Yui.

O usuário vem exclusivamente do token (CurrentUser). Erros de domínio
(LLM, rate limit, conversa inexistente) são convertidos em HTTP pelos
handlers globais registrados em app/main.py.
"""
from fastapi import APIRouter

from app.api.deps import CurrentUser, Yui
from app.api.schemas import ChatRequest, ChatResponse

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
