"""Esquemas de entrada e saída da API (Pydantic)."""
import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    reply: str
    model: str
    memories_used: int


class MemoryCreateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    category: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=4000)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryResponse(BaseModel):
    id: uuid.UUID
    user_id: str
    category: str
    content: str
    relevance: float
    created_at: datetime

    model_config = {"from_attributes": True}
