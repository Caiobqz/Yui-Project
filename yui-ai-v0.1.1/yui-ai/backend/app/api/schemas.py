"""Esquemas de entrada e saída da API (Pydantic)."""
import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

# --- Autenticação -----------------------------------------------------------


class UserRegisterRequest(BaseModel):
    email: EmailStr
    # máximo 72: limite do bcrypt (bytes além disso seriam ignorados no hash)
    password: str = Field(min_length=8, max_length=72)
    name: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    name: str | None
    plan: str
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Chat --------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    reply: str
    model: str
    memories_used: int


# --- Memórias ----------------------------------------------------------------


class MemoryCreateRequest(BaseModel):
    category: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=1000)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryResponse(BaseModel):
    id: uuid.UUID
    memory_type: str
    category: str
    content: str
    relevance: float
    confidence: float
    source: str
    usage_count: int
    last_used_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Tarefas e notas -----------------------------------------------------------


class TaskResponse(BaseModel):
    id: uuid.UUID
    parent_id: uuid.UUID | None
    title: str
    description: str | None
    due_at: datetime | None
    status: str
    position: int
    created_at: datetime

    model_config = {"from_attributes": True}


class NoteResponse(BaseModel):
    id: uuid.UUID
    title: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Permissões ---------------------------------------------------------------


class PermissionResponse(BaseModel):
    tool_name: str
    category: str
    allowed: bool
    # "default" (política da ferramenta) ou "user" (decisão explícita).
    source: str

    model_config = {"from_attributes": True}


class PermissionUpdateRequest(BaseModel):
    allowed: bool
