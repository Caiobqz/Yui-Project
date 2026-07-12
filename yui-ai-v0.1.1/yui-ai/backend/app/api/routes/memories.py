"""Gerenciamento da memória de longo prazo do usuário autenticado."""
import uuid

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DbSession
from app.api.schemas import MemoryCreateRequest, MemoryResponse
from app.services.memory_service import MemoryService

router = APIRouter(prefix="/memories", tags=["memories"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreateRequest, user: CurrentUser, session: DbSession
) -> MemoryResponse:
    service = MemoryService(session)
    entry = await service.create(
        user_id=user.id,
        category=payload.category,
        content=payload.content,
        relevance=payload.relevance,
    )
    return MemoryResponse.model_validate(entry)


@router.get("")
async def list_memories(user: CurrentUser, session: DbSession) -> list[MemoryResponse]:
    service = MemoryService(session)
    entries = await service.list_by_user(user.id)
    return [MemoryResponse.model_validate(e) for e in entries]


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> None:
    service = MemoryService(session)
    deleted = await service.delete(user.id, memory_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memória não encontrada."
        )
