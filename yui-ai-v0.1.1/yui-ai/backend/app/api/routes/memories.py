"""Rotas de gerenciamento da memória de longo prazo."""
import uuid

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DbSession
from app.api.schemas import MemoryCreateRequest, MemoryResponse
from app.services.memory_service import MemoryService

router = APIRouter(prefix="/memories", tags=["memories"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreateRequest, session: DbSession
) -> MemoryResponse:
    service = MemoryService(session)
    entry = await service.create(
        user_id=payload.user_id,
        category=payload.category,
        content=payload.content,
        relevance=payload.relevance,
    )
    return MemoryResponse.model_validate(entry)


@router.get("/{user_id}")
async def list_memories(user_id: str, session: DbSession) -> list[MemoryResponse]:
    service = MemoryService(session)
    entries = await service.list_by_user(user_id)
    return [MemoryResponse.model_validate(e) for e in entries]


@router.delete("/{user_id}/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    user_id: str, memory_id: uuid.UUID, session: DbSession
) -> None:
    service = MemoryService(session)
    deleted = await service.delete(user_id, memory_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Memória não encontrada."
        )
