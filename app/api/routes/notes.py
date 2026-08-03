"""Consulta de notas do usuário autenticado (criação via ferramenta create_note)."""
from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.api.schemas import NoteResponse
from app.models.note import Note

router = APIRouter(prefix="/notes", tags=["notes"])


@router.get("")
async def list_notes(user: CurrentUser, session: DbSession) -> list[NoteResponse]:
    result = await session.execute(
        select(Note).where(Note.user_id == user.id).order_by(Note.created_at.desc())
    )
    return [NoteResponse.model_validate(n) for n in result.scalars()]
