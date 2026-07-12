"""Ferramentas de anotações (create_note, list_notes)."""
from typing import Any

from sqlalchemy import select

from app.models.note import Note
from app.services.llm.base import ToolSpec
from app.tools.base import Tool, ToolContext


async def _create_note(ctx: ToolContext, args: dict[str, Any]) -> str:
    async with ctx.session_factory() as session:
        note = Note(
            user_id=ctx.user_id,
            title=str(args["title"]),
            content=str(args["content"]),
        )
        session.add(note)
        await session.commit()
        return f"Nota criada: '{note.title}'"


async def _list_notes(ctx: ToolContext, args: dict[str, Any]) -> str:
    async with ctx.session_factory() as session:
        result = await session.execute(
            select(Note)
            .where(Note.user_id == ctx.user_id)
            .order_by(Note.created_at.desc())
            .limit(20)
        )
        notes = list(result.scalars())
    if not notes:
        return "Nenhuma nota encontrada."
    return "Notas (mais recentes primeiro):\n" + "\n".join(
        f"- {n.title}: {n.content[:120]}" for n in notes
    )


def note_tools() -> list[Tool]:
    return [
        Tool(
            spec=ToolSpec(
                name="create_note",
                description="Salva uma anotação do usuário (título + conteúdo).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["title", "content"],
                },
            ),
            handler=_create_note,
        ),
        Tool(
            spec=ToolSpec(
                name="list_notes",
                description="Lista as anotações mais recentes do usuário.",
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            handler=_list_notes,
        ),
    ]
