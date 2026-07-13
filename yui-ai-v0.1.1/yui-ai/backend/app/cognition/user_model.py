"""Adaptation Engine + Relationship Model — o modelo do usuário.

Adaptação: notas de comunicação aprendidas pós-turno ("prefere exemplos
práticos"), com deduplicação lexical e teto — viram orientação de estilo
no prompt. Relacionamento: desde quando e quantas vezes usuário e Yui
interagem, dando continuidade entre sessões.
"""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.base import utcnow
from app.models.user_profile import UserProfile
from app.services.memory_service import _lexical_overlap


class UserModelService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(self, user_id: uuid.UUID) -> UserProfile:
        profile = await self._session.scalar(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        if profile is None:
            profile = UserProfile(user_id=user_id, preferences=[])
            self._session.add(profile)
            await self._session.flush()
        return profile

    @staticmethod
    def register_interaction(profile: UserProfile) -> None:
        profile.interaction_count += 1
        profile.last_interaction_at = utcnow()

    @staticmethod
    def apply_adaptation(profile: UserProfile, notes: list[str]) -> int:
        """Incorpora notas de adaptação novas; retorna quantas foram aceitas.

        Deduplicação lexical (nota ~equivalente já existente é descartada) e
        teto de notas (as mais antigas saem — o aprendizado recente prevalece).
        """
        max_notes = get_settings().adaptation_max_notes
        current = list(profile.preferences or [])
        added = 0
        for note in notes:
            note = note.strip()
            if not note or len(note) > 200:
                continue
            # Limiar mais baixo que o de memórias: notas são frases curtas e
            # uma palavra extra não deve gerar nota "nova" equivalente.
            if any(_lexical_overlap(note, existing) >= 0.7 for existing in current):
                continue
            current.append(note)
            added += 1
        if added:
            # JSON mutável: reatribui a lista para o SQLAlchemy detectar.
            profile.preferences = current[-max_notes:]
        return added

    @staticmethod
    def relationship_line(profile: UserProfile) -> str | None:
        """Linha de continuidade do relacionamento para o prompt."""
        if profile.interaction_count <= 0:
            return "Esta é a primeira conversa de vocês."
        first = profile.created_at.strftime("%d/%m/%Y")
        return (
            f"Vocês interagem desde {first}; esta é a conversa de número "
            f"{profile.interaction_count + 1}."
        )
