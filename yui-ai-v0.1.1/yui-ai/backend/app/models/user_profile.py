"""Perfil cognitivo do usuário: adaptação e relacionamento.

Adaptation Engine: `preferences` guarda notas de comunicação aprendidas
("prefere exemplos práticos", "gosta de respostas curtas"), com teto e
deduplicação — injetadas no prompt como orientação de estilo por usuário.

Relationship Model: `interaction_count`/`last_interaction_at` (+ created_at
como primeira interação) dão continuidade ao relacionamento; marcos ficam
em memórias do tipo "relationship".
"""
import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class UserProfile(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "user_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    # Notas de adaptação aprendidas (lista de frases curtas).
    preferences: Mapped[list[str]] = mapped_column(
        JSON, default=list, nullable=False
    )
    interaction_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    last_interaction_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Interação em que a última pergunta de curiosidade foi sugerida —
    # espaçamento determinístico para a conversa nunca virar interrogatório.
    last_curiosity_interaction: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
