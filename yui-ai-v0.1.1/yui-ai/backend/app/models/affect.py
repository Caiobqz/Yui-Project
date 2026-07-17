"""Affective State Model — estado afetivo computacional PERSISTENTE (v0.5).

Realismo (regra da identidade): a Yui não sente emoções humanas. Estas
dimensões são estados computacionais que persistem entre conversas e
influenciam decisões futuras — o tom da companhia (bloco do prompt) e o
julgamento de iniciativas (Bússola Moral). A lógica de atualização e
decaimento vive em app/cognition/affect.py; aqui fica só a persistência.

Dimensões (0..1):
- warmth   apego construído pela convivência (cresce e decai devagar);
- joy      alegria contextual (conquistas recentes; decai rápido);
- concern  preocupação protetora (frustração/dificuldade repetidas).
"""
import uuid

from sqlalchemy import Float, ForeignKey, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class AffectiveState(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "affective_states"

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    warmth: Mapped[float] = mapped_column(
        Float, default=0.1, server_default="0.1", nullable=False
    )
    joy: Mapped[float] = mapped_column(
        Float, default=0.0, server_default="0.0", nullable=False
    )
    concern: Mapped[float] = mapped_column(
        Float, default=0.0, server_default="0.0", nullable=False
    )
