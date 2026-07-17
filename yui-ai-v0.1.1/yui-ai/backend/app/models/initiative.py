"""Iniciativas autônomas persistentes (v0.5).

Registro de cada ação autônoma aprovada pela Bússola Moral, fechando o
ciclo do Judgement Engine (Decisão → Execução → Aprendizado): a iniciativa
nasce "pending", é entregue no próximo momento natural de conversa
("delivered") e nunca é reproposta dentro do cooldown (`dedupe_key`).
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin

INITIATIVE_STATUSES = ("pending", "delivered", "dismissed")


class InitiativeRecord(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "initiatives"
    __table_args__ = (
        CheckConstraint(
            "status IN ({})".format(
                ", ".join(f"'{s}'" for s in INITIATIVE_STATUSES)
            ),
            name="ck_initiatives_status",
        ),
        Index("ix_initiatives_user_status", "user_id", "status"),
        Index("ix_initiatives_user_key", "user_id", "dedupe_key"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    # Identidade estável da situação (ex.: "check_in:<plan_id>") — base do
    # cooldown que impede repropor a mesma iniciativa.
    dedupe_key: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="pending", server_default="pending", nullable=False
    )
    delivered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
