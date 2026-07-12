"""Tarefas e planos do usuário.

Um plano é uma tarefa "pai" cujas etapas são tarefas filhas (`parent_id`),
ordenadas por `position`. O progresso do plano deriva do status das filhas.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin

TASK_STATUSES = ("pending", "done", "cancelled")


class Task(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ({})".format(", ".join(f"'{s}'" for s in TASK_STATUSES)),
            name="ck_tasks_status",
        ),
        Index("ix_tasks_user_status", "user_id", "status"),
        Index("ix_tasks_parent", "parent_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(16), default="pending", server_default="pending", nullable=False
    )
    # Ordem da etapa dentro de um plano.
    position: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
