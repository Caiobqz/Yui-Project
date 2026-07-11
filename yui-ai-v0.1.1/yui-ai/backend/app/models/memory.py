"""Memória de longo prazo: informações autorizadas pelo usuário."""
from sqlalchemy import Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryEntry(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_entries"

    user_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    category: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    relevance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)

    # v0.2: coluna de embedding (pgvector) será adicionada via migration,
    # sem quebrar o contrato do MemoryService.
