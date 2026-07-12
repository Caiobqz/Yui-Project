"""Memória de longo prazo: informações sobre o usuário.

Cada memória carrega conteúdo, categoria, importância (`relevance`),
confiança da extração, origem (usuário ou extração automática), embedding
para busca semântica e a data da última utilização em uma resposta.
"""
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin
from app.models.types import EmbeddingType

MEMORY_SOURCES = ("user", "extracted")


class MemoryEntry(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_entries"
    __table_args__ = (
        Index("ix_memory_entries_user_created", "user_id", "created_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    category: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Importância da memória (0..1) — pondera o ranking de recuperação.
    relevance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    # Confiança na informação (1.0 para memórias criadas pelo próprio usuário).
    confidence: Mapped[float] = mapped_column(
        Float, default=1.0, server_default="1.0", nullable=False
    )
    # "user" (criada explicitamente) ou "extracted" (extração automática).
    source: Mapped[str] = mapped_column(
        String(16), default="user", server_default="user", nullable=False
    )
    # Última vez que a memória foi usada em uma resposta.
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Vetor semântico (pgvector no PostgreSQL). Nulo quando embeddings estão
    # desabilitados — a recuperação cai para busca lexical.
    embedding: Mapped[list[float] | None] = mapped_column(
        EmbeddingType, nullable=True
    )
