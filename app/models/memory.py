"""Memory System — memória de longo prazo hierárquica.

Tipos de memória (`memory_type`):
- semantic      informações permanentes ("gosta de programação");
- episodic      eventos importantes ("terminou o projeto X");
- procedural    como o usuário faz/prefere fazer tarefas;
- relationship  marcos da interação com a Yui.

A memória de trabalho (working memory) não vive aqui: é o histórico recente
(Redis) + o resumo da conversa (conversations.summary).

Cada memória carrega importância (`relevance`), confiança, origem, data,
frequência de uso (`usage_count`) e última utilização — insumos do ranking
com decaimento (services/memory_service.py) e da manutenção
(services/memory_maintenance.py).
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin
from app.models.types import EmbeddingType

MEMORY_SOURCES = ("user", "extracted")
MEMORY_TYPES = ("semantic", "episodic", "procedural", "relationship")


class MemoryEntry(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_entries"
    __table_args__ = (
        CheckConstraint(
            "memory_type IN ({})".format(", ".join(f"'{t}'" for t in MEMORY_TYPES)),
            name="ck_memory_entries_type",
        ),
        Index("ix_memory_entries_user_created", "user_id", "created_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    memory_type: Mapped[str] = mapped_column(
        String(16), default="semantic", server_default="semantic", nullable=False
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
    # Frequência de uso: reforçada quando a memória entra numa resposta ou é
    # reconfirmada pela extração (consolidação).
    usage_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
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
    # Proveniência do vetor: qual backend e modelo o geraram. Nulo para
    # memórias sem embedding e para as criadas antes desta coluna existir
    # (migration aditiva, sem backfill). Usado para diagnóstico e para
    # decidir quando uma reindexação é necessária após trocar de provedor
    # de embeddings — nunca para lógica de negócio em tempo de resposta.
    embedding_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
