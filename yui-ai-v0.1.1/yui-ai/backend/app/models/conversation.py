"""Conversas e mensagens persistidas (histórico completo — fonte de verdade)."""
import uuid

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin

VALID_ROLES = ("user", "assistant")


class Conversation(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "conversations"
    __table_args__ = (
        # Listagens de conversas do usuário ordenadas por data.
        Index("ix_conversations_user_created", "user_id", "created_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.sequence",
    )


class Message(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ({})".format(", ".join(f"'{r}'" for r in VALID_ROLES)),
            name="ck_messages_role",
        ),
        # Ordenação determinística do turno: timestamps podem colidir quando o
        # par usuário/assistente é gravado no mesmo flush; sequence não.
        Index(
            "ix_messages_conversation_sequence",
            "conversation_id",
            "sequence",
            unique=True,
        ),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Posição da mensagem dentro da conversa (1, 2, 3, ...), atribuída na
    # persistência do turno sob lock da linha da conversa.
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
