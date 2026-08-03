"""Permission System: autorização por usuário e ferramenta.

Sem registro, vale o default da ferramenta (`Tool.default_allowed`) —
ferramentas de produtividade nascem permitidas; categorias sensíveis
futuras (arquivos, sistema operacional, calendário externo) nascem
negadas até o usuário conceder explicitamente.
"""
import uuid

from sqlalchemy import Boolean, ForeignKey, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class ToolPermission(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "tool_permissions"
    __table_args__ = (
        UniqueConstraint("user_id", "tool_name", name="uq_tool_permissions_user_tool"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False)
