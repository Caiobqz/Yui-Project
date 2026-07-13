"""Base declarativa e colunas comuns."""
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_aware(dt: datetime) -> datetime:
    """Normaliza datetimes para UTC-aware.

    O SQLite (testes) devolve datetimes naive mesmo em colunas
    timezone=True; aritmética com utcnow() exige ambos aware.
    """
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class UUIDMixin:
    # sqlalchemy.Uuid é portável: UUID nativo no PostgreSQL, CHAR no SQLite
    # (usado nos testes de integração).
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )


class TimestampMixin:
    # default (ORM) + server_default: inserts fora do ORM também recebem valor.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        server_default=func.now(),
        nullable=False,
    )
