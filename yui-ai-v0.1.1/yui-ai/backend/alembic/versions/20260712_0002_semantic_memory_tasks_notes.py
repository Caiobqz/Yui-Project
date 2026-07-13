"""Yui v0.2: memória semântica (pgvector), resumo de conversas, tarefas e notas.

- memory_entries: embedding (vector no PostgreSQL, JSON nos demais),
  confidence, source, last_used_at.
- conversations: summary + summary_up_to_sequence (compactação de contexto).
- Novas tabelas: tasks (planos/etapas) e notes.

Revision ID: 0002_semantic_memory
Revises: 0001_initial_schema
Create Date: 2026-07-12
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002_semantic_memory"
down_revision: str | None = "0001_initial_schema"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    is_postgres = op.get_bind().dialect.name == "postgresql"

    # --- Memória semântica ---------------------------------------------------
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        embedding_type: sa.types.TypeEngine = Vector()
    else:
        embedding_type = sa.JSON()

    op.add_column(
        "memory_entries", sa.Column("embedding", embedding_type, nullable=True)
    )
    op.add_column(
        "memory_entries",
        sa.Column("confidence", sa.Float(), server_default="1.0", nullable=False),
    )
    op.add_column(
        "memory_entries",
        sa.Column("source", sa.String(16), server_default="user", nullable=False),
    )
    op.add_column(
        "memory_entries",
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )

    # --- Compactação de contexto --------------------------------------------
    op.add_column("conversations", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column(
        "conversations",
        sa.Column(
            "summary_up_to_sequence",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
        ),
    )

    # --- Tarefas / planos -----------------------------------------------------
    op.create_table(
        "tasks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_id",
            sa.Uuid(),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(16), server_default="pending", nullable=False),
        sa.Column("position", sa.Integer(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'done', 'cancelled')", name="ck_tasks_status"
        ),
    )
    op.create_index("ix_tasks_user_status", "tasks", ["user_id", "status"])
    op.create_index("ix_tasks_parent", "tasks", ["parent_id"])

    # --- Notas -----------------------------------------------------------------
    op.create_table(
        "notes",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_notes_user_created", "notes", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_table("notes")
    op.drop_table("tasks")
    op.drop_column("conversations", "summary_up_to_sequence")
    op.drop_column("conversations", "summary")
    op.drop_column("memory_entries", "last_used_at")
    op.drop_column("memory_entries", "source")
    op.drop_column("memory_entries", "confidence")
    op.drop_column("memory_entries", "embedding")
