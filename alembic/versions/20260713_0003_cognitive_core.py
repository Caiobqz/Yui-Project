"""Yui v0.3 — Cognitive Core: memória hierárquica, perfil do usuário e permissões.

- memory_entries: memory_type (semantic/episodic/procedural/relationship)
  e usage_count (frequência de uso / consolidação).
- user_profiles: Adaptation Engine (notas aprendidas) + Relationship Model
  (contagem e data das interações).
- tool_permissions: Permission System (autorização por usuário/ferramenta).

Revision ID: 0003_cognitive_core
Revises: 0002_semantic_memory
Create Date: 2026-07-13
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_cognitive_core"
down_revision: str | None = "0002_semantic_memory"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "memory_entries",
        sa.Column(
            "memory_type", sa.String(16), server_default="semantic", nullable=False
        ),
    )
    op.add_column(
        "memory_entries",
        sa.Column("usage_count", sa.Integer(), server_default="0", nullable=False),
    )

    op.create_table(
        "user_profiles",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("preferences", sa.JSON(), nullable=False),
        sa.Column(
            "interaction_count", sa.Integer(), server_default="0", nullable=False
        ),
        sa.Column("last_interaction_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", name="uq_user_profiles_user"),
    )

    op.create_table(
        "tool_permissions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool_name", sa.String(64), nullable=False),
        sa.Column("allowed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint(
            "user_id", "tool_name", name="uq_tool_permissions_user_tool"
        ),
    )


def downgrade() -> None:
    op.drop_table("tool_permissions")
    op.drop_table("user_profiles")
    op.drop_column("memory_entries", "usage_count")
    op.drop_column("memory_entries", "memory_type")
