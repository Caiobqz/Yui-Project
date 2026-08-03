"""Yui v0.5 — Companion Core: afeto persistente e iniciativas.

- affective_states: dimensões afetivas computacionais por usuário
  (warmth/joy/concern), persistentes entre conversas.
- initiatives: registro e entrega das ações autônomas aprovadas pela
  Bússola Moral (dedupe/cooldown fecham o ciclo do Judgement Engine).
- user_profiles.last_curiosity_interaction: espaçamento determinístico das
  perguntas de curiosidade.

Revision ID: 0004_companion_core
Revises: 0003_cognitive_core
Create Date: 2026-07-17
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_companion_core"
down_revision: str | None = "0003_cognitive_core"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "affective_states",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("warmth", sa.Float(), server_default="0.1", nullable=False),
        sa.Column("joy", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("concern", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", name="uq_affective_states_user"),
    )

    op.create_table(
        "initiatives",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("dedupe_key", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("rationale", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), server_default="pending", nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'delivered', 'dismissed')",
            name="ck_initiatives_status",
        ),
    )
    op.create_index("ix_initiatives_user_status", "initiatives", ["user_id", "status"])
    op.create_index("ix_initiatives_user_key", "initiatives", ["user_id", "dedupe_key"])

    op.add_column(
        "user_profiles",
        sa.Column("last_curiosity_interaction", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_profiles", "last_curiosity_interaction")
    op.drop_index("ix_initiatives_user_key", table_name="initiatives")
    op.drop_index("ix_initiatives_user_status", table_name="initiatives")
    op.drop_table("initiatives")
    op.drop_table("affective_states")
