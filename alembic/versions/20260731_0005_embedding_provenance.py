"""Yui v0.5.1 — Proveniência de embeddings.

- memory_entries.embedding_provider / embedding_model: qual backend e
  modelo geraram cada vetor persistido. Colunas NULLABLE, migration
  puramente aditiva — memórias existentes ficam com NULL (sem embedding
  conhecido de proveniência), nada é reescrito ou invalidado.

Motivação: a introdução de provedores de embeddings locais (llama.cpp
embeddings, sentence-transformers) ao lado dos comerciais torna possível
trocar de provedor/modelo ao longo do tempo. Sem este registro, não há
como saber quais memórias precisam de reindexação após uma troca.

Revision ID: 0005_embedding_provenance
Revises: 0004_companion_core
Create Date: 2026-07-31
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_embedding_provenance"
down_revision: str | None = "0004_companion_core"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "memory_entries",
        sa.Column("embedding_provider", sa.String(32), nullable=True),
    )
    op.add_column(
        "memory_entries",
        sa.Column("embedding_model", sa.String(128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("memory_entries", "embedding_model")
    op.drop_column("memory_entries", "embedding_provider")
