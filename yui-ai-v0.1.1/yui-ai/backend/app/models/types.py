"""Tipos de coluna portáveis entre PostgreSQL e SQLite (testes)."""
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator


class EmbeddingType(TypeDecorator):
    """Vetor de embedding: pgvector no PostgreSQL, JSON nos demais dialetos.

    No PostgreSQL habilita busca por similaridade nativa (operador `<=>`);
    no SQLite dos testes o vetor é armazenado como JSON e a similaridade é
    calculada em Python.
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector())
        return dialect.type_descriptor(JSON())

    def process_result_value(
        self, value: Any, dialect: Dialect
    ) -> list[float] | None:
        if value is None:
            return None
        # pgvector pode devolver ndarray; JSON devolve list — normaliza.
        return [float(x) for x in value]
