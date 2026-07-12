"""Memória de longo prazo: persistência e recuperação (lexical e semântica).

Duas estratégias de recuperação:
- `retrieve_semantic` — similaridade de cosseno sobre embeddings. No
  PostgreSQL usa o operador nativo do pgvector (`<=>`); em outros dialetos
  (SQLite nos testes) calcula em Python.
- `retrieve_relevant` — sobreposição lexical, usada como fallback quando
  embeddings estão desabilitados ou a memória não tem vetor.

O ranking combina similaridade com a importância da memória (`relevance`).
"""
import re
import unicodedata
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.base import utcnow
from app.models.memory import MemoryEntry
from app.services.embeddings.base import cosine_similarity

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MIN_TOKEN_LENGTH = 3  # tokens menores raramente carregam significado

# Palavras muito comuns em PT-BR que não carregam significado para o ranking.
_STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "em", "no", "na",
    "nos", "nas", "um", "uma", "que", "e", "ou", "para", "por", "com", "se",
    "eu", "voce", "você", "meu", "minha", "como", "esta", "está", "é", "ser",
}


def _normalize(text: str) -> set[str]:
    """Minúsculas, sem acentos, sem stopwords e sem tokens curtos."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return {
        w
        for w in _WORD_RE.findall(text)
        if w not in _STOPWORDS and len(w) >= _MIN_TOKEN_LENGTH
    }


def _lexical_overlap(a: str, b: str) -> float:
    """Similaridade de Jaccard entre os conjuntos de tokens normalizados."""
    ta, tb = _normalize(a), _normalize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class MemoryService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._settings = get_settings()

    async def create(
        self,
        user_id: uuid.UUID,
        category: str,
        content: str,
        relevance: float = 0.5,
        confidence: float = 1.0,
        source: str = "user",
        embedding: list[float] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            user_id=user_id,
            category=category,
            content=content,
            relevance=max(0.0, min(1.0, relevance)),
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            embedding=embedding,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def list_by_user(self, user_id: uuid.UUID) -> list[MemoryEntry]:
        result = await self._session.execute(
            select(MemoryEntry)
            .where(MemoryEntry.user_id == user_id)
            .order_by(MemoryEntry.created_at.desc())
        )
        return list(result.scalars().all())

    async def delete(self, user_id: uuid.UUID, memory_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            select(MemoryEntry).where(
                MemoryEntry.id == memory_id, MemoryEntry.user_id == user_id
            )
        )
        entry = result.scalar_one_or_none()
        if entry is None:
            return False
        await self._session.delete(entry)
        return True

    async def touch(self, entries: list[MemoryEntry]) -> None:
        """Marca as memórias como utilizadas agora (last_used_at)."""
        now: datetime = utcnow()
        for entry in entries:
            entry.last_used_at = now

    # --- Recuperação semântica ------------------------------------------------

    async def retrieve_semantic(
        self, user_id: uuid.UUID, query_embedding: list[float]
    ) -> list[MemoryEntry]:
        """Memórias mais próximas do vetor da consulta.

        Ranking: cosseno × (0.5 + importância), com corte mínimo de
        similaridade para não injetar contexto irrelevante no prompt.
        """
        limit = self._settings.memory_retrieval_limit
        threshold = self._settings.memory_similarity_threshold

        if self._session.get_bind().dialect.name == "postgresql":  # pragma: no cover
            # pgvector: pré-seleciona os N*4 mais próximos no banco e refina
            # o ranking (importância) em Python.
            distance = MemoryEntry.embedding.op("<=>")(query_embedding)
            result = await self._session.execute(
                select(MemoryEntry)
                .where(
                    MemoryEntry.user_id == user_id,
                    MemoryEntry.embedding.is_not(None),
                )
                .order_by(distance)
                .limit(limit * 4)
            )
            candidates = list(result.scalars())
        else:
            result = await self._session.execute(
                select(MemoryEntry).where(
                    MemoryEntry.user_id == user_id,
                    MemoryEntry.embedding.is_not(None),
                )
            )
            candidates = list(result.scalars())

        scored: list[tuple[float, MemoryEntry]] = []
        for memory in candidates:
            similarity = cosine_similarity(query_embedding, memory.embedding or [])
            if similarity < threshold:
                continue
            scored.append((similarity * (0.5 + memory.relevance), memory))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [memory for _, memory in scored[:limit]]

    async def find_duplicate(
        self,
        user_id: uuid.UUID,
        content: str,
        embedding: list[float] | None,
    ) -> MemoryEntry | None:
        """Retorna uma memória existente equivalente à candidata, se houver.

        Com embedding: cosseno acima de `memory_duplicate_threshold`.
        Sem embedding: Jaccard lexical acima de 0.8.
        """
        existing = await self.list_by_user(user_id)
        for memory in existing:
            if embedding is not None and memory.embedding is not None:
                if (
                    cosine_similarity(embedding, memory.embedding)
                    >= self._settings.memory_duplicate_threshold
                ):
                    return memory
            elif _lexical_overlap(content, memory.content) >= 0.8:
                return memory
        return None

    # --- Recuperação lexical (fallback) ----------------------------------------

    async def retrieve_relevant(
        self, user_id: uuid.UUID, query: str
    ) -> list[MemoryEntry]:
        """Fallback por sobreposição lexical, ponderada pela importância."""
        memories = await self.list_by_user(user_id)
        if not memories:
            return []

        query_terms = _normalize(query)
        scored: list[tuple[float, MemoryEntry]] = []
        for memory in memories:
            terms = _normalize(memory.content) | _normalize(memory.category)
            overlap = len(query_terms & terms)
            if overlap == 0:
                continue
            scored.append((overlap * (0.5 + memory.relevance), memory))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            memory for _, memory in scored[: self._settings.memory_retrieval_limit]
        ]
