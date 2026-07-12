"""Memória de longo prazo: informações autorizadas pelo usuário (PostgreSQL).

Na v0.1.x a recuperação usa sobreposição lexical simples ponderada pela
relevância da memória. A assinatura de `retrieve_relevant` foi desenhada
para que a v0.2 troque a implementação por embeddings + busca vetorial
(pgvector) sem alterar quem consome o serviço.

Nota de escala: carrega todas as memórias do usuário para pontuar em
memória. Aceitável para volumes pessoais; a busca vetorial da v0.2 move o
ranking para o banco.
"""
import re
import unicodedata
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.memory import MemoryEntry

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


class MemoryService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._retrieval_limit = get_settings().memory_retrieval_limit

    async def create(
        self,
        user_id: uuid.UUID,
        category: str,
        content: str,
        relevance: float = 0.5,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            user_id=user_id,
            category=category,
            content=content,
            relevance=max(0.0, min(1.0, relevance)),
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

    async def retrieve_relevant(
        self, user_id: uuid.UUID, query: str
    ) -> list[MemoryEntry]:
        """Retorna as memórias mais relacionadas à mensagem do usuário."""
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
            score = overlap * (0.5 + memory.relevance)
            scored.append((score, memory))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [memory for _, memory in scored[: self._retrieval_limit]]
