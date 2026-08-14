"""Memória de longo prazo: persistência e recuperação (lexical e semântica).

Duas estratégias de recuperação:
- `retrieve_semantic` — similaridade de cosseno sobre embeddings. No
  PostgreSQL usa o operador nativo do pgvector (`<=>`); em outros dialetos
  (SQLite nos testes) calcula em Python.
- `retrieve_relevant` — sobreposição lexical, usada como fallback quando
  embeddings estão desabilitados ou a memória não tem vetor.

O ranking combina similaridade, importância (`relevance`) e RECÊNCIA: a
"relevância atual" de uma memória decai com o tempo desde o último uso,
com meia-vida por tipo (episódica decai rápido; semântica, devagar). Um
piso garante que memórias antigas continuem encontráveis.
"""
import logging
import math
import re
import unicodedata
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.base import ensure_aware, utcnow
from app.models.memory import MemoryEntry
from app.services.embeddings.base import cosine_similarity

logger = logging.getLogger("yui.memory_service")

# Meia-vida do decaimento, em dias, por tipo de memória.
_HALF_LIFE_DAYS: dict[str, float] = {
    "episodic": 30.0,
    "relationship": 180.0,
    "procedural": 365.0,
    "semantic": 365.0,
}
_RECENCY_FLOOR = 0.3


def recency_factor(memory: MemoryEntry, now: datetime | None = None) -> float:
    """Fator de recência (piso.. 1.0) baseado no último uso da memória."""
    now = now or utcnow()
    # Objeto ainda não persistido (sem timestamps) é tratado como recém-criado.
    raw_reference = memory.last_used_at or memory.created_at
    if raw_reference is None:
        return 1.0
    reference = ensure_aware(raw_reference)
    age_days = max(0.0, (now - reference).total_seconds() / 86_400)
    half_life = _HALF_LIFE_DAYS.get(memory.memory_type, 365.0)
    return max(_RECENCY_FLOOR, math.pow(0.5, age_days / half_life))


def memory_score(similarity: float, memory: MemoryEntry) -> float:
    """Relevância atual: similaridade × importância × recência."""
    return similarity * (0.5 + memory.relevance) * recency_factor(memory)

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


def lexical_overlap(a: str, b: str) -> float:
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
        memory_type: str = "semantic",
        embedding: list[float] | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            user_id=user_id,
            category=category,
            content=content,
            relevance=max(0.0, min(1.0, relevance)),
            confidence=max(0.0, min(1.0, confidence)),
            source=source,
            memory_type=memory_type,
            embedding=embedding,
            # Só faz sentido registrar proveniência quando há vetor de fato;
            # memórias sem embedding (fallback lexical) ficam com ambos NULL.
            embedding_provider=embedding_provider if embedding is not None else None,
            embedding_model=embedding_model if embedding is not None else None,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    @staticmethod
    def reinforce(entry: MemoryEntry, content: str, confidence: float = 1.0) -> None:
        """Consolidação: uma informação reconfirmada fica mais forte.

        Chamado quando a extração encontra uma memória equivalente à
        candidata — em vez de duplicar, reforça a existente.

        O `content` da candidata SUBSTITUI o texto salvo. Duas entradas só
        chegam aqui por serem semanticamente equivalentes (ver
        `find_duplicate`), mas "equivalente" cobre tanto uma reformulação
        (mesma informação, texto novo) quanto uma ATUALIZAÇÃO de um
        atributo com valor novo (ex.: jogo/cor/cargo favorito mudou) — os
        dois casos têm alta similaridade com o texto antigo. Manter o texto
        mais ANTIGO nesses casos é o problema: uma preferência atualizada
        ficaria salva com o valor desatualizado, e ainda mais reforçada
        (confiança/recência maiores) a cada reconfirmação — o inverso do
        pretendido. Preservar o texto mais RECENTE resolve os dois casos
        sem precisar distinguir "reformulação" de "atualização" (uma
        contradição semântica com baixa similaridade textual, abaixo do
        threshold de duplicidade, ainda não é coberta por este método —
        ver observação na auditoria).
        """
        entry.content = content
        entry.usage_count += 1
        entry.confidence = max(entry.confidence, min(1.0, confidence))
        entry.last_used_at = utcnow()

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
        """Marca as memórias como utilizadas agora (recência + frequência)."""
        now: datetime = utcnow()
        for entry in entries:
            entry.last_used_at = now
            entry.usage_count += 1

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

        candidates: list[MemoryEntry]
        if self._session.get_bind().dialect.name == "postgresql":  # pragma: no cover
            # pgvector: pré-seleciona os N*4 mais próximos no banco e refina
            # o ranking (importância) em Python.
            distance = MemoryEntry.embedding.op("<=>")(query_embedding)
            try:
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
            except DBAPIError:
                # pgvector exige que todos os vetores comparados na mesma
                # consulta tenham a MESMA dimensão. Se o provedor de
                # embeddings foi trocado sem reindexação, memórias antigas
                # (dimensão diferente) fazem o operador `<=>` levantar erro
                # para a consulta inteira. Em vez de derrubar o turno,
                # degrada para buscar todas as memórias do usuário e filtrar
                # em Python — nenhuma memória é invalidada ou apagada, só
                # ranqueada com o fallback mais lento até uma reindexação.
                await self._session.rollback()
                logger.warning(
                    "Consulta pgvector falhou (provável mistura de dimensões "
                    "de embedding após troca de provedor); usando fallback "
                    "em Python para o usuário %s.",
                    user_id,
                )
                result = await self._session.execute(
                    select(MemoryEntry).where(
                        MemoryEntry.user_id == user_id,
                        MemoryEntry.embedding.is_not(None),
                    )
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
            stored_vector = memory.embedding or []
            if len(stored_vector) != len(query_embedding):
                # Vetor com dimensão diferente do provedor atual (memória
                # criada por um provedor/modelo anterior, ainda sem
                # reindexação). Não invalida a memória — apenas a exclui
                # desta rodada de ranking semântico; ela continua
                # recuperável pela busca lexical.
                logger.debug(
                    "Memória %s ignorada no ranking semântico: dimensão %d "
                    "≠ %d da consulta atual.",
                    memory.id,
                    len(stored_vector),
                    len(query_embedding),
                )
                continue
            similarity = cosine_similarity(query_embedding, stored_vector)
            if similarity < threshold:
                continue
            scored.append((memory_score(similarity, memory), memory))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [memory for _, memory in scored[:limit]]

    async def score_candidates(
        self,
        user_id: uuid.UUID,
        query: str,
        query_embedding: list[float] | None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Candidatas (memória, similaridade) acima do corte, SEM limitar.

        Base do Attention Manager: devolve mais candidatas do que o limite
        final para que a seleção por atenção (com penalidade de redundância)
        tenha material para escolher. Usa embeddings quando há vetor da
        consulta; caso contrário, sobreposição lexical.
        """
        threshold = self._settings.memory_similarity_threshold
        memories = await self.list_by_user(user_id)
        out: list[tuple[MemoryEntry, float]] = []
        for memory in memories:
            if query_embedding is not None and memory.embedding is not None:
                similarity = cosine_similarity(query_embedding, memory.embedding)
                if similarity < threshold:
                    continue
            else:
                similarity = lexical_overlap(query, memory.content) or lexical_overlap(
                    query, memory.category
                )
                if similarity <= 0.0:
                    continue
            out.append((memory, similarity))
        return out

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
            elif lexical_overlap(content, memory.content) >= 0.8:
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
            scored.append((memory_score(float(overlap), memory), memory))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            memory for _, memory in scored[: self._settings.memory_retrieval_limit]
        ]
