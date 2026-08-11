"""Memory Agent — cria, recupera, consolida e deduplica memórias.

Criação por dois caminhos:
- `remember` — pedido explícito (ferramenta save_memory ou API);
- `store_candidates` — candidatas propostas pelo TurnAnalyzer no pós-turno.

Consolidação: quando uma candidata equivale a uma memória existente
(similaridade de embedding ou lexical), a existente é REFORÇADA
(usage_count, confiança, recência) em vez de duplicada — informações
reconfirmadas ficam mais fortes; as demais decaem (ver memory_service).

Recuperação: semântica (embeddings) quando disponível, lexical como
fallback; memórias usadas têm recência e frequência atualizadas.
"""
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.guardian import GuardianAgent
from app.cognition.analyzer import MemoryCandidate
from app.core.config import get_settings
from app.models.memory import MemoryEntry
from app.services.embeddings.base import EmbeddingProvider, validate_embedding_dimension
from app.services.llm.base import LLMProvider
from app.services.memory_service import MemoryService

logger = logging.getLogger("yui.memory_agent")


class MemoryAgent:
    def __init__(
        self,
        llm: LLMProvider,
        embeddings: EmbeddingProvider | None,
        session_factory: async_sessionmaker[AsyncSession],
        guardian: GuardianAgent | None = None,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._session_factory = session_factory
        self._guardian = guardian or GuardianAgent()

    # --- Embeddings -------------------------------------------------------------

    async def embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """Vetoriza textos; None quando embeddings estão desabilitados/falham.

        Único ponto de produção de embeddings do sistema — cobre tanto a
        escrita (via `_store`) quanto a consulta (via `embed_query`). Cada
        vetor tem sua dimensão validada aqui, antes de qualquer persistência
        ou comparação de similaridade: um vetor com dimensão inesperada é
        tratado exatamente como qualquer outra falha do provedor (log +
        degradação para o fallback lexical), nunca propagado adiante.
        """
        if self._embeddings is None or not texts:
            return None
        try:
            vectors = await self._embeddings.embed(texts)
            for vector in vectors:
                validate_embedding_dimension(
                    vector,
                    self._embeddings.dimension,
                    context="MemoryAgent.embed_texts",
                )
        except Exception:
            logger.warning(
                "Provedor de embeddings falhou; usando fallback lexical.",
                exc_info=True,
            )
            return None
        return vectors

    async def embed_query(self, text: str) -> list[float] | None:
        vectors = await self.embed_texts([text])
        return vectors[0] if vectors else None

    # --- Recuperação -------------------------------------------------------------

    async def retrieve(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        query_text: str,
        query_embedding: list[float] | None,
    ) -> list[MemoryEntry]:
        service = MemoryService(session)
        if query_embedding is not None:
            entries = await service.retrieve_semantic(user_id, query_embedding)
        else:
            entries = await service.retrieve_relevant(user_id, query_text)
        await service.touch(entries)
        return entries

    # --- Criação e consolidação -----------------------------------------------------

    async def remember(
        self,
        user_id: uuid.UUID,
        content: str,
        category: str = "geral",
        importance: float = 0.7,
        confidence: float = 1.0,
        source: str = "user",
        memory_type: str = "semantic",
    ) -> MemoryEntry | None:
        """Salva (ou reforça) uma memória. None se rejeitada pelo Guardian."""
        entry, _created = await self._store(
            user_id,
            content=content,
            category=category,
            importance=importance,
            confidence=confidence,
            source=source,
            memory_type=memory_type,
        )
        return entry

    async def store_candidates(
        self, user_id: uuid.UUID, candidates: list[MemoryCandidate]
    ) -> int:
        """Persiste candidatas do TurnAnalyzer. Retorna quantas são NOVAS.

        Candidatas equivalentes a memórias existentes reforçam a existente
        (consolidação) e não contam como novas.
        """
        settings = get_settings()
        created_count = 0
        for candidate in candidates:
            if candidate.confidence < settings.memory_min_confidence:
                continue
            _entry, created = await self._store(
                user_id,
                content=candidate.content,
                category=candidate.category,
                importance=candidate.importance,
                confidence=candidate.confidence,
                source="extracted",
                memory_type=candidate.memory_type,
            )
            if created:
                created_count += 1
        if created_count:
            logger.info(
                "Memória: %d nova(s) para o usuário %s.", created_count, user_id
            )
        return created_count

    async def _store(
        self,
        user_id: uuid.UUID,
        content: str,
        category: str,
        importance: float,
        confidence: float,
        source: str,
        memory_type: str,
    ) -> tuple[MemoryEntry | None, bool]:
        """Retorna (memória criada/reforçada ou None, foi_criada)."""
        rejection = self._guardian.screen_memory_content(content)
        if rejection is not None:
            logger.info("Memória rejeitada pelo Guardian: %s.", rejection)
            return None, False

        vectors = await self.embed_texts([content])
        embedding = vectors[0] if vectors else None
        # Proveniência só é conhecida (e só faz sentido registrar) quando o
        # provedor de embeddings existe E produziu um vetor com sucesso —
        # nunca quando caiu para o fallback lexical.
        embedding_provider = self._embeddings.provider_name if embedding is not None and self._embeddings else None
        embedding_model = self._embeddings.model_name if embedding is not None and self._embeddings else None

        async with self._session_factory() as session:
            service = MemoryService(session)
            duplicate = await service.find_duplicate(user_id, content, embedding)
            if duplicate is not None:
                service.reinforce(duplicate, content.strip(), confidence)
                await session.commit()
                logger.debug("Memória consolidada (reforço em %s).", duplicate.id)
                return duplicate, False
            entry = await service.create(
                user_id,
                category=category,
                content=content.strip(),
                relevance=importance,
                confidence=confidence,
                source=source,
                memory_type=memory_type,
                embedding=embedding,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )
            await session.commit()
            return entry, True
