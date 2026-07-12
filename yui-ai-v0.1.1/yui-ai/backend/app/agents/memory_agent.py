"""Memory Agent — cria, recupera e deduplica memórias sobre o usuário.

Criação por dois caminhos:
- `remember` — pedido explícito (ferramenta save_memory);
- `extract_from_turn` — extração automática pós-turno: um LLM analisa o par
  usuário/assistente e propõe memórias com categoria, importância e
  confiança; o Guardian faz a triagem e a deduplicação usa embeddings.

Recuperação: semântica (embeddings) quando disponível, lexical como
fallback. Memórias usadas numa resposta têm `last_used_at` atualizado.
"""
import json
import logging
import re
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.guardian import GuardianAgent
from app.core.config import get_settings
from app.models.memory import MemoryEntry
from app.services.embeddings.base import EmbeddingProvider
from app.services.llm.base import ChatMessage, LLMProvider
from app.services.memory_service import MemoryService
from app.services.usage_service import build_usage_record

logger = logging.getLogger("yui.memory_agent")

_EXTRACTION_SYSTEM_PROMPT = """Você é o componente de memória da Yui, uma assistente pessoal.
Analise o turno de conversa e extraia APENAS informações duráveis e importantes \
sobre o usuário: objetivos, preferências, interesses, conhecimentos e hábitos relevantes.

NÃO extraia: saudações, pedidos pontuais, opiniões da assistente, trivialidades, \
nem dados sensíveis (senhas, tokens, documentos, dados financeiros).

Responda SOMENTE com JSON válido, sem texto adicional, no formato:
{"memories": [{"content": "...", "category": "...", "importance": 0.0, "confidence": 0.0}]}

- content: frase curta e autossuficiente sobre o usuário (ex.: "Quer trabalhar com IA").
- category: uma palavra (ex.: objetivos, preferencias, interesses, habitos, conhecimento).
- importance e confidence: números entre 0 e 1.
Se não houver nada que valha a pena lembrar: {"memories": []}"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_extraction(text: str) -> list[dict[str, Any]]:
    """Extrai a lista de memórias do JSON retornado pelo modelo (tolerante)."""
    candidate = text.strip()
    fence = _JSON_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        logger.debug("Extração de memória: resposta não é JSON válido.")
        return []
    memories = data.get("memories") if isinstance(data, dict) else None
    if not isinstance(memories, list):
        return []
    return [m for m in memories if isinstance(m, dict)]


def _clamp(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


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
        """Vetoriza textos; None quando embeddings estão desabilitados/falham."""
        if self._embeddings is None or not texts:
            return None
        try:
            return await self._embeddings.embed(texts)
        except Exception:
            logger.warning(
                "Provedor de embeddings falhou; usando fallback lexical.",
                exc_info=True,
            )
            return None

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

    # --- Criação explícita ---------------------------------------------------------

    async def remember(
        self,
        user_id: uuid.UUID,
        content: str,
        category: str = "geral",
        importance: float = 0.7,
        confidence: float = 1.0,
        source: str = "user",
    ) -> MemoryEntry | None:
        """Salva uma memória com triagem e deduplicação. None se rejeitada."""
        rejection = self._guardian.screen_memory_content(content)
        if rejection is not None:
            logger.info("Memória rejeitada pelo Guardian: %s.", rejection)
            return None

        vectors = await self.embed_texts([content])
        embedding = vectors[0] if vectors else None

        async with self._session_factory() as session:
            service = MemoryService(session)
            duplicate = await service.find_duplicate(user_id, content, embedding)
            if duplicate is not None:
                logger.info("Memória duplicada ignorada (existente %s).", duplicate.id)
                return None
            entry = await service.create(
                user_id,
                category=category,
                content=content.strip(),
                relevance=importance,
                confidence=confidence,
                source=source,
                embedding=embedding,
            )
            await session.commit()
            return entry

    # --- Extração automática pós-turno ---------------------------------------------

    async def extract_from_turn(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_text: str,
        assistant_text: str,
    ) -> int:
        """Analisa o turno e salva memórias novas. Retorna quantas salvou."""
        settings = get_settings()
        turn = f"Usuário: {user_text}\nYui: {assistant_text}"
        response = await self._llm.generate(
            _EXTRACTION_SYSTEM_PROMPT,
            [ChatMessage(role="user", content=turn)],
        )

        saved = 0
        for candidate in _parse_extraction(response.content):
            content = str(candidate.get("content") or "").strip()
            confidence = _clamp(candidate.get("confidence"), default=0.0)
            if not content or confidence < settings.memory_min_confidence:
                continue
            entry = await self.remember(
                user_id,
                content=content,
                category=str(candidate.get("category") or "geral")[:64],
                importance=_clamp(candidate.get("importance"), default=0.5),
                confidence=confidence,
                source="extracted",
            )
            if entry is not None:
                saved += 1

        # Contabiliza a chamada de extração (custo real, mesmo sem memórias).
        async with self._session_factory() as session:
            session.add(build_usage_record(user_id, conversation_id, response))
            await session.commit()

        if saved:
            logger.info(
                "Extração: %d memória(s) nova(s) para o usuário %s.", saved, user_id
            )
        return saved
