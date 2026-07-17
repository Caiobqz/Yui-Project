"""Yui Core — orquestra o fluxo cognitivo de cada turno.

Fluxo cognitivo (Reasoning Engine):

    Entrada
      → contexto emocional (heurística, sem custo)
      → rate limit + embedding da consulta          (fase 0, sem banco)
      → conversa + memórias relevantes (com decaimento) + modelo do usuário
        (adaptação/relacionamento) + permissões + curiosidade + histórico
        (rehidratação) + resumo                     (fase 1, conexão curta)
      → estratégia → system prompt (CognitiveState)
      → loop agêntico: LLM ⇄ ferramentas
        (Guardian valida + permissões; TaskAgent executa)  (fase 2, SEM banco)
      → persistência do turno + interação + uso + estado afetivo +
        entrega de iniciativa                       (fase 3, conexão curta)
      → pós-turno em background (modelo utilitário): TurnAnalyzer propõe
        memórias (Memory System) e adaptação (Adaptation Engine);
        manutenção de memória periódica; resumo de conversas longas;
        geração de iniciativas (determinística, Judgement Engine).

Este módulo não conhece HTTP nem SDKs de IA: apenas coordena serviços/agentes.
"""
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.guardian import GuardianAgent
from app.agents.memory_agent import MemoryAgent
from app.agents.task_agent import TaskAgent
from app.cognition.affect import AffectService
from app.cognition.analyzer import TurnAnalyzer
from app.cognition.curiosity import CuriosityEngine
from app.cognition.emotional_context import (
    EmotionalContext,
)
from app.cognition.emotional_context import (
    analyze as analyze_emotional_context,
)
from app.cognition.goal_engine import GoalEngine
from app.cognition.self_model import build_self_model
from app.cognition.user_model import UserModelService
from app.core.background import spawn
from app.core.config import get_settings
from app.core.exceptions import ConversationNotFoundError
from app.core.metrics import METRICS
from app.memory.short_term import ShortTermMemory
from app.models.base import utcnow
from app.models.conversation import Conversation, Message
from app.services.context_orchestrator import ContextOrchestrator
from app.services.embeddings.base import EmbeddingProvider
from app.services.history_service import HistoryService, prepare_for_llm
from app.services.initiative_service import InitiativeService
from app.services.llm.base import ChatMessage, LLMProvider, LLMResponse
from app.services.memory_maintenance import MemoryMaintenance
from app.services.permission_service import PermissionService
from app.services.rate_limiter import RateLimiter
from app.services.summary_service import ConversationSummarizer
from app.services.usage_service import build_usage_record
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry

logger = logging.getLogger("yui.core")

_FALLBACK_REPLY = (
    "Não consegui concluir a ação dentro do limite de etapas. Pode reformular?"
)


@dataclass(frozen=True)
class YuiReply:
    conversation_id: uuid.UUID
    content: str
    model: str
    memories_used: int


@dataclass(frozen=True)
class _TurnSetup:
    conversation_id: uuid.UUID
    system_prompt: str
    messages: list[ChatMessage]
    memories_used: int
    emotional: EmotionalContext
    permission_overrides: dict[str, bool] = field(default_factory=dict)
    # Iniciativa própria pendente injetada neste turno (marcada como
    # entregue na fase 3 — a Yui oferece o assunto uma vez, nunca insiste).
    initiative_id: uuid.UUID | None = None


class YuiCore:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        short_term: ShortTermMemory,
        llm: LLMProvider,
        rate_limiter: RateLimiter,
        embeddings: EmbeddingProvider | None = None,
        registry: ToolRegistry | None = None,
        utility_llm: LLMProvider | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._short_term = short_term
        self._history = HistoryService(short_term)
        self._llm = llm
        # Trabalho cognitivo de bastidor roda num modelo mais barato.
        self._utility_llm = utility_llm or llm
        self._rate_limiter = rate_limiter
        self._registry = registry or ToolRegistry()
        self._guardian = GuardianAgent()
        self._task_agent = TaskAgent(self._registry)
        self._memory_agent = MemoryAgent(llm, embeddings, session_factory)
        self._curiosity = CuriosityEngine()
        self._analyzer = TurnAnalyzer(self._utility_llm)
        self._summarizer = ConversationSummarizer(self._utility_llm, session_factory)
        self._maintenance = MemoryMaintenance(session_factory)
        # Context Orchestrator: único montador do prompt de companhia (v0.4).
        # O Goal Engine é compartilhado: a análise de objetivos roda UMA vez
        # por turno e alimenta curiosidade, atenção e contexto (v0.5).
        self._self_model = build_self_model(self._registry)
        self._goal_engine = GoalEngine()
        self._orchestrator = ContextOrchestrator(
            self._memory_agent, self._self_model, goal_engine=self._goal_engine
        )

    # ------------------------------------------------------------------ turno

    async def process_message(
        self,
        user_id: uuid.UUID,
        plan: str,
        text: str,
        conversation_id: uuid.UUID | None = None,
    ) -> YuiReply:
        settings = get_settings()
        await self._rate_limiter.enforce(user_id, plan)

        setup = await self._prepare_turn(user_id, text, conversation_id)

        # Fase 2 — loop agêntico, sem conexão de banco aberta.
        responses: list[LLMResponse] = []
        messages = list(setup.messages)
        ctx = self._tool_context(user_id, setup.conversation_id)
        tools = self._registry.specs() or None
        final_text = _FALLBACK_REPLY
        for _ in range(settings.llm_max_tool_iterations):
            response = await self._llm.generate(
                setup.system_prompt, messages, tools=tools
            )
            responses.append(response)
            if not response.tool_calls:
                final_text = response.content
                break
            await self._handle_tool_calls(ctx, setup, messages, response)

        await self._finalize_turn(user_id, setup, text, final_text, responses)
        self._schedule_post_turn(user_id, setup.conversation_id, text, final_text)

        return YuiReply(
            conversation_id=setup.conversation_id,
            content=final_text,
            model=responses[-1].model,
            memories_used=setup.memories_used,
        )

    async def stream_message(
        self,
        user_id: uuid.UUID,
        plan: str,
        text: str,
        conversation_id: uuid.UUID | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Mesmo fluxo do process_message, emitindo eventos em tempo real.

        Eventos: {"type": "delta", "text"} | {"type": "tool", "tools"} |
        {"type": "done", ...}. Erros são tratados pela rota (SSE).
        """
        settings = get_settings()
        await self._rate_limiter.enforce(user_id, plan)

        setup = await self._prepare_turn(user_id, text, conversation_id)

        responses: list[LLMResponse] = []
        messages = list(setup.messages)
        ctx = self._tool_context(user_id, setup.conversation_id)
        tools = self._registry.specs() or None
        final_text = _FALLBACK_REPLY
        for _ in range(settings.llm_max_tool_iterations):
            response: LLMResponse | None = None
            async for chunk in self._llm.generate_stream(
                setup.system_prompt, messages, tools=tools
            ):
                if chunk.delta:
                    yield {"type": "delta", "text": chunk.delta}
                if chunk.response is not None:
                    response = chunk.response
            if response is None:
                raise RuntimeError("Streaming terminou sem resposta consolidada.")
            responses.append(response)
            if not response.tool_calls:
                final_text = response.content
                break
            yield {
                "type": "tool",
                "tools": [call.name for call in response.tool_calls],
            }
            await self._handle_tool_calls(ctx, setup, messages, response)

        await self._finalize_turn(user_id, setup, text, final_text, responses)
        self._schedule_post_turn(user_id, setup.conversation_id, text, final_text)

        yield {
            "type": "done",
            "conversation_id": str(setup.conversation_id),
            "model": responses[-1].model,
            "memories_used": setup.memories_used,
        }

    # -------------------------------------------------------------- pós-turno

    async def run_post_turn(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Análise cognitiva pós-turno (fora do caminho da resposta).

        Uma chamada do modelo utilitário produz memórias candidatas e notas
        de adaptação; em seguida rodam a manutenção periódica da memória e o
        resumo de conversas longas. Falhas são logadas, nunca propagadas.
        """
        settings = get_settings()
        interaction_count = 0

        if settings.memory_extraction_enabled:
            try:
                analysis = await self._analyzer.analyze(user_text, assistant_text)
                await self._memory_agent.store_candidates(user_id, analysis.memories)
                async with self._session_factory() as session:
                    profile = await UserModelService(session).get_or_create(user_id)
                    if analysis.adaptation:
                        UserModelService.apply_adaptation(profile, analysis.adaptation)
                    interaction_count = profile.interaction_count
                    if analysis.response is not None:
                        session.add(
                            build_usage_record(
                                user_id, conversation_id, analysis.response
                            )
                        )
                    await session.commit()
            except Exception:
                logger.exception("Análise cognitiva pós-turno falhou.")

            # Manutenção periódica: esquecimento de memórias obsoletas.
            interval = settings.memory_maintenance_interval
            if interval > 0 and interaction_count > 0 and interaction_count % interval == 0:
                try:
                    await self._maintenance.run(user_id)
                except Exception:
                    logger.exception("Manutenção de memória falhou.")

        if settings.summarization_enabled:
            try:
                await self._summarizer.maybe_summarize(user_id, conversation_id)
            except Exception:
                logger.exception("Sumarização falhou (pós-turno).")

        # Iniciativas (v0.5): julga situações atuais e registra as aprovadas
        # — determinístico, sem LLM; cooldown e teto garantem raridade.
        if settings.initiative_generation_enabled and settings.autonomy_enabled:
            try:
                async with self._session_factory() as session:
                    await InitiativeService(session).generate(user_id)
                    await session.commit()
            except Exception:
                logger.exception("Geração de iniciativas falhou (pós-turno).")

    def _schedule_post_turn(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_text: str,
        assistant_text: str,
    ) -> None:
        settings = get_settings()
        if not (
            settings.memory_extraction_enabled
            or settings.summarization_enabled
            or (settings.initiative_generation_enabled and settings.autonomy_enabled)
        ):
            return
        spawn(
            self.run_post_turn(user_id, conversation_id, user_text, assistant_text),
            name=f"post-turn:{conversation_id}",
        )

    # ------------------------------------------------------------------ fases

    async def _prepare_turn(
        self,
        user_id: uuid.UUID,
        text: str,
        conversation_id: uuid.UUID | None,
    ) -> _TurnSetup:
        settings = get_settings()
        started = time.monotonic()

        # Contexto emocional: heurística pura, sem custo.
        emotional = analyze_emotional_context(text)

        # Fase 0 — embedding da consulta ANTES de abrir a sessão.
        query_embedding = await self._memory_agent.embed_query(text)

        # Fase 1 — leitura do estado cognitivo (conexão curta). O Context
        # Orchestrator monta o prompt de companhia (atenção + objetivos +
        # world/self model), o único caminho de montagem de contexto.
        async with self._session_factory() as session:
            conversation = await self._resolve_conversation(
                session, user_id, conversation_id
            )
            resolved_id = conversation.id
            summary = conversation.summary

            profile = await UserModelService(session).get_or_create(user_id)
            adaptation_notes = list(profile.preferences or [])
            relationship = UserModelService.relationship_line(profile)
            permission_overrides = await PermissionService(session).overrides(user_id)
            goal_states = await self._goal_engine.analyze(session, user_id)
            curiosity = await self._curiosity.suggest(
                session, user_id, profile, goal_states=goal_states
            )
            history = await self._history.load(session, resolved_id)

            # v0.5 — estado afetivo persistente e iniciativa própria pendente.
            affect = (
                await AffectService(session).snapshot(user_id)
                if settings.affect_enabled
                else None
            )
            pending_initiative = (
                await InitiativeService(session).pending_for_turn(user_id)
                if settings.autonomy_enabled
                else None
            )

            system_prompt, memories_used = await self._orchestrator.build(
                session=session,
                user_id=user_id,
                text=text,
                query_embedding=query_embedding,
                summary=summary,
                adaptation_notes=adaptation_notes,
                relationship=relationship,
                emotional=emotional,
                curiosity=curiosity,
                goal_states=goal_states,
                affect=affect,
                initiative=(
                    pending_initiative.description
                    if pending_initiative is not None
                    else None
                ),
            )
            if curiosity is not None:
                # Registra a sugestão para o espaçamento da curiosidade
                # (a conversa nunca vira interrogatório).
                profile.last_curiosity_interaction = profile.interaction_count
            await session.commit()

        METRICS.observe("reasoning.ms", (time.monotonic() - started) * 1000)
        METRICS.incr("turn.processed")

        user_message = ChatMessage(role="user", content=text)
        messages = prepare_for_llm(
            [*history, user_message], settings.llm_max_history_chars
        )
        return _TurnSetup(
            conversation_id=resolved_id,
            system_prompt=system_prompt,
            messages=messages,
            memories_used=memories_used,
            emotional=emotional,
            permission_overrides=permission_overrides,
            initiative_id=(
                pending_initiative.id if pending_initiative is not None else None
            ),
        )

    def _tool_context(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> ToolContext:
        return ToolContext(
            user_id=user_id,
            conversation_id=conversation_id,
            session_factory=self._session_factory,
            llm=self._llm,
        )

    async def _handle_tool_calls(
        self,
        ctx: ToolContext,
        setup: _TurnSetup,
        messages: list[ChatMessage],
        response: LLMResponse,
    ) -> None:
        """Valida (Guardian + permissões), executa (TaskAgent) e devolve
        os resultados ao modelo."""
        messages.append(
            ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )
        for call in response.tool_calls:
            error = self._guardian.validate_tool_call(
                self._registry, call, setup.permission_overrides
            )
            result = error if error is not None else await self._task_agent.execute(ctx, call)
            messages.append(
                ChatMessage(
                    role="tool",
                    content=self._guardian.clamp_tool_result(result),
                    tool_call_id=call.id,
                )
            )

    async def _finalize_turn(
        self,
        user_id: uuid.UUID,
        setup: _TurnSetup,
        user_text: str,
        assistant_text: str,
        responses: list[LLMResponse],
    ) -> None:
        settings = get_settings()
        conversation_id = setup.conversation_id
        # Fase 3 — persistência (conexão curta).
        async with self._session_factory() as session:
            await self._persist_turn(
                session, user_id, conversation_id, user_text, assistant_text, responses
            )
            # Relationship Model: registra a interação (UPDATE atômico —
            # turnos concorrentes do mesmo usuário não perdem incrementos).
            await UserModelService(session).get_or_create(user_id)
            await UserModelService.register_interaction(session, user_id)
            # Affective State (v0.5): os eventos do turno atualizam o estado
            # persistente — determinístico e barato (1 UPDATE na mesma conexão).
            if settings.affect_enabled:
                await AffectService(session).register_turn(user_id, setup.emotional)
            # Iniciativa injetada neste turno foi entregue: a Yui ofereceu o
            # assunto uma vez e não vai insistir.
            if setup.initiative_id is not None:
                await InitiativeService(session).mark_delivered(
                    user_id, setup.initiative_id
                )
            await session.commit()

        # Cache de curto prazo (par gravado atomicamente) e contador de tokens.
        await self._short_term.append_many(
            str(conversation_id),
            [
                ChatMessage(role="user", content=user_text),
                ChatMessage(role="assistant", content=assistant_text),
            ],
        )
        total_tokens = sum(
            (r.input_tokens or 0) + (r.output_tokens or 0) for r in responses
        )
        await self._rate_limiter.register_tokens(user_id, total_tokens)

    async def _resolve_conversation(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID | None,
    ) -> Conversation:
        """Valida a conversa informada ou cria uma nova.

        Conversa inexistente OU de outro usuário → 404 (a resposta não revela
        se o id existe).
        """
        if conversation_id is not None:
            conversation = await session.get(Conversation, conversation_id)
            if conversation is None or conversation.user_id != user_id:
                raise ConversationNotFoundError(
                    f"Conversa {conversation_id} não encontrada."
                )
            return conversation

        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        await session.flush()
        return conversation

    async def _persist_turn(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_text: str,
        assistant_text: str,
        responses: list[LLMResponse],
    ) -> None:
        # Lock da linha da conversa: serializa turnos concorrentes na mesma
        # conversa e garante sequence sem colisão (no SQLite dos testes o
        # FOR UPDATE é no-op — aceitável, o banco é single-writer).
        result = await session.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .with_for_update()
        )
        conversation = result.scalar_one()

        last_sequence = (
            await session.scalar(
                select(func.max(Message.sequence)).where(
                    Message.conversation_id == conversation_id
                )
            )
        ) or 0

        session.add_all(
            [
                Message(
                    conversation_id=conversation_id,
                    sequence=last_sequence + 1,
                    role="user",
                    content=user_text,
                ),
                Message(
                    conversation_id=conversation_id,
                    sequence=last_sequence + 2,
                    role="assistant",
                    content=assistant_text,
                ),
            ]
        )
        conversation.updated_at = utcnow()
        for response in responses:
            session.add(build_usage_record(user_id, conversation_id, response))
