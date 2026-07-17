"""Context Orchestrator — o ÚNICO montador de contexto de companhia.

Nenhum outro módulo monta o prompt de conversa diretamente. O orquestrador
reúne, na ordem determinística do fluxo cognitivo:

    Attention Manager → Memory → Goals → Relationship → World Model →
    Self Model → Identity → Prompt Builder → LLM

Cada bloco possui orçamento próprio de caracteres (proxy determinístico de
tokens), aplicado antes da renderização. Prompts UTILITÁRIOS (resumo,
extração) não são contexto de companhia e seguem fora deste fluxo, por
design — este orquestrador governa o que a Yui envia ao conversar.
"""
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.memory_agent import MemoryAgent
from app.cognition.attention import AttentionContext, AttentionManager
from app.cognition.curiosity import CuriosityHint
from app.cognition.emotional_context import EmotionalContext
from app.cognition.goal_engine import GoalEngine
from app.cognition.knowledge_graph import build_knowledge_graph
from app.cognition.reasoning import CognitiveState
from app.cognition.self_model import SelfModel
from app.cognition.world_model import build_world_model
from app.core.config import get_settings
from app.core.metrics import METRICS
from app.models.memory import MemoryEntry
from app.services.context_service import build_system_prompt
from app.services.memory_service import MemoryService


def _clip(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 1)].rstrip() + "…"


class ContextOrchestrator:
    def __init__(
        self,
        memory_agent: MemoryAgent,
        self_model: SelfModel,
        attention: AttentionManager | None = None,
        goal_engine: GoalEngine | None = None,
    ) -> None:
        self._memory_agent = memory_agent
        self._self_model = self_model
        self._attention = attention or AttentionManager()
        self._goals = goal_engine or GoalEngine()

    async def select_memories(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        text: str,
        query_embedding: list[float] | None,
        attention_ctx: AttentionContext,
    ) -> list[MemoryEntry]:
        """Recupera candidatas e aplica o Attention Manager (contexto já pronto)."""
        service = MemoryService(session)
        candidates = await service.score_candidates(user_id, text, query_embedding)
        selected = self._attention.select(candidates, attention_ctx)
        for scored in selected:
            METRICS.observe("attention.score", scored.score)
        METRICS.observe("memory.retrieved", len(selected))
        entries = [s.memory for s in selected]
        await service.touch(entries)
        return entries

    async def build(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        text: str,
        query_embedding: list[float] | None,
        summary: str | None,
        adaptation_notes: list[str],
        relationship: str | None,
        emotional: EmotionalContext,
        curiosity: CuriosityHint | None,
    ) -> tuple[str, int]:
        """Monta o system prompt de companhia. Retorna (prompt, memórias_usadas)."""
        settings = get_settings()

        # Estado dos objetivos calculado UMA vez por turno e reaproveitado
        # pela atenção, pelo grafo e pelo bloco de objetivos (antes: 3× por turno).
        goal_states = await self._goals.analyze(session, user_id)
        goal_terms = GoalEngine.active_terms(goal_states)
        graph_terms: set[str] = set()
        if goal_terms:
            graph = await build_knowledge_graph(session, user_id)
            goal_labels = {
                s.title for s in goal_states if s.status in ("active", "stalled")
            }
            graph_terms = graph.related_terms(goal_labels)

        memories = await self.select_memories(
            session,
            user_id,
            text,
            query_embedding,
            AttentionContext(goal_terms=goal_terms, graph_terms=graph_terms),
        )
        goal_lines = [
            f"{s.title}: {s.done}/{s.total} etapas ({_status_pt(s.status)})"
            + (f"; próxima: {s.next_step}" if s.next_step else "")
            for s in goal_states
        ]
        world = build_world_model(self._self_model)

        state = CognitiveState(
            memories=_clip_memories(memories, settings.ctx_budget_memories_chars),
            summary=_clip(summary, settings.ctx_budget_summary_chars) if summary else None,
            adaptation_notes=_clip_lines(
                adaptation_notes, settings.ctx_budget_adaptation_chars
            ),
            relationship=relationship,
            emotional=emotional,
            curiosity=curiosity,
            self_prompt=_clip(self._self_model.to_prompt(), settings.ctx_budget_self_chars),
            environment_prompt=_clip(
                world.environment_prompt(), settings.ctx_budget_world_chars
            ),
            general_boundary=world.general_boundary(),
            goals=_clip_lines(goal_lines, settings.ctx_budget_goals_chars),
        )
        return build_system_prompt(state), len(memories)


def _status_pt(status: str) -> str:
    return {
        "active": "ativo",
        "stalled": "parado",
        "abandoned": "abandonado",
        "completed": "concluído",
    }.get(status, status)


def _clip_lines(lines: list[str], budget: int) -> list[str]:
    """Mantém as primeiras linhas que couberem no orçamento."""
    kept: list[str] = []
    total = 0
    for line in lines:
        if kept and total + len(line) > budget:
            break
        kept.append(line)
        total += len(line)
    return kept


def _clip_memories(memories: list[MemoryEntry], budget: int) -> list[MemoryEntry]:
    kept: list[MemoryEntry] = []
    total = 0
    for memory in memories:
        size = len(memory.content) + len(memory.category)
        if kept and total + size > budget:
            break
        kept.append(memory)
        total += size
    return kept
