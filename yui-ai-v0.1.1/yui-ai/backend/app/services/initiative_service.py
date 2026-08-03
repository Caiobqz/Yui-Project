"""Iniciativas autônomas persistentes (v0.5) — o registro de vontade da Yui.

O Judgement Engine decide O QUE fazer; este serviço governa QUANDO e SE a
decisão vira presença, garantindo as propriedades comportamentais da
companheira:

- raras          teto de iniciativas pendentes por usuário;
- nunca repetitivas   cooldown por `dedupe_key` (mesma situação não é
                 reproposta dentro da janela, mesmo se já entregue);
- oportunas      a entrega acontece no próximo momento natural de conversa
                 (o Context Orchestrator injeta a pendente mais antiga como
                 diretiva; o modelo decide se o momento é adequado);
- com ciclo fechado   registro → entrega → métricas (Aprendizado do fluxo
                 Percepção→…→Execução→Aprendizado).

A geração roda no pós-turno (fora do caminho da resposta) e é 100%
determinística — nenhuma chamada de LLM.
"""
import uuid
from datetime import timedelta

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.cognition.affect import AffectService
from app.cognition.judgement import JudgementEngine
from app.cognition.user_model import UserModelService
from app.core.config import get_settings
from app.core.metrics import METRICS
from app.models.base import utcnow
from app.models.initiative import InitiativeRecord


class InitiativeService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def generate(self, user_id: uuid.UUID) -> int:
        """Julga situações atuais e registra iniciativas aprovadas.

        Retorna quantas foram registradas. Aplica vínculo (Relationship
        Model) e preocupação (Affective State) ao julgamento, o teto de
        pendentes e o cooldown por situação.
        """
        settings = get_settings()
        if not settings.autonomy_enabled:
            return 0

        pending = await self._session.scalar(
            select(func.count())
            .select_from(InitiativeRecord)
            .where(
                InitiativeRecord.user_id == user_id,
                InitiativeRecord.status == "pending",
            )
        )
        pending = pending or 0
        if pending >= settings.initiative_max_pending:
            return 0

        profile = await UserModelService(self._session).get_or_create(user_id)
        affect = await AffectService(self._session).snapshot(user_id)
        proposals = await JudgementEngine().propose_initiatives(
            self._session,
            user_id,
            relationship=UserModelService.relationship_strength(profile),
            concern=affect.concern,
        )
        if not proposals:
            return 0

        # Situações em cooldown (janela recente, entregues ou não) e
        # pendentes de qualquer idade nunca são repropostas.
        cutoff = utcnow() - timedelta(days=settings.initiative_cooldown_days)
        rows = await self._session.execute(
            select(InitiativeRecord.dedupe_key).where(
                InitiativeRecord.user_id == user_id,
                or_(
                    InitiativeRecord.created_at >= cutoff,
                    InitiativeRecord.status == "pending",
                ),
            )
        )
        blocked_keys = {key for (key,) in rows}

        created = 0
        for proposal in proposals:
            key = proposal.dedupe_key or proposal.kind
            if key in blocked_keys:
                continue
            if pending + created >= settings.initiative_max_pending:
                break
            self._session.add(
                InitiativeRecord(
                    user_id=user_id,
                    kind=proposal.kind,
                    dedupe_key=key,
                    description=proposal.description,
                    score=proposal.evaluation.score,
                    confidence=proposal.evaluation.confidence,
                    rationale=proposal.evaluation.rationale,
                )
            )
            blocked_keys.add(key)
            created += 1
        if created:
            await self._session.flush()
            METRICS.incr("initiative.recorded", by=created)
        return created

    async def pending_for_turn(self, user_id: uuid.UUID) -> InitiativeRecord | None:
        """A pendente mais antiga — no máximo UMA iniciativa por turno."""
        return await self._session.scalar(
            select(InitiativeRecord)
            .where(
                InitiativeRecord.user_id == user_id,
                InitiativeRecord.status == "pending",
            )
            .order_by(InitiativeRecord.created_at)
            .limit(1)
        )

    async def mark_delivered(
        self, user_id: uuid.UUID, initiative_id: uuid.UUID
    ) -> None:
        """Marca a iniciativa como entregue (UPDATE idempotente).

        Marcada mesmo que o modelo escolha não mencioná-la: a oportunidade
        foi dada e a Yui não insiste — prudência acima de cobrança.
        """
        await self._session.execute(
            update(InitiativeRecord)
            .where(
                InitiativeRecord.id == initiative_id,
                InitiativeRecord.user_id == user_id,
                InitiativeRecord.status == "pending",
            )
            .values(status="delivered", delivered_at=utcnow())
        )
        METRICS.incr("initiative.delivered")

    async def list_recent(
        self, user_id: uuid.UUID, limit: int = 20
    ) -> list[InitiativeRecord]:
        result = await self._session.execute(
            select(InitiativeRecord)
            .where(InitiativeRecord.user_id == user_id)
            .order_by(InitiativeRecord.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars())
