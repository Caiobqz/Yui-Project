"""Superfícies read-only do núcleo autônomo (v0.4).

- GET /self             Self Model (identidade, capacidades, saúde) — read-only.
- GET /metrics          Observabilidade (contadores/amostras agregados).
- GET /initiatives      Ações autônomas aprovadas pela Bússola Moral.
- GET /knowledge-graph  Grafo de entidades derivado do usuário.

Todas exigem autenticação. Nenhuma modifica o Self Model — ele é imutável.
"""
from typing import Any

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession
from app.cognition.knowledge_graph import build_knowledge_graph
from app.cognition.self_model import get_self_model
from app.core.metrics import METRICS
from app.services.initiative_service import InitiativeService

router = APIRouter(tags=["system"])


@router.get("/self")
async def self_model(_user: CurrentUser) -> dict[str, Any]:
    return get_self_model().to_dict()


@router.get("/metrics")
async def metrics(_user: CurrentUser) -> dict[str, Any]:
    return METRICS.snapshot()


@router.get("/initiatives")
async def initiatives(user: CurrentUser, session: DbSession) -> list[dict[str, Any]]:
    """Iniciativas registradas da Yui (v0.5: persistidas, não recomputadas).

    A geração acontece no ciclo cognitivo dela (pós-turno) — a rota apenas
    observa o registro. Campos da v0.4 preservados; `decision` é sempre
    "proceed" porque só aprovadas são registradas.
    """
    records = await InitiativeService(session).list_recent(user.id)
    return [
        {
            "id": str(r.id),
            "kind": r.kind,
            "description": r.description,
            "decision": "proceed",
            "confidence": r.confidence,
            "score": r.score,
            "rationale": r.rationale,
            "status": r.status,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]


@router.get("/knowledge-graph")
async def knowledge_graph(user: CurrentUser, session: DbSession) -> dict[str, Any]:
    graph = await build_knowledge_graph(session, user.id)
    return graph.to_dict()
