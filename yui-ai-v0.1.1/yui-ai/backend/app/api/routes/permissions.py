"""Permission System: consulta e concessão/revogação de ferramentas."""
from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DbSession
from app.api.schemas import PermissionResponse, PermissionUpdateRequest
from app.services.permission_service import PermissionService
from app.tools.registry import build_default_registry

router = APIRouter(prefix="/permissions", tags=["permissions"])


@router.get("")
async def list_permissions(
    user: CurrentUser, session: DbSession
) -> list[PermissionResponse]:
    registry = build_default_registry()
    entries = await PermissionService(session).effective(user.id, registry)
    return [PermissionResponse.model_validate(e) for e in entries]


@router.put("/{tool_name}")
async def set_permission(
    tool_name: str,
    payload: PermissionUpdateRequest,
    user: CurrentUser,
    session: DbSession,
) -> PermissionResponse:
    registry = build_default_registry()
    tool = registry.get(tool_name)
    if tool is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ferramenta desconhecida.",
        )
    await PermissionService(session).set_permission(
        user.id, tool_name, payload.allowed
    )
    return PermissionResponse(
        tool_name=tool_name,
        category=tool.category,
        allowed=payload.allowed,
        source="user",
    )
