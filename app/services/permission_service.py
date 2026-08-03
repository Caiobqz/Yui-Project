"""Permission System — autorização por usuário e ferramenta.

Resolução: registro explícito do usuário > default da ferramenta
(`Tool.default_allowed`). Ferramentas de produtividade nascem permitidas;
categorias sensíveis futuras (arquivos, sistema operacional, calendário
externo) devem registrar `default_allowed=False` — negadas até o usuário
conceder via API.
"""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool_permission import ToolPermission
from app.tools.registry import ToolRegistry


class PermissionService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def overrides(self, user_id: uuid.UUID) -> dict[str, bool]:
        """Decisões explícitas do usuário, por nome de ferramenta."""
        result = await self._session.execute(
            select(ToolPermission).where(ToolPermission.user_id == user_id)
        )
        return {p.tool_name: p.allowed for p in result.scalars()}

    async def set_permission(
        self, user_id: uuid.UUID, tool_name: str, allowed: bool
    ) -> ToolPermission:
        permission = await self._session.scalar(
            select(ToolPermission).where(
                ToolPermission.user_id == user_id,
                ToolPermission.tool_name == tool_name,
            )
        )
        if permission is None:
            permission = ToolPermission(
                user_id=user_id, tool_name=tool_name, allowed=allowed
            )
            self._session.add(permission)
        else:
            permission.allowed = allowed
        await self._session.flush()
        return permission

    async def effective(
        self, user_id: uuid.UUID, registry: ToolRegistry
    ) -> list[dict[str, object]]:
        """Visão efetiva das permissões (default + decisões do usuário)."""
        overrides = await self.overrides(user_id)
        entries: list[dict[str, object]] = []
        for spec in registry.specs():
            tool = registry.get(spec.name)
            assert tool is not None
            allowed = overrides.get(spec.name, tool.default_allowed)
            entries.append(
                {
                    "tool_name": spec.name,
                    "category": tool.category,
                    "allowed": allowed,
                    "source": "user" if spec.name in overrides else "default",
                }
            )
        return entries
