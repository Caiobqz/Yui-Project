"""Registro de ferramentas disponíveis para o modelo."""
from collections.abc import Iterable

from app.core.config import get_settings
from app.services.llm.base import ToolSpec
from app.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {t.spec.name: t for t in tools}

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)


def build_default_registry() -> ToolRegistry:
    """Monta o conjunto padrão de ferramentas conforme a configuração."""
    # Imports locais: os módulos de ferramenta importam agentes tardiamente,
    # e o registry é montado uma vez por request (barato — só dataclasses).
    from app.tools.memories import memory_tools
    from app.tools.notes import note_tools
    from app.tools.planner import planner_tools
    from app.tools.tasks import task_tools

    tools: list[Tool] = [
        *task_tools(),
        *note_tools(),
        *memory_tools(),
        *planner_tools(),
    ]
    if get_settings().web_search_enabled:
        from app.tools.search import search_tools

        tools.extend(search_tools())
    return ToolRegistry(tools)
