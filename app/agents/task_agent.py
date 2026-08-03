"""Task Agent — executa chamadas de ferramenta já validadas pelo Guardian.

Falhas de execução nunca derrubam o turno: viram texto de erro devolvido ao
modelo, que decide como comunicar ao usuário.
"""
import logging

from app.services.llm.base import ToolCall
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry

logger = logging.getLogger("yui.task_agent")


class TaskAgent:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, ctx: ToolContext, call: ToolCall) -> str:
        tool = self._registry.get(call.name)
        if tool is None:  # defensivo — o Guardian valida antes
            return f"Ferramenta desconhecida: '{call.name}'."
        try:
            return await tool.handler(ctx, call.arguments)
        except Exception:
            logger.exception(
                "Falha ao executar a ferramenta '%s' (usuário %s).",
                call.name,
                ctx.user_id,
            )
            return (
                f"A ferramenta '{call.name}' falhou ao executar. "
                "Informe o usuário e sugira tentar novamente."
            )
