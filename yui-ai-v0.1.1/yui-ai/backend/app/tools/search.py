"""Ferramenta de busca externa (web_search) — delega ao ResearchAgent.

Registrada apenas quando WEB_SEARCH_ENABLED=true.
"""
from typing import Any

from app.services.llm.base import ToolSpec
from app.tools.base import Tool, ToolContext


async def _web_search(ctx: ToolContext, args: dict[str, Any]) -> str:
    # Import tardio: evita ciclo tools -> agents -> tools.
    from app.agents.research_agent import ResearchAgent

    return await ResearchAgent().search(str(args["query"]))


def search_tools() -> list[Tool]:
    return [
        Tool(
            spec=ToolSpec(
                name="web_search",
                description=(
                    "Busca informações na web. Use para fatos recentes ou "
                    "conhecimento fora do seu treinamento."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Termos de busca."},
                    },
                    "required": ["query"],
                },
            ),
            handler=_web_search,
        ),
    ]
