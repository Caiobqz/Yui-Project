"""Ferramenta de memória explícita (save_memory).

Usada quando o usuário PEDE para a Yui lembrar de algo. A extração
automática pós-turno é responsabilidade do MemoryAgent; esta ferramenta
cobre o pedido explícito, com a mesma triagem do Guardian e deduplicação.
"""
from typing import Any

from app.services.llm.base import ToolSpec
from app.tools.base import Tool, ToolContext


async def _save_memory(ctx: ToolContext, args: dict[str, Any]) -> str:
    # Import tardio: evita ciclo tools -> agents -> tools.
    from app.agents.memory_agent import MemoryAgent
    from app.services.embeddings.factory import get_embedding_provider

    agent = MemoryAgent(
        llm=ctx.llm,
        embeddings=get_embedding_provider(),
        session_factory=ctx.session_factory,
    )
    saved = await agent.remember(
        user_id=ctx.user_id,
        content=str(args["content"]),
        category=str(args.get("category") or "geral"),
        importance=float(args.get("importance") or 0.7),
    )
    if saved is None:
        return (
            "A informação não foi salva: ou já existe uma memória equivalente, "
            "ou o conteúdo contém dados sensíveis que não devem ser armazenados."
        )
    return f"Memória salva ([{saved.category}] {saved.content})"


def memory_tools() -> list[Tool]:
    return [
        Tool(
            spec=ToolSpec(
                name="save_memory",
                description=(
                    "Salva uma informação importante e duradoura sobre o usuário "
                    "(objetivo, preferência, interesse, hábito). Use quando o "
                    "usuário pedir explicitamente para você lembrar de algo."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "A informação, em frase curta."},
                        "category": {"type": "string", "description": "Categoria (ex.: estudos, saude, trabalho)."},
                        "importance": {
                            "type": "number",
                            "description": "Importância de 0 a 1 (padrão 0.7).",
                        },
                    },
                    "required": ["content"],
                },
            ),
            handler=_save_memory,
        ),
    ]
