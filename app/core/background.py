"""Execução de trabalho pós-resposta (fire-and-forget com referência forte).

Usado para extração de memórias e sumarização: rodam depois que a resposta
foi entregue, sem segurar a requisição. Referências fortes evitam que o GC
cancele tasks em andamento; exceções são logadas, nunca propagadas.
"""
import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger("yui.background")

_tasks: set[asyncio.Task[Any]] = set()


def spawn(coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task[Any]:
    task = asyncio.create_task(coro, name=name)
    _tasks.add(task)
    task.add_done_callback(_on_done)
    return task


def _on_done(task: asyncio.Task[Any]) -> None:
    _tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Tarefa de background '%s' falhou.", task.get_name(), exc_info=exc)


async def shutdown(timeout: float = 10.0) -> None:
    """Drena tarefas pendentes no desligamento da aplicação.

    Sem isso, o lifespan fecharia engine/Redis com análises pós-turno ainda
    em andamento, gerando erros de conexão no shutdown.
    """
    pending = {t for t in _tasks if not t.done()}
    if not pending:
        return
    logger.info("Aguardando %d tarefa(s) de background no shutdown...", len(pending))
    _done, still_pending = await asyncio.wait(pending, timeout=timeout)
    for task in still_pending:
        logger.warning("Cancelando tarefa de background '%s'.", task.get_name())
        task.cancel()
