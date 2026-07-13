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
