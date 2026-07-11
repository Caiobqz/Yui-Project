"""Testes da memória de curto prazo com um Redis simulado."""

import pytest

from app.memory.short_term import ShortTermMemory
from app.services.llm.base import ChatMessage


class _FakePipeline:
    def __init__(self, store: dict[str, list[str]]) -> None:
        self._store = store
        self._ops: list[tuple] = []

    def rpush(self, key: str, *values: str) -> None:
        self._ops.append(("rpush", key, values))

    def ltrim(self, key: str, start: int, end: int) -> None:
        self._ops.append(("ltrim", key, start, end))

    def expire(self, key: str, ttl: int) -> None:
        self._ops.append(("expire", key, ttl))

    async def execute(self) -> None:
        for op in self._ops:
            if op[0] == "rpush":
                self._store.setdefault(op[1], []).extend(op[2])
            elif op[0] == "ltrim":
                _, key, start, end = op
                items = self._store.get(key, [])
                end = len(items) if end == -1 else end + 1
                self._store[key] = items[start if start >= 0 else max(0, len(items) + start): end]


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, list[str]] = {}

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self.store)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        return self.store.get(key, [])

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


@pytest.mark.asyncio
async def test_append_many_writes_pair_in_single_pipeline() -> None:
    redis = _FakeRedis()
    stm = ShortTermMemory(redis)  # type: ignore[arg-type]

    await stm.append_many(
        "conv1",
        [
            ChatMessage(role="user", content="Oi"),
            ChatMessage(role="assistant", content="Olá!"),
        ],
    )

    history = await stm.get_history("conv1")
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[1].content == "Olá!"


@pytest.mark.asyncio
async def test_history_respects_max_messages() -> None:
    redis = _FakeRedis()
    stm = ShortTermMemory(redis)  # type: ignore[arg-type]
    stm._max_messages = 4  # noqa: SLF001 — configuração de teste

    for i in range(6):
        await stm.append("conv1", ChatMessage(role="user", content=f"msg {i}"))

    history = await stm.get_history("conv1")
    assert len(history) == 4
    assert history[0].content == "msg 2"
    assert history[-1].content == "msg 5"


@pytest.mark.asyncio
async def test_append_many_with_empty_list_is_noop() -> None:
    redis = _FakeRedis()
    stm = ShortTermMemory(redis)  # type: ignore[arg-type]
    await stm.append_many("conv1", [])
    assert await stm.get_history("conv1") == []
