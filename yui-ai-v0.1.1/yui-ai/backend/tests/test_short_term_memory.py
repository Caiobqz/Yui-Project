"""Testes da memória de curto prazo com um Redis simulado."""
from app.memory.short_term import ShortTermMemory
from app.services.llm.base import ChatMessage
from tests.fakes import FakeRedis


async def test_append_many_writes_pair_in_single_pipeline() -> None:
    redis = FakeRedis()
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


async def test_history_respects_max_messages() -> None:
    redis = FakeRedis()
    stm = ShortTermMemory(redis)  # type: ignore[arg-type]
    stm._max_messages = 4  # noqa: SLF001 — configuração de teste

    for i in range(6):
        await stm.append("conv1", ChatMessage(role="user", content=f"msg {i}"))

    history = await stm.get_history("conv1")
    assert len(history) == 4
    assert history[0].content == "msg 2"
    assert history[-1].content == "msg 5"


async def test_append_many_with_empty_list_is_noop() -> None:
    redis = FakeRedis()
    stm = ShortTermMemory(redis)  # type: ignore[arg-type]
    await stm.append_many("conv1", [])
    assert await stm.get_history("conv1") == []
