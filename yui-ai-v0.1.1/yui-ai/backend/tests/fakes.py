"""Dublês de teste compartilhados (Redis e provedor de IA)."""
from app.services.llm.base import ChatMessage, LLMProvider, LLMResponse


class FakePipeline:
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
                begin = start if start >= 0 else max(0, len(items) + start)
                self._store[key] = items[begin:end]


class FakeRedis:
    """Implementa o subconjunto de comandos usado pela Yui."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.kv: dict[str, int] = {}

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self.lists)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        return self.lists.get(key, [])

    async def delete(self, key: str) -> None:
        self.lists.pop(key, None)
        self.kv.pop(key, None)

    async def incr(self, key: str) -> int:
        self.kv[key] = self.kv.get(key, 0) + 1
        return self.kv[key]

    async def incrby(self, key: str, amount: int) -> int:
        self.kv[key] = self.kv.get(key, 0) + amount
        return self.kv[key]

    async def get(self, key: str) -> int | None:
        return self.kv.get(key)

    async def expire(self, key: str, ttl: int) -> bool:
        return True

    async def ping(self) -> bool:
        return True


class FakeLLM(LLMProvider):
    """Provedor determinístico: ecoa a última mensagem do usuário."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def generate(
        self, system_prompt: str, messages: list[ChatMessage]
    ) -> LLMResponse:
        self.calls.append((system_prompt, messages))
        last_user = next(m for m in reversed(messages) if m.role == "user")
        return LLMResponse(
            content=f"eco: {last_user.content}",
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
        )
