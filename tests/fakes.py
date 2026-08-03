"""Dublês de teste compartilhados (Redis, provedor de IA e embeddings)."""
import hashlib
import math

from app.services.embeddings.base import EmbeddingProvider
from app.services.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolSpec


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
    """Provedor determinístico.

    Sem script: ecoa a última mensagem do usuário.
    Com script: devolve as respostas na ordem (permite roteirizar chamadas
    de ferramenta, extração de memórias, planos etc.).
    """

    def __init__(self, script: list[LLMResponse] | None = None) -> None:
        self.calls: list[tuple[str, list[ChatMessage], list[ToolSpec] | None]] = []
        self.script: list[LLMResponse] = list(script or [])

    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        self.calls.append((system_prompt, messages, tools))
        if self.script:
            return self.script.pop(0)
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        content = f"eco: {last_user.content}" if last_user else "eco:"
        return LLMResponse(
            content=content,
            model="fake-model",
            input_tokens=10,
            output_tokens=5,
        )


class FakeEmbeddings(EmbeddingProvider):
    """Embeddings determinísticos.

    Textos presentes em `mapping` usam o vetor definido pelo teste (para
    controlar similaridade); os demais recebem um vetor pseudo-aleatório
    derivado do hash do texto (quase ortogonal aos demais).
    """

    def __init__(
        self,
        mapping: dict[str, list[float]] | None = None,
        dimension: int = 8,
    ) -> None:
        self.mapping = dict(mapping or {})
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-embeddings"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        if text in self.mapping:
            return self.mapping[text]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i] / 255.0 - 0.5 for i in range(self.dimension)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]
