"""Contrato abstrato para provedores de embeddings.

Embeddings alimentam a memória semântica (RAG): memórias e consultas são
vetorizadas e comparadas por similaridade de cosseno, permitindo recuperar
"quero trabalhar com IA" a partir de "qual área devo estudar?" mesmo sem
palavras em comum.
"""
import math
from abc import ABC, abstractmethod


class EmbeddingError(RuntimeError):
    """Erro de comunicação com o provedor de embeddings."""


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Vetoriza uma lista de textos (uma chamada em lote)."""
        raise NotImplementedError


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similaridade de cosseno em Python puro (listas pequenas; sem numpy)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
