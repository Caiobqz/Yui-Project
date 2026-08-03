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


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Um vetor produzido não tem a dimensão declarada pelo provedor.

    Tratado como falha do provedor (mesmo caminho de EmbeddingError): quem
    chama `embed()` deve degradar para o fallback lexical, nunca persistir
    ou comparar um vetor com dimensão inconsistente.
    """


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Vetoriza uma lista de textos (uma chamada em lote)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensão dos vetores produzidos por este provedor/modelo.

        Usada para validar cada vetor antes da persistência e antes de
        comparações de similaridade — nunca inferida a partir do tamanho
        de uma lista, sempre declarada explicitamente pelo provedor.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nome curto do backend (ex.: 'openai', 'sentence_transformers').

        Usado apenas para registro de proveniência (qual provedor/modelo
        gerou cada embedding persistido) — nunca para lógica de negócio.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nome do modelo em uso (ex.: 'text-embedding-3-small',
        'BAAI/bge-small-en-v1.5'). Usado apenas para proveniência."""
        raise NotImplementedError


def validate_embedding_dimension(vector: list[float], expected: int, *, context: str) -> None:
    """Levanta `EmbeddingDimensionMismatchError` se `vector` não tiver
    exatamente `expected` componentes.

    Chamado antes de qualquer persistência ou comparação de similaridade —
    nunca depois. `context` identifica onde a checagem ocorreu (mensagens
    de erro acionáveis, sem expor detalhes internos ao cliente da API).
    """
    if len(vector) != expected:
        raise EmbeddingDimensionMismatchError(
            f"Vetor de embedding com dimensão {len(vector)} não corresponde à "
            f"dimensão declarada pelo provedor ({expected}) em '{context}'. "
            "Isso geralmente indica um provedor/modelo de embeddings trocado "
            "sem reindexação das memórias existentes."
        )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similaridade de cosseno em Python puro (listas pequenas; sem numpy).

    Vetores de dimensões diferentes retornam 0.0 (nenhuma similaridade)
    em vez de levantar exceção — usado como rede de segurança adicional ao
    comparar contra memórias persistidas antes de uma eventual troca de
    provedor de embeddings sem reindexação (ver memory_service.py).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
