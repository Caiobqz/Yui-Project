"""Provedor de embeddings local via sentence-transformers (offline após o
primeiro download do modelo).

`SentenceTransformer.encode()` é síncrono e pode ser intensivo em CPU —
roda em thread dedicada (`asyncio.to_thread`), mesma disciplina usada em
LlamaCppProvider, para não bloquear o event loop.

Dimensão medida por uma inferência de sondagem no `__init__` (uma string
curta) em vez de depender de um nome de método específico da biblioteca
(que mudou entre versões) — garante a dimensão REAL do modelo carregado,
não uma suposição.
"""
import asyncio

from app.core.config import Settings
from app.services.embeddings.base import (
    EmbeddingError,
    EmbeddingProvider,
    validate_embedding_dimension,
)

# String curta e neutra usada apenas para medir a dimensão de saída do
# modelo carregado — nunca persistida, nunca comparada a nada.
_DIMENSION_PROBE_TEXT = "_yui_dimension_probe_"


class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        # Import tardio: evita exigir `sentence-transformers` (e o torch,
        # que traz consigo) quando outro backend de embeddings está em uso.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=sentence_transformers exige o pacote "
                "'sentence-transformers' instalado (extra opcional: "
                "pip install -e '.[local-embeddings]')."
            ) from exc

        self._model_name = settings.embedding_model
        try:
            # device=... é repassado sem fallback silencioso: se o operador
            # configurar 'cuda' sem CUDA disponível, a biblioteca/torch
            # levanta um erro claro em vez de degradar para CPU sem aviso.
            self._model = SentenceTransformer(
                self._model_name, device=settings.embedding_device
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Falha ao carregar o modelo de embeddings "
                f"'{self._model_name}' (device='{settings.embedding_device}'): "
                f"{exc}. Verifique o nome do modelo, a conectividade (o "
                "primeiro uso exige download do Hugging Face Hub) e se o "
                "device configurado está disponível nesta máquina."
            ) from exc

        try:
            probe = self._model.encode([_DIMENSION_PROBE_TEXT])
        except Exception as exc:
            raise EmbeddingError(
                f"Falha ao medir a dimensão do modelo '{self._model_name}': {exc}"
            ) from exc
        self._dimension = len(probe[0])

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "sentence_transformers"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            raw = await asyncio.to_thread(self._model.encode, texts)
        except Exception as exc:
            raise EmbeddingError(
                f"Erro ao gerar embeddings com '{self._model_name}': {exc}"
            ) from exc
        vectors = [[float(x) for x in row] for row in raw]
        for vector in vectors:
            validate_embedding_dimension(
                vector,
                self._dimension,
                context=f"SentenceTransformersProvider/{self._model_name}",
            )
        return vectors
