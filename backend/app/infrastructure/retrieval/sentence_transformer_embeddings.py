from typing import Any, Sequence

from sentence_transformers import (
    SentenceTransformer,
)

from app.application.ports.embedding_provider import (
    EmbeddingProvider,
    EmbeddingResult,
)


DEFAULT_MODEL_NAME = (
    "intfloat/multilingual-e5-base"
)

DEFAULT_MODEL_REVISION = (
    "d128750597153bb5987e10b1c3493a34e5a4502a"
)

DEFAULT_DIMENSION = 768


class SentenceTransformerEmbeddingProvider(
    EmbeddingProvider
):
    """Local embedding adapter for GovKnowledge.

    Model-specific behavior and provenance remain
    behind the application-owned EmbeddingProvider
    boundary.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str = DEFAULT_MODEL_REVISION,
        model: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_revision = model_revision

        self._model = (
            model
            if model is not None
            else SentenceTransformer(
                model_name,
                revision=model_revision,
            )
        )

        dimension = (
            self._model.get_embedding_dimension()
        )

        if dimension is None:
            raise ValueError(
                "Embedding model did not report "
                "its vector dimension."
            )

        self._dimension = int(
            dimension
        )

        if (
            model_name == DEFAULT_MODEL_NAME
            and self._dimension
            != DEFAULT_DIMENSION
        ):
            raise ValueError(
                "Unexpected multilingual-e5-base "
                f"dimension: {self._dimension}."
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingResult:
        cleaned = text.strip()

        if not cleaned:
            raise ValueError(
                "Query text cannot be empty."
            )

        vectors = self._encode(
            [
                f"query: {cleaned}",
            ]
        )

        return self._result(
            vectors[0]
        )

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingResult, ...]:
        if not texts:
            return ()

        prepared: list[str] = []

        for text in texts:
            cleaned = text.strip()

            if not cleaned:
                raise ValueError(
                    "Passage text cannot be empty."
                )

            prepared.append(
                f"passage: {cleaned}"
            )

        vectors = self._encode(
            prepared
        )

        if len(vectors) != len(prepared):
            raise ValueError(
                "Embedding provider returned "
                "an unexpected vector count."
            )

        return tuple(
            self._result(vector)
            for vector in vectors
        )

    def _encode(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        encoded = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vectors = encoded.tolist()

        for vector in vectors:
            if len(vector) != self._dimension:
                raise ValueError(
                    "Embedding vector has "
                    "unexpected dimension."
                )

        return vectors

    def _result(
        self,
        values: list[float],
    ) -> EmbeddingResult:
        return EmbeddingResult(
            values=tuple(
                float(value)
                for value in values
            ),
            model_name=self._model_name,
            model_revision=(
                self._model_revision
            ),
            dimension=self._dimension,
        )
