from dataclasses import dataclass
from uuid import UUID

from app.application.ports.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
    ChunkEmbeddingWrite,
)
from app.application.ports.embedding_provider import (
    EmbeddingProvider,
)


@dataclass(frozen=True)
class IndexDocumentResult:
    version_id: UUID
    model_name: str
    model_revision: str
    indexed_count: int


class DocumentIndexingService:
    def __init__(
        self,
        *,
        repository: ChunkEmbeddingRepository,
        embedding_provider: EmbeddingProvider,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError(
                "Embedding batch size must be positive."
            )

        self._repository = repository
        self._embedding_provider = (
            embedding_provider
        )
        self._batch_size = batch_size

    def index_version(
        self,
        *,
        version_id: UUID,
    ) -> IndexDocumentResult:
        chunks = (
            self._repository.list_missing_chunks(
                version_id=version_id,
                model_name=(
                    self._embedding_provider.model_name
                ),
                model_revision=(
                    self._embedding_provider.model_revision
                ),
            )
        )

        if not chunks:
            return IndexDocumentResult(
                version_id=version_id,
                model_name=(
                    self._embedding_provider.model_name
                ),
                model_revision=(
                    self._embedding_provider.model_revision
                ),
                indexed_count=0,
            )

        pending: list[
            ChunkEmbeddingWrite
        ] = []

        for start in range(
            0,
            len(chunks),
            self._batch_size,
        ):
            batch = chunks[
                start:
                start + self._batch_size
            ]

            results = (
                self._embedding_provider
                .embed_passages(
                    [
                        chunk.content
                        for chunk in batch
                    ]
                )
            )

            if len(results) != len(batch):
                raise ValueError(
                    "Embedding provider returned "
                    "an unexpected result count."
                )

            for chunk, result in zip(
                batch,
                results,
                strict=True,
            ):
                if (
                    result.model_name
                    != self._embedding_provider.model_name
                ):
                    raise ValueError(
                        "Embedding model name mismatch."
                    )

                if (
                    result.model_revision
                    != self._embedding_provider.model_revision
                ):
                    raise ValueError(
                        "Embedding model revision mismatch."
                    )

                if (
                    result.dimension
                    != self._embedding_provider.dimension
                ):
                    raise ValueError(
                        "Embedding dimension mismatch."
                    )

                if (
                    len(result.values)
                    != result.dimension
                ):
                    raise ValueError(
                        "Embedding vector length mismatch."
                    )

                pending.append(
                    ChunkEmbeddingWrite(
                        chunk_id=chunk.chunk_id,
                        model_name=(
                            result.model_name
                        ),
                        model_revision=(
                            result.model_revision
                        ),
                        dimension=(
                            result.dimension
                        ),
                        normalized=True,
                        values=result.values,
                    )
                )

        self._repository.save_embeddings(
            version_id=version_id,
            embeddings=tuple(
                pending
            ),
        )

        return IndexDocumentResult(
            version_id=version_id,
            model_name=(
                self._embedding_provider.model_name
            ),
            model_revision=(
                self._embedding_provider.model_revision
            ),
            indexed_count=len(
                pending
            ),
        )
