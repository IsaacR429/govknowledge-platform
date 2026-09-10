from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class ChunkForEmbedding:
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    chunk_index: int
    content: str


@dataclass(frozen=True)
class ChunkEmbeddingWrite:
    chunk_id: UUID
    model_name: str
    model_revision: str
    dimension: int
    normalized: bool
    values: tuple[float, ...]


class ChunkEmbeddingRepository(Protocol):
    def list_missing_chunks(
        self,
        *,
        version_id: UUID,
        model_name: str,
        model_revision: str,
    ) -> tuple[ChunkForEmbedding, ...]:
        ...

    def save_embeddings(
        self,
        *,
        version_id: UUID,
        embeddings: tuple[
            ChunkEmbeddingWrite,
            ...,
        ],
    ) -> None:
        ...
