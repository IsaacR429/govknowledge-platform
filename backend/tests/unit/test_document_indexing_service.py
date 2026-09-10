from uuid import uuid4

import pytest

from app.application.ports.chunk_embedding_repository import (
    ChunkForEmbedding,
)
from app.application.ports.embedding_provider import (
    EmbeddingResult,
)
from app.application.services.document_indexing import (
    DocumentIndexingService,
)


class FakeRepository:
    def __init__(self, chunks):
        self.chunks = tuple(chunks)
        self.saved = None
        self.request = None

    def list_missing_chunks(
        self,
        *,
        version_id,
        model_name,
        model_revision,
    ):
        self.request = (
            version_id,
            model_name,
            model_revision,
        )

        return self.chunks

    def save_embeddings(
        self,
        *,
        version_id,
        embeddings,
    ):
        self.saved = (
            version_id,
            embeddings,
        )


class FakeProvider:
    model_name = "test-model"
    model_revision = "test-revision"
    dimension = 3

    def __init__(self):
        self.batches = []

    def embed_passages(
        self,
        texts,
    ):
        self.batches.append(
            list(texts)
        )

        return tuple(
            EmbeddingResult(
                values=(
                    1.0,
                    0.0,
                    0.0,
                ),
                model_name=self.model_name,
                model_revision=(
                    self.model_revision
                ),
                dimension=self.dimension,
            )
            for _ in texts
        )


def chunks(count):
    document_id = uuid4()
    version_id = uuid4()

    return (
        version_id,
        [
            ChunkForEmbedding(
                chunk_id=uuid4(),
                document_id=document_id,
                document_version_id=version_id,
                chunk_index=index,
                content=f"Chunk {index}",
            )
            for index in range(count)
        ],
    )


def test_indexing_batches_and_persists():
    version_id, items = chunks(5)

    repository = FakeRepository(
        items
    )
    provider = FakeProvider()

    service = DocumentIndexingService(
        repository=repository,
        embedding_provider=provider,
        batch_size=2,
    )

    result = service.index_version(
        version_id=version_id
    )

    assert provider.batches == [
        ["Chunk 0", "Chunk 1"],
        ["Chunk 2", "Chunk 3"],
        ["Chunk 4"],
    ]

    assert result.indexed_count == 5

    assert repository.saved is not None

    saved_version, saved = (
        repository.saved
    )

    assert saved_version == version_id
    assert len(saved) == 5

    assert all(
        item.dimension == 3
        for item in saved
    )

    assert all(
        item.normalized is True
        for item in saved
    )


def test_already_indexed_version_is_noop():
    version_id = uuid4()

    repository = FakeRepository(
        ()
    )

    provider = FakeProvider()

    service = DocumentIndexingService(
        repository=repository,
        embedding_provider=provider,
    )

    result = service.index_version(
        version_id=version_id
    )

    assert result.indexed_count == 0
    assert provider.batches == []
    assert repository.saved is None


def test_invalid_batch_size_is_rejected():
    with pytest.raises(
        ValueError,
        match="batch size must be positive",
    ):
        DocumentIndexingService(
            repository=FakeRepository(()),
            embedding_provider=FakeProvider(),
            batch_size=0,
        )
