from __future__ import annotations

import sys
from uuid import UUID

from sqlalchemy import (
    distinct,
    func,
    select,
)

from app.application.services.document_indexing import (
    DocumentIndexingService,
)
from app.domain.enums.processing import ProcessingStatus
from app.infrastructure.database.models.documents import (
    DocumentChunk,
    DocumentVersion,
)
from app.infrastructure.database.models.embeddings import (
    ChunkEmbedding,
)
from app.infrastructure.database.repositories.chunk_embeddings import (
    PostgresChunkEmbeddingRepository,
)
from app.infrastructure.database.session import (
    SessionLocal,
)
from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    SentenceTransformerEmbeddingProvider,
)


EXPECTED_CHUNKS = 107


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m "
            "scripts.prove_real_embedding_index "
            "<version-uuid>"
        )

    version_id = UUID(
        sys.argv[1]
    )

    db = SessionLocal()

    try:
        version = db.get(
            DocumentVersion,
            version_id,
        )

        if version is None:
            raise RuntimeError(
                "Document version not found."
            )

        if (
            version.processing_status
            != ProcessingStatus.SUCCEEDED.value
        ):
            raise RuntimeError(
                "Document version must be "
                "SUCCEEDED before indexing."
            )

        chunk_count = db.scalar(
            select(
                func.count(
                    DocumentChunk.id
                )
            ).where(
                DocumentChunk.document_version_id
                == version_id
            )
        )

        print()
        print(
            "=== Canonical source ==="
        )
        print(
            "Version ID:",
            version_id,
        )
        print(
            "Processing status:",
            version.processing_status,
        )
        print(
            "Canonical chunks:",
            chunk_count,
        )

        assert chunk_count == EXPECTED_CHUNKS

        existing_before = db.scalar(
            select(
                func.count(
                    ChunkEmbedding.id
                )
            )
            .join(
                DocumentChunk,
                ChunkEmbedding.chunk_id
                == DocumentChunk.id,
            )
            .where(
                DocumentChunk.document_version_id
                == version_id,
                ChunkEmbedding.model_name
                == DEFAULT_MODEL_NAME,
                ChunkEmbedding.model_revision
                == DEFAULT_MODEL_REVISION,
            )
        )

        print(
            "Embeddings before:",
            existing_before,
        )

        # End the inspection transaction
        # before loading/running the model.
        db.rollback()

        print()
        print(
            "Loading pinned E5 model..."
        )

        provider = (
            SentenceTransformerEmbeddingProvider()
        )

        print(
            "Model:",
            provider.model_name,
        )
        print(
            "Revision:",
            provider.model_revision,
        )
        print(
            "Dimension:",
            provider.dimension,
        )

        assert (
            provider.model_name
            == DEFAULT_MODEL_NAME
        )

        assert (
            provider.model_revision
            == DEFAULT_MODEL_REVISION
        )

        assert (
            provider.dimension
            == DEFAULT_DIMENSION
        )

        repository = (
            PostgresChunkEmbeddingRepository(
                db
            )
        )

        service = DocumentIndexingService(
            repository=repository,
            embedding_provider=provider,
            batch_size=32,
        )

        print()
        print(
            "Running first indexing pass..."
        )

        first = service.index_version(
            version_id=version_id
        )

        expected_new = (
            EXPECTED_CHUNKS
            - existing_before
        )

        assert (
            first.indexed_count
            == expected_new
        )

        db.expire_all()

        total_embeddings = db.scalar(
            select(
                func.count(
                    ChunkEmbedding.id
                )
            )
            .join(
                DocumentChunk,
                ChunkEmbedding.chunk_id
                == DocumentChunk.id,
            )
            .where(
                DocumentChunk.document_version_id
                == version_id,
                ChunkEmbedding.model_name
                == DEFAULT_MODEL_NAME,
                ChunkEmbedding.model_revision
                == DEFAULT_MODEL_REVISION,
            )
        )

        distinct_chunks = db.scalar(
            select(
                func.count(
                    distinct(
                        ChunkEmbedding.chunk_id
                    )
                )
            )
            .join(
                DocumentChunk,
                ChunkEmbedding.chunk_id
                == DocumentChunk.id,
            )
            .where(
                DocumentChunk.document_version_id
                == version_id,
                ChunkEmbedding.model_name
                == DEFAULT_MODEL_NAME,
                ChunkEmbedding.model_revision
                == DEFAULT_MODEL_REVISION,
            )
        )

        invalid_metadata = db.scalar(
            select(
                func.count(
                    ChunkEmbedding.id
                )
            )
            .join(
                DocumentChunk,
                ChunkEmbedding.chunk_id
                == DocumentChunk.id,
            )
            .where(
                DocumentChunk.document_version_id
                == version_id,
                ChunkEmbedding.model_name
                == DEFAULT_MODEL_NAME,
                ChunkEmbedding.model_revision
                == DEFAULT_MODEL_REVISION,
                (
                    (
                        ChunkEmbedding.dimension
                        != DEFAULT_DIMENSION
                    )
                    | (
                        ChunkEmbedding.normalized
                        .is_(False)
                    )
                ),
            )
        )

        assert (
            total_embeddings
            == EXPECTED_CHUNKS
        )

        assert (
            distinct_chunks
            == EXPECTED_CHUNKS
        )

        assert invalid_metadata == 0

        db.rollback()

        print()
        print(
            "Running idempotency pass..."
        )

        second = service.index_version(
            version_id=version_id
        )

        assert second.indexed_count == 0

        print()
        print(
            "=== Embedding persistence proof ==="
        )
        print(
            "New embeddings:",
            first.indexed_count,
        )
        print(
            "Total embeddings:",
            total_embeddings,
        )
        print(
            "Distinct chunk IDs:",
            distinct_chunks,
        )
        print(
            "Invalid metadata rows:",
            invalid_metadata,
        )
        print(
            "Second-pass writes:",
            second.indexed_count,
        )
        print()
        print(
            "Real embedding indexing:",
            "PASS",
        )
        print(
            "Indexing idempotency:",
            "PASS",
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
