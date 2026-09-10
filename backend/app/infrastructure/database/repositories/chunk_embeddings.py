from uuid import UUID

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.application.ports.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
    ChunkEmbeddingWrite,
    ChunkForEmbedding,
)
from app.domain.enums.processing import ProcessingStatus
from app.infrastructure.database.models.documents import (
    DocumentChunk,
    DocumentVersion,
)
from app.infrastructure.database.models.embeddings import (
    ChunkEmbedding,
)


class PostgresChunkEmbeddingRepository(
    ChunkEmbeddingRepository
):
    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def list_missing_chunks(
        self,
        *,
        version_id: UUID,
        model_name: str,
        model_revision: str,
    ) -> tuple[ChunkForEmbedding, ...]:
        try:
            version = self._db.get(
                DocumentVersion,
                version_id,
            )

            if version is None:
                raise ValueError(
                    "Document version does not exist."
                )

            if (
                version.processing_status
                != ProcessingStatus.SUCCEEDED.value
            ):
                raise ValueError(
                    "Only a successfully processed "
                    "document version can be indexed."
                )

            already_embedded = (
                exists()
                .where(
                    ChunkEmbedding.chunk_id
                    == DocumentChunk.id,
                    ChunkEmbedding.model_name
                    == model_name,
                    ChunkEmbedding.model_revision
                    == model_revision,
                )
            )

            rows = self._db.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.document_id,
                    DocumentChunk.document_version_id,
                    DocumentChunk.chunk_index,
                    DocumentChunk.content,
                )
                .where(
                    DocumentChunk.document_version_id
                    == version_id,
                    ~already_embedded,
                )
                .order_by(
                    DocumentChunk.chunk_index
                )
            ).all()

            return tuple(
                ChunkForEmbedding(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    document_version_id=(
                        row.document_version_id
                    ),
                    chunk_index=row.chunk_index,
                    content=row.content,
                )
                for row in rows
            )

        finally:
            # Close the read transaction before
            # potentially expensive model inference.
            self._db.rollback()

    def save_embeddings(
        self,
        *,
        version_id: UUID,
        embeddings: tuple[
            ChunkEmbeddingWrite,
            ...,
        ],
    ) -> None:
        if not embeddings:
            return

        chunk_ids = {
            item.chunk_id
            for item in embeddings
        }

        if len(chunk_ids) != len(embeddings):
            raise ValueError(
                "Duplicate chunk IDs in embedding batch."
            )

        valid_chunk_ids = set(
            self._db.scalars(
                select(
                    DocumentChunk.id
                ).where(
                    DocumentChunk.document_version_id
                    == version_id,
                    DocumentChunk.id.in_(
                        chunk_ids
                    ),
                )
            )
        )

        if valid_chunk_ids != chunk_ids:
            self._db.rollback()

            raise ValueError(
                "Embedding batch contains a chunk "
                "outside the requested document version."
            )

        for item in embeddings:
            self._db.add(
                ChunkEmbedding(
                    chunk_id=item.chunk_id,
                    model_name=item.model_name,
                    model_revision=(
                        item.model_revision
                    ),
                    dimension=item.dimension,
                    normalized=item.normalized,
                    embedding=list(
                        item.values
                    ),
                )
            )

        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
