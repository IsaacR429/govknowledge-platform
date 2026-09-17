from sqlalchemy import select
from sqlalchemy.orm import Session

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.application.ports.retrieval import (
    RetrievalPort,
    RetrievedChunk,
)
from app.domain.enums.permission import (
    PermissionAction,
)
from app.domain.enums.processing import (
    ProcessingStatus,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentChunk,
    DocumentVersion,
)
from app.infrastructure.database.models.embeddings import (
    ChunkEmbedding,
    EMBEDDING_DIMENSION,
)


class PostgresVectorRetrievalAdapter(
    RetrievalPort
):
    """Exact pgvector retrieval inside an ACL scope."""

    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def search_authorized(
        self,
        *,
        scope: AuthorizationScope,
        query_vector: tuple[float, ...],
        model_name: str,
        model_revision: str,
        top_k: int,
    ) -> tuple[RetrievedChunk, ...]:
        if scope.action != PermissionAction.READ:
            raise ValueError(
                "Retrieval requires a READ "
                "authorization scope."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        # Fail closed: an empty authorization scope
        # must never become an unrestricted query.
        if not scope.document_ids:
            return ()

        if len(query_vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                "Query vector must have "
                f"{EMBEDDING_DIMENSION} dimensions."
            )

        distance = (
            ChunkEmbedding.embedding.cosine_distance(
                list(query_vector)
            )
        )

        statement = (
            select(
                DocumentChunk.id.label(
                    "chunk_id"
                ),
                DocumentChunk.document_id,
                DocumentChunk.document_version_id,
                DocumentChunk.page_id,
                DocumentChunk.page_start.label(
                    "page_number"
                ),
                DocumentChunk.page_end.label("page_end"),
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                DocumentChunk.section_title,
                DocumentChunk.language,
                distance.label(
                    "distance"
                ),
            )
            .join(
                ChunkEmbedding,
                ChunkEmbedding.chunk_id
                == DocumentChunk.id,
            )
            .join(
                Document,
                Document.id
                == DocumentChunk.document_id,
            )
            .join(
                DocumentVersion,
                DocumentVersion.id
                == DocumentChunk.document_version_id,
            )
            .where(
                # SECURITY BOUNDARY:
                # authorization constrains the
                # candidate set BEFORE ranking.
                DocumentChunk.document_id.in_(
                    tuple(
                        scope.document_ids
                    )
                ),
                Document.status == "ACTIVE",
                DocumentVersion.processing_status
                == ProcessingStatus.SUCCEEDED.value,
                ChunkEmbedding.model_name
                == model_name,
                ChunkEmbedding.model_revision
                == model_revision,
            )
            .order_by(
                distance
            )
            .limit(
                top_k
            )
        )

        rows = self._db.execute(
            statement
        ).all()

        return tuple(
            RetrievedChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                document_version_id=(
                    row.document_version_id
                ),
                page_id=row.page_id,
                page_number=row.page_number,
                page_end=row.page_end,
                chunk_index=row.chunk_index,
                content=row.content,
                section_title=row.section_title,
                language=row.language,
                score=(
                    1.0
                    - float(
                        row.distance
                    )
                ),
            )
            for row in rows
        )
