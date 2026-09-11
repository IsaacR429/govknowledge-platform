from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.application.ports.citation_source_repository import (
    CitationSource,
    CitationSourceRepository,
)
from app.infrastructure.database.models.documents import (
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
)


class PostgresCitationSourceRepository(
    CitationSourceRepository
):
    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def get_by_chunk_id(
        self,
        chunk_id: UUID,
    ) -> CitationSource | None:
        statement = (
            select(
                DocumentChunk,
                DocumentPage,
                DocumentVersion,
            )
            .join(
                DocumentPage,
                DocumentPage.id
                == DocumentChunk.page_id,
            )
            .join(
                DocumentVersion,
                DocumentVersion.id
                == DocumentChunk.document_version_id,
            )
            .where(
                DocumentChunk.id == chunk_id
            )
        )

        row = self._db.execute(
            statement
        ).one_or_none()

        if row is None:
            return None

        chunk, page, version = row

        return CitationSource(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_version_id=version.id,
            page_id=page.id,
            page_number=page.page_number,
            content=chunk.content,
        )
