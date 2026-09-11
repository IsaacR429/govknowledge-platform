from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.application.ports.source_page_repository import (
    SourcePage,
    SourcePageRepository,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentPage,
    DocumentVersion,
)


class PostgresSourcePageRepository(
    SourcePageRepository
):
    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def get_by_id(
        self,
        page_id: UUID,
    ) -> SourcePage | None:
        statement = (
            select(
                DocumentPage,
                DocumentVersion,
                Document,
            )
            .join(
                DocumentVersion,
                DocumentVersion.id
                == DocumentPage.document_version_id,
            )
            .join(
                Document,
                Document.id
                == DocumentVersion.document_id,
            )
            .where(
                DocumentPage.id == page_id
            )
        )

        row = self._db.execute(
            statement
        ).one_or_none()

        if row is None:
            return None

        page, version, document = row

        return SourcePage(
            page_id=page.id,
            document_id=document.id,
            document_version_id=(
                version.id
            ),
            page_number=(
                page.page_number
            ),
            extracted_text=(
                page.extracted_text
            ),
            metadata=dict(
                page.metadata_json
                or {}
            ),
            document_title=(
                document.title
            ),
            issuing_organization=(
                document
                .issuing_organization
            ),
            jurisdiction=(
                document.jurisdiction
            ),
            document_language=(
                document.language
            ),
            publication_date=(
                document.publication_date
            ),
            source_url=(
                document.source_url
            ),
            version_number=(
                version.version_number
            ),
            original_filename=(
                version.original_filename
            ),
        )
