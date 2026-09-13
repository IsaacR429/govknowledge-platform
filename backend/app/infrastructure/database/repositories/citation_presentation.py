from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.application.ports.citation_presentation_repository import (
    CitationPresentation,
    CitationPresentationRepository,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentPage,
    DocumentVersion,
)


class PostgresCitationPresentationRepository(
    CitationPresentationRepository
):
    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def get_by_page_id(
        self,
        *,
        page_id: UUID,
        document_id: UUID,
    ) -> CitationPresentation | None:
        statement = (
            select(
                DocumentPage.id,
                Document.id,
                Document.title,
                Document.issuing_organization,
                Document.jurisdiction,
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
                DocumentPage.id == page_id,
                Document.id == document_id,
            )
        )

        row = self._db.execute(
            statement
        ).one_or_none()

        if row is None:
            return None

        return CitationPresentation(
            page_id=row[0],
            document_id=row[1],
            document_title=row[2],
            issuing_organization=row[3],
            jurisdiction=row[4],
        )
