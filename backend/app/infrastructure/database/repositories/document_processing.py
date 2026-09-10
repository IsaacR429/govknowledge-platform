from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.application.ports.document_parser import (
    ParsedDocument,
)
from app.application.ports.document_processing_repository import (
    DocumentProcessingRepository,
)
from app.domain.enums.processing import ProcessingStatus
from app.infrastructure.database.models.documents import (
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)


class PostgresDocumentProcessingRepository(
    DocumentProcessingRepository
):
    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def _load(
        self,
        *,
        version_id: UUID,
        processing_job_id: UUID,
    ) -> tuple[
        DocumentVersion,
        DocumentProcessingJob,
    ]:
        version = self._db.get(
            DocumentVersion,
            version_id,
        )

        job = self._db.get(
            DocumentProcessingJob,
            processing_job_id,
        )

        if version is None:
            raise ValueError(
                "Document version does not exist."
            )

        if job is None:
            raise ValueError(
                "Processing job does not exist."
            )

        if (
            job.document_version_id
            != version.id
        ):
            raise ValueError(
                "Processing job does not belong "
                "to document version."
            )

        return version, job

    def mark_processing(
        self,
        *,
        version_id: UUID,
        processing_job_id: UUID,
    ) -> None:
        version, job = self._load(
            version_id=version_id,
            processing_job_id=processing_job_id,
        )

        now = datetime.now(
            timezone.utc
        )

        version.processing_status = (
            ProcessingStatus.PROCESSING.value
        )

        job.status = (
            ProcessingStatus.PROCESSING.value
        )

        job.started_at = now
        job.completed_at = None
        job.error_code = None
        job.error_message = None

        self._db.commit()

    def complete_processing(
        self,
        *,
        document_id: UUID,
        version_id: UUID,
        processing_job_id: UUID,
        parsed: ParsedDocument,
    ) -> None:
        version, job = self._load(
            version_id=version_id,
            processing_job_id=processing_job_id,
        )

        if version.document_id != document_id:
            raise ValueError(
                "Document version does not belong "
                "to document."
            )

        existing_page = self._db.scalar(
            select(
                DocumentPage.id
            ).where(
                DocumentPage.document_version_id
                == version_id
            ).limit(1)
        )

        if existing_page is not None:
            raise ValueError(
                "Document version already contains "
                "parsed pages."
            )

        page_rows: dict[
            int,
            DocumentPage,
        ] = {}

        try:
            for parsed_page in parsed.pages:
                page = DocumentPage(
                    document_version_id=version_id,
                    page_number=(
                        parsed_page.page_number
                    ),
                    extracted_text=(
                        parsed_page.text
                    ),
                    metadata_json=(
                        parsed_page.metadata
                    ),
                )

                self._db.add(page)

                page_rows[
                    parsed_page.page_number
                ] = page

            # We need database-generated page UUIDs
            # before creating chunk foreign keys.
            self._db.flush()

            for parsed_chunk in parsed.chunks:
                page = page_rows.get(
                    parsed_chunk.page_number
                )

                if page is None:
                    raise ValueError(
                        "Chunk references a page "
                        "that was not parsed."
                    )

                chunk = DocumentChunk(
                    document_id=document_id,
                    document_version_id=version_id,
                    page_id=page.id,
                    page_start=(
                        parsed_chunk.page_number
                    ),
                    page_end=(
                        parsed_chunk.page_number
                    ),
                    chunk_index=(
                        parsed_chunk.chunk_index
                    ),
                    section_title=(
                        parsed_chunk.section_title
                    ),
                    content=(
                        parsed_chunk.content
                    ),
                    language=(
                        parsed_chunk.language
                    ),
                    metadata_json=(
                        parsed_chunk.metadata
                    ),
                )

                self._db.add(chunk)

            version.processing_status = (
                ProcessingStatus.SUCCEEDED.value
            )

            job.status = (
                ProcessingStatus.SUCCEEDED.value
            )

            job.completed_at = datetime.now(
                timezone.utc
            )

            self._db.commit()

        except Exception:
            self._db.rollback()
            raise

    def fail_processing(
        self,
        *,
        version_id: UUID,
        processing_job_id: UUID,
        error_code: str,
        error_message: str,
    ) -> None:
        version, job = self._load(
            version_id=version_id,
            processing_job_id=processing_job_id,
        )

        version.processing_status = (
            ProcessingStatus.FAILED.value
        )

        job.status = (
            ProcessingStatus.FAILED.value
        )

        job.completed_at = datetime.now(
            timezone.utc
        )

        job.error_code = error_code

        # Avoid dumping enormous exception payloads
        # into PostgreSQL.
        job.error_message = (
            error_message[:4000]
        )

        self._db.commit()
