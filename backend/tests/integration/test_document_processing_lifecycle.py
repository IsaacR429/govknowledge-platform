from io import BytesIO
from uuid import uuid4

import pytest

from sqlalchemy import delete, func, select

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.application.services.document_ingestion import (
    CreateDocumentRequest,
    DocumentIngestionService,
)
from app.application.services.document_processing import (
    DocumentProcessingService,
    ProcessDocumentRequest,
)
from app.domain.enums.processing import (
    ProcessingStatus,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from app.infrastructure.database.repositories.document_ingestion import (
    PostgresDocumentIngestionRepository,
)
from app.infrastructure.database.repositories.document_processing import (
    PostgresDocumentProcessingRepository,
)
from app.infrastructure.database.session import (
    SessionLocal,
)
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)


class CountingParser:
    def __init__(
        self,
        parsed: ParsedDocument,
    ):
        self.parsed = parsed
        self.calls = 0

    def parse(
        self,
        file_path,
        *,
        language,
    ):
        self.calls += 1
        return self.parsed


def valid_parsed_document():
    return ParsedDocument(
        pages=(
            ParsedPage(
                page_number=1,
                text="Page one",
                metadata={},
            ),
            ParsedPage(
                page_number=2,
                text="Page two",
                metadata={},
            ),
        ),
        chunks=(
            ParsedChunk(
                page_number=1,
                chunk_index=0,
                content="Page one",
                section_title=None,
                language="en",
                metadata={
                    "page_number": 1,
                },
            ),
            ParsedChunk(
                page_number=2,
                chunk_index=1,
                content="Page two",
                section_title=None,
                language="en",
                metadata={
                    "page_number": 2,
                },
            ),
        ),
    )


def invalid_parsed_document():
    return ParsedDocument(
        pages=(
            ParsedPage(
                page_number=1,
                text="Only page one",
                metadata={},
            ),
        ),
        chunks=(
            ParsedChunk(
                page_number=2,
                chunk_index=0,
                content=(
                    "This chunk references "
                    "a missing page."
                ),
                section_title=None,
                language="en",
                metadata={
                    "page_number": 2,
                },
            ),
        ),
    )


def ingest_document(
    db,
    storage,
):
    service = DocumentIngestionService(
        repository=(
            PostgresDocumentIngestionRepository(
                db
            )
        ),
        storage=storage,
    )

    return service.create_initial_upload(
        request=CreateDocumentRequest(
            title=(
                "Lifecycle Test "
                f"{uuid4()}"
            ),
            description=(
                "Processing lifecycle "
                "integration test."
            ),
            document_type="TREATY",
            issuing_organization=(
                "GovKnowledge Test"
            ),
            jurisdiction="Test",
            language="en",
            publication_date=None,
            source_url=None,
            confidentiality_level=(
                "PUBLIC"
            ),
            original_filename="test.pdf",
        ),
        source=BytesIO(
            b"%PDF-1.4 fake test content"
        ),
    )


def cleanup_document(
    db,
    *,
    document_id,
    version_id,
):
    db.rollback()

    db.execute(
        delete(
            DocumentChunk
        ).where(
            DocumentChunk.document_version_id
            == version_id
        )
    )

    db.execute(
        delete(
            DocumentPage
        ).where(
            DocumentPage.document_version_id
            == version_id
        )
    )

    db.execute(
        delete(
            DocumentProcessingJob
        ).where(
            DocumentProcessingJob.document_version_id
            == version_id
        )
    )

    db.execute(
        delete(
            DocumentVersion
        ).where(
            DocumentVersion.id
            == version_id
        )
    )

    db.execute(
        delete(
            Document
        ).where(
            Document.id
            == document_id
        )
    )

    db.commit()


def test_successful_processing_cannot_be_rerun(
    tmp_path,
):
    db = SessionLocal()

    storage = LocalFileStorage(
        tmp_path / "documents"
    )

    created = None

    try:
        created = ingest_document(
            db,
            storage,
        )

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        assert version is not None
        storage_path = version.storage_path

        parser = CountingParser(
            valid_parsed_document()
        )

        service = DocumentProcessingService(
            repository=(
                PostgresDocumentProcessingRepository(
                    db
                )
            ),
            storage=storage,
            parser=parser,
        )

        request = ProcessDocumentRequest(
            document_id=created.document_id,
            version_id=created.version_id,
            processing_job_id=(
                created.processing_job_id
            ),
            storage_path=(
                storage_path
            ),
            language="en",
        )

        service.process(request)

        assert parser.calls == 1

        with pytest.raises(
            ValueError,
            match=(
                "Processing job must be "
                "PENDING"
            ),
        ):
            service.process(request)

        # Parser must not execute twice.
        assert parser.calls == 1

        db.expire_all()

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        job = db.get(
            DocumentProcessingJob,
            created.processing_job_id,
        )

        assert (
            version.processing_status
            == ProcessingStatus.SUCCEEDED.value
        )

        assert (
            job.status
            == ProcessingStatus.SUCCEEDED.value
        )

        page_count = db.scalar(
            select(
                func.count(
                    DocumentPage.id
                )
            ).where(
                DocumentPage.document_version_id
                == created.version_id
            )
        )

        chunk_count = db.scalar(
            select(
                func.count(
                    DocumentChunk.id
                )
            ).where(
                DocumentChunk.document_version_id
                == created.version_id
            )
        )

        assert page_count == 2
        assert chunk_count == 2

    finally:
        if created is not None:
            cleanup_document(
                db,
                document_id=(
                    created.document_id
                ),
                version_id=(
                    created.version_id
                ),
            )

        db.close()


def test_failed_persistence_rolls_back_pages_and_chunks(
    tmp_path,
):
    db = SessionLocal()

    storage = LocalFileStorage(
        tmp_path / "documents"
    )

    created = None

    try:
        created = ingest_document(
            db,
            storage,
        )

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        assert version is not None
        storage_path = version.storage_path

        service = DocumentProcessingService(
            repository=(
                PostgresDocumentProcessingRepository(
                    db
                )
            ),
            storage=storage,
            parser=CountingParser(
                invalid_parsed_document()
            ),
        )

        request = ProcessDocumentRequest(
            document_id=created.document_id,
            version_id=created.version_id,
            processing_job_id=(
                created.processing_job_id
            ),
            storage_path=(
                storage_path
            ),
            language="en",
        )

        with pytest.raises(
            ValueError,
            match=(
                "Chunk references a page"
            ),
        ):
            service.process(request)

        db.expire_all()

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        job = db.get(
            DocumentProcessingJob,
            created.processing_job_id,
        )

        assert (
            version.processing_status
            == ProcessingStatus.FAILED.value
        )

        assert (
            job.status
            == ProcessingStatus.FAILED.value
        )

        assert (
            job.error_code
            == "DOCUMENT_PROCESSING_FAILED"
        )

        page_count = db.scalar(
            select(
                func.count(
                    DocumentPage.id
                )
            ).where(
                DocumentPage.document_version_id
                == created.version_id
            )
        )

        chunk_count = db.scalar(
            select(
                func.count(
                    DocumentChunk.id
                )
            ).where(
                DocumentChunk.document_version_id
                == created.version_id
            )
        )

        # The page was flushed before the
        # invalid chunk was discovered.
        # Rollback must still remove it.
        assert page_count == 0
        assert chunk_count == 0

    finally:
        if created is not None:
            cleanup_document(
                db,
                document_id=(
                    created.document_id
                ),
                version_id=(
                    created.version_id
                ),
            )

        db.close()


def test_failed_document_can_succeed_on_second_attempt(
    tmp_path,
):
    from app.application.services.document_processing_retry import (
        DocumentProcessingRetryService,
    )

    db = SessionLocal()

    storage = LocalFileStorage(
        tmp_path / "documents"
    )

    created = None

    try:
        created = ingest_document(
            db,
            storage,
        )

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        assert version is not None
        storage_path = version.storage_path

        repository = (
            PostgresDocumentProcessingRepository(
                db
            )
        )

        # ---------------------------------
        # Attempt 1 deliberately fails.
        # ---------------------------------

        failing_service = (
            DocumentProcessingService(
                repository=repository,
                storage=storage,
                parser=CountingParser(
                    invalid_parsed_document()
                ),
            )
        )

        attempt_one_request = (
            ProcessDocumentRequest(
                document_id=(
                    created.document_id
                ),
                version_id=(
                    created.version_id
                ),
                processing_job_id=(
                    created.processing_job_id
                ),
                storage_path=storage_path,
                language="en",
            )
        )

        with pytest.raises(
            ValueError,
            match="Chunk references a page",
        ):
            failing_service.process(
                attempt_one_request
            )

        db.expire_all()

        attempt_one = db.get(
            DocumentProcessingJob,
            created.processing_job_id,
        )

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        assert attempt_one is not None
        assert version is not None

        assert attempt_one.attempt_number == 1

        assert (
            attempt_one.status
            == ProcessingStatus.FAILED.value
        )

        assert (
            version.processing_status
            == ProcessingStatus.FAILED.value
        )

        # ---------------------------------
        # Create attempt 2.
        # ---------------------------------

        retry_service = (
            DocumentProcessingRetryService(
                repository=repository
            )
        )

        retry = retry_service.create_retry(
            version_id=created.version_id
        )

        assert retry.attempt_number == 2

        attempt_two = db.get(
            DocumentProcessingJob,
            retry.processing_job_id,
        )

        assert attempt_two is not None

        assert (
            attempt_two.status
            == ProcessingStatus.PENDING.value
        )

        # Creating another retry while attempt
        # 2 is pending must be impossible.
        with pytest.raises(
            ValueError,
            match=(
                "pending or active processing job"
            ),
        ):
            retry_service.create_retry(
                version_id=created.version_id
            )

        # ---------------------------------
        # Attempt 2 succeeds.
        # ---------------------------------

        successful_parser = CountingParser(
            valid_parsed_document()
        )

        successful_service = (
            DocumentProcessingService(
                repository=repository,
                storage=storage,
                parser=successful_parser,
            )
        )

        result = successful_service.process(
            ProcessDocumentRequest(
                document_id=(
                    created.document_id
                ),
                version_id=(
                    created.version_id
                ),
                processing_job_id=(
                    retry.processing_job_id
                ),
                storage_path=storage_path,
                language="en",
            )
        )

        assert result.page_count == 2
        assert result.chunk_count == 2
        assert successful_parser.calls == 1

        db.expire_all()

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        jobs = list(
            db.scalars(
                select(
                    DocumentProcessingJob
                )
                .where(
                    DocumentProcessingJob.document_version_id
                    == created.version_id
                )
                .order_by(
                    DocumentProcessingJob.attempt_number
                )
            )
        )

        assert [
            job.attempt_number
            for job in jobs
        ] == [1, 2]

        assert [
            job.status
            for job in jobs
        ] == [
            ProcessingStatus.FAILED.value,
            ProcessingStatus.SUCCEEDED.value,
        ]

        assert (
            version.processing_status
            == ProcessingStatus.SUCCEEDED.value
        )

        # Attempt-one failure history must
        # remain intact after retry success.
        assert (
            jobs[0].error_code
            == "DOCUMENT_PROCESSING_FAILED"
        )

        assert jobs[1].error_code is None
        assert jobs[1].error_message is None

        page_count = db.scalar(
            select(
                func.count(
                    DocumentPage.id
                )
            ).where(
                DocumentPage.document_version_id
                == created.version_id
            )
        )

        chunk_count = db.scalar(
            select(
                func.count(
                    DocumentChunk.id
                )
            ).where(
                DocumentChunk.document_version_id
                == created.version_id
            )
        )

        assert page_count == 2
        assert chunk_count == 2

        # Once the version succeeds, another
        # retry is forbidden.
        with pytest.raises(
            ValueError,
            match=(
                "Only a FAILED document version "
                "can be retried"
            ),
        ):
            retry_service.create_retry(
                version_id=created.version_id
            )

    finally:
        if created is not None:
            cleanup_document(
                db,
                document_id=(
                    created.document_id
                ),
                version_id=(
                    created.version_id
                ),
            )

        db.close()
