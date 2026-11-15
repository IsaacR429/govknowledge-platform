from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from sqlalchemy import delete, func, select

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.application.services.citation_mapping import (
    EvidenceSpanCitationMapper,
)
from app.application.services.citation_validation import (
    CitationValidator,
    InvalidCitationError,
)
from app.application.services.document_indexing import (
    DocumentIndexingService,
)
from app.application.services.document_ingestion import (
    CreateDocumentRequest,
    DocumentIngestionService,
)
from app.application.services.document_processing import (
    DocumentProcessingService,
    ProcessDocumentRequest,
)
from app.domain.enums.permission import PermissionAction
from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
    ProcessingStatus,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.embeddings import (
    ChunkEmbedding,
    EMBEDDING_DIMENSION,
)
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from app.infrastructure.database.repositories.chunk_embeddings import (
    PostgresChunkEmbeddingRepository,
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
from app.infrastructure.retrieval.postgres_vector_retrieval import (
    PostgresVectorRetrievalAdapter,
)
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)


class RecordingAuditService:
    def __init__(self):
        self.calls = []

    def record(
        self,
        event,
    ):
        self.calls.append(event)
        return uuid4()


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


class DeterministicTableTestEmbedding:
    model_name = "qa-reviewed-table-index-test"
    model_revision = "fixture-1"
    dimension = EMBEDDING_DIMENSION

    def __init__(self):
        self.calls = []

    def embed_passages(self, texts):
        self.calls.append(tuple(texts))

        # Unit-length vector with the database's expected dimension.
        values = (1.0,) + (0.0,) * (self.dimension - 1)

        return tuple(
            SimpleNamespace(
                model_name=self.model_name,
                model_revision=self.model_revision,
                dimension=self.dimension,
                values=values,
            )
            for _ in texts
        )


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
    source=None,
    actor_user_id=None,
    correlation_id=None,
):
    from app.application.services.audit import AuditService
    from app.infrastructure.database.repositories.audit import (
        PostgresAuditRepository,
    )

    audit_service = AuditService(
        repository=PostgresAuditRepository(db)
    )

    service = DocumentIngestionService(
        repository=(
            PostgresDocumentIngestionRepository(
                db
            )
        ),
        storage=storage,
        audit=audit_service,
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
        source=(
            source
            if source is not None
            else BytesIO(b"%PDF-1.4 fake test content")
        ),
        actor_user_id=actor_user_id or uuid4(),
        correlation_id=correlation_id or uuid4(),
    )


def cleanup_document(
    db,
    *,
    document_id,
    version_id,
):
    db.rollback()

    db.execute(
        delete(ChunkEmbedding).where(
            ChunkEmbedding.chunk_id.in_(
                select(DocumentChunk.id).where(
                    DocumentChunk.document_version_id == version_id
                )
            )
        )
    )

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
            audit=RecordingAuditService(),
        )

        request = ProcessDocumentRequest(
            processing_job_id=(
                created.processing_job_id
            ),
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


def test_invalid_parsed_document_persists_no_pages_or_chunks(
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
            audit=RecordingAuditService(),
        )

        request = ProcessDocumentRequest(
            processing_job_id=(
                created.processing_job_id
            ),
        )

        with pytest.raises(
            ValueError,
            match=(
                "Chunk references an unknown page"
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

        assert (
            job.error_message
            == "Document processing failed."
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

        # Validation occurs before canonical
        # pages or chunks are persisted.
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
                audit=RecordingAuditService(),
            )
        )

        attempt_one_request = (
            ProcessDocumentRequest(
                processing_job_id=(
                    created.processing_job_id
                ),
            )
        )

        with pytest.raises(
            ValueError,
            match="Chunk references an unknown page",
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
                audit=RecordingAuditService(),
            )
        )

        result = successful_service.process(
            ProcessDocumentRequest(
                processing_job_id=(
                    retry.processing_job_id
                ),
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



def test_processing_target_is_derived_from_database(
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

        document = db.get(
            Document,
            created.document_id,
        )

        assert version is not None
        assert document is not None

        repository = (
            PostgresDocumentProcessingRepository(
                db
            )
        )

        target = repository.mark_processing(
            processing_job_id=(
                created.processing_job_id
            )
        )

        assert (
            target.processing_job_id
            == created.processing_job_id
        )

        assert (
            target.version_id
            == version.id
        )

        assert (
            target.document_id
            == document.id
        )

        assert (
            target.storage_path
            == version.storage_path
        )

        assert (
            target.language
            == document.language
        )

        # The repository, not the caller,
        # performed the lifecycle transition.
        db.expire_all()

        persisted_version = db.get(
            DocumentVersion,
            created.version_id,
        )

        persisted_job = db.get(
            DocumentProcessingJob,
            created.processing_job_id,
        )

        assert (
            persisted_version.processing_status
            == ProcessingStatus.PROCESSING.value
        )

        assert (
            persisted_job.status
            == ProcessingStatus.PROCESSING.value
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



def test_parser_outcome_persists_across_fresh_session(
    tmp_path,
):
    db = SessionLocal()
    verification_db = None
    created = None

    storage = LocalFileStorage(
        tmp_path / "documents"
    )

    lineage = {
        "execution_mode": "ISOLATED_WORKER",
        "parser_family": "docling",
        "original_sha256": "abc123",
        "normalization_attempted": False,
        "normalization_attempt_count": 0,
        "metrics": {
            "worker_spawn_ms": 12.5,
            "parser_execution_ms": 750.0,
        },
    }

    try:
        created = ingest_document(
            db,
            storage,
        )

        repository = (
            PostgresDocumentProcessingRepository(
                db
            )
        )

        target = repository.mark_processing(
            processing_job_id=(
                created.processing_job_id
            )
        )

        repository.record_parser_outcome(
            version_id=target.version_id,
            processing_job_id=(
                target.processing_job_id
            ),
            parsing_status=(
                ParsingStatus.FAILED
            ),
            failure_class=(
                ParserFailureClass.TIMEOUT
            ),
            parser_lineage=lineage,
        )

        # Read through a completely new SQLAlchemy
        # Session. This proves record_parser_outcome()
        # committed the values instead of merely
        # mutating the original ORM identity map.
        verification_db = SessionLocal()

        persisted = verification_db.get(
            DocumentProcessingJob,
            created.processing_job_id,
        )

        assert persisted is not None

        # Parser outcome recording is independent
        # from the overall processing lifecycle.
        assert (
            persisted.status
            == ProcessingStatus.PROCESSING.value
        )

        assert (
            persisted.parsing_status
            == ParsingStatus.FAILED.value
        )

        assert (
            persisted.failure_class
            == ParserFailureClass.TIMEOUT.value
        )

        assert (
            persisted.parser_lineage_json
            == lineage
        )

    finally:
        if verification_db is not None:
            verification_db.close()

        if created is not None:
            cleanup_document(
                db,
                document_id=created.document_id,
                version_id=created.version_id,
            )

        db.close()


def test_unapproved_table_evidence_persists_no_pages_or_chunks(
    tmp_path,
):
    from sqlalchemy import text

    db = SessionLocal()
    storage = LocalFileStorage(tmp_path / "documents")
    created = None

    try:
        # Fail before writing anything if this test is pointed
        # at the live database instead of the isolated QA database.
        database_name = db.scalar(
            text("SELECT current_database()")
        )
        assert database_name == "govknowledge_qa", (
            "This test must run against the isolated QA database."
        )

        created = ingest_document(db, storage)

        original = valid_parsed_document()
        candidate_chunk = ParsedChunk(
            page_number=1,
            chunk_index=0,
            content=(
                "Organization: DRAC. "
                "Marked activities: Creation, Diffusion, Training."
            ),
            section_title="The main players",
            language="en",
            metadata={
                "evidence_type": "vector_marker_table",
                "validation_status": "candidate",
                "table_id": "page-19-table-1",
            },
        )

        parsed = ParsedDocument(
            pages=original.pages,
            chunks=(candidate_chunk, *original.chunks[1:]),
        )

        service = DocumentProcessingService(
            repository=PostgresDocumentProcessingRepository(db),
            storage=storage,
            parser=CountingParser(parsed),
            audit=RecordingAuditService(),
        )

        with pytest.raises(
            ValueError,
            match=r"(?i)table evidence",
        ):
            service.process(
                ProcessDocumentRequest(
                    processing_job_id=created.processing_job_id,
                )
            )

        # Verify persisted state through a fresh database session,
        # not just the session that encountered the error.
        with SessionLocal() as verification_db:
            version = verification_db.get(
                DocumentVersion,
                created.version_id,
            )
            job = verification_db.get(
                DocumentProcessingJob,
                created.processing_job_id,
            )

            assert version is not None
            assert job is not None
            assert (
                version.processing_status
                == ProcessingStatus.FAILED.value
            )
            assert job.status == ProcessingStatus.FAILED.value

            page_count = verification_db.scalar(
                select(func.count(DocumentPage.id)).where(
                    DocumentPage.document_version_id
                    == created.version_id
                )
            )
            chunk_count = verification_db.scalar(
                select(func.count(DocumentChunk.id)).where(
                    DocumentChunk.document_version_id
                    == created.version_id
                )
            )

            assert page_count == 0
            assert chunk_count == 0

    finally:
        if created is not None:
            cleanup_document(
                db,
                document_id=created.document_id,
                version_id=created.version_id,
            )
        db.close()
