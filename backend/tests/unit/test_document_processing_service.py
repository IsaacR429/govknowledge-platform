from pathlib import Path
from uuid import uuid4

import pytest

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.application.ports.document_processing_repository import (
    ProcessingTarget,
)
from app.application.services.audit import (
    AuditPersistenceError,
)
from app.application.services.document_processing import (
    DocumentProcessingService,
    ProcessDocumentRequest,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)


class FakeStorage:
    def __init__(self, path: Path):
        self.path = path

    def resolve(
        self,
        storage_path: str,
    ) -> Path:
        assert storage_path == "document.pdf"
        return self.path


class SuccessfulParser:
    def parse(
        self,
        file_path: Path,
        *,
        language: str,
    ) -> ParsedDocument:
        return ParsedDocument(
            pages=(
                ParsedPage(
                    page_number=1,
                    text="Page un.",
                    metadata={},
                ),
                ParsedPage(
                    page_number=2,
                    text="Page deux.",
                    metadata={},
                ),
            ),
            chunks=(
                ParsedChunk(
                    page_number=1,
                    chunk_index=0,
                    content="Page un.",
                    section_title=None,
                    language=language,
                    metadata={},
                ),
                ParsedChunk(
                    page_number=2,
                    chunk_index=1,
                    content="Page deux.",
                    section_title=None,
                    language=language,
                    metadata={},
                ),
            ),
        )


class FailingParser:
    def parse(
        self,
        file_path: Path,
        *,
        language: str,
    ):
        raise RuntimeError(
            "Simulated parser failure"
        )


class RecordingRepository:
    def __init__(
        self,
        target,
    ):
        self.target = target
        self.calls = []

    def mark_processing(
        self,
        *,
        processing_job_id,
    ):
        self.calls.append(
            (
                "processing",
                processing_job_id,
            )
        )

        assert (
            processing_job_id
            == self.target.processing_job_id
        )

        return self.target

    def complete_processing(
        self,
        *,
        document_id,
        version_id,
        processing_job_id,
        parsed,
        resolve_stored_file=None,
    ):
        self.calls.append(
            (
                "complete",
                document_id,
                version_id,
                processing_job_id,
                len(parsed.pages),
                len(parsed.chunks),
            )
        )

    def fail_processing(
        self,
        *,
        version_id,
        processing_job_id,
        error_code,
        error_message,
    ):
        self.calls.append(
            (
                "failed",
                version_id,
                processing_job_id,
                error_code,
                error_message,
            )
        )


class FakeAuditService:
    def __init__(
        self,
        *,
        fail_on_event=None,
    ):
        self.calls = []
        self.fail_on_event = (
            fail_on_event
        )

    def record(
        self,
        event,
    ):
        self.calls.append(event)

        if (
            event.event_type
            == self.fail_on_event
        ):
            raise AuditPersistenceError(
                "Audit unavailable."
            )

        return uuid4()


def make_request():
    return ProcessDocumentRequest(
        processing_job_id=uuid4(),
    )


def make_target(
    request,
):
    return ProcessingTarget(
        processing_job_id=(
            request.processing_job_id
        ),
        version_id=uuid4(),
        document_id=uuid4(),
        storage_path="document.pdf",
        language="fr",
    )


def service(
    *,
    repository,
    storage,
    parser,
    audit=None,
):
    return DocumentProcessingService(
        repository=repository,
        storage=storage,
        parser=parser,
        audit=(
            audit
            if audit is not None
            else FakeAuditService()
        ),
    )


def test_processing_service_completes_document(
    tmp_path,
):
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"test")

    request = make_request()
    target = make_target(request)
    repository = RecordingRepository(
        target
    )
    audit = FakeAuditService()

    result = service(
        repository=repository,
        storage=FakeStorage(file_path),
        parser=SuccessfulParser(),
        audit=audit,
    ).process(
        request
    )

    assert result.page_count == 2
    assert result.chunk_count == 2

    assert [
        call[0]
        for call in repository.calls
    ] == [
        "processing",
        "complete",
    ]

    assert [
        event.event_type
        for event in audit.calls
    ] == [
        AuditEventType
        .DOCUMENT_PROCESSING_STARTED,
        AuditEventType
        .DOCUMENT_PROCESSING_COMPLETED,
    ]

    started, completed = audit.calls

    assert (
        started.outcome
        == AuditOutcome.SUCCEEDED
    )
    assert (
        completed.outcome
        == AuditOutcome.SUCCEEDED
    )

    assert (
        started.correlation_id
        == request.processing_job_id
    )
    assert (
        completed.correlation_id
        == request.processing_job_id
    )

    assert started.actor_user_id is None

    assert (
        completed.metadata[
            "page_count"
        ]
        == 2
    )

    assert (
        completed.metadata[
            "chunk_count"
        ]
        == 2
    )


def test_processing_service_marks_failure(
    tmp_path,
):
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"test")

    actor_user_id = uuid4()
    correlation_id = uuid4()

    base_request = make_request()

    request = ProcessDocumentRequest(
        processing_job_id=(
            base_request.processing_job_id
        ),
        actor_user_id=actor_user_id,
        correlation_id=correlation_id,
    )
    target = make_target(request)
    repository = RecordingRepository(
        target
    )
    audit = FakeAuditService()

    with pytest.raises(
        RuntimeError,
        match="Simulated parser failure",
    ):
        service(
            repository=repository,
            storage=FakeStorage(file_path),
            parser=FailingParser(),
            audit=audit,
        ).process(
            request
        )

    assert [
        call[0]
        for call in repository.calls
    ] == [
        "processing",
        "failed",
    ]

    assert (
        repository.calls[1][3]
        == "DOCUMENT_PROCESSING_FAILED"
    )

    assert (
        repository.calls[1][4]
        == "Document processing failed."
    )

    assert (
        "Simulated parser failure"
        not in repository.calls[1][4]
    )

    assert [
        event.event_type
        for event in audit.calls
    ] == [
        AuditEventType
        .DOCUMENT_PROCESSING_STARTED,
        AuditEventType
        .DOCUMENT_PROCESSING_FAILED,
    ]

    started = audit.calls[0]
    failed = audit.calls[1]

    assert (
        started.actor_user_id
        == actor_user_id
    )
    assert (
        started.correlation_id
        == correlation_id
    )

    assert (
        failed.actor_user_id
        == actor_user_id
    )
    assert (
        failed.correlation_id
        == correlation_id
    )

    assert (
        failed.resource_id
        == request.processing_job_id
    )
    assert (
        failed.document_id
        == target.document_id
    )

    assert failed.outcome == AuditOutcome.FAILED

    assert (
        failed.error_code
        == "DOCUMENT_PROCESSING_FAILED"
    )

    # Raw parser exception text must not
    # enter the audit payload.
    assert (
        "Simulated parser failure"
        not in str(failed.metadata)
    )


def test_started_audit_failure_does_not_block_processing(
    tmp_path,
):
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"test")

    request = make_request()
    target = make_target(request)
    repository = RecordingRepository(
        target
    )

    result = service(
        repository=repository,
        storage=FakeStorage(file_path),
        parser=SuccessfulParser(),
        audit=FakeAuditService(
            fail_on_event=(
                AuditEventType
                .DOCUMENT_PROCESSING_STARTED
            )
        ),
    ).process(
        request
    )

    assert result.page_count == 2

    assert [
        call[0]
        for call in repository.calls
    ] == [
        "processing",
        "complete",
    ]


def test_completed_audit_failure_does_not_mark_job_failed(
    tmp_path,
):
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"test")

    request = make_request()
    target = make_target(request)
    repository = RecordingRepository(
        target
    )

    result = service(
        repository=repository,
        storage=FakeStorage(file_path),
        parser=SuccessfulParser(),
        audit=FakeAuditService(
            fail_on_event=(
                AuditEventType
                .DOCUMENT_PROCESSING_COMPLETED
            )
        ),
    ).process(
        request
    )

    assert result.page_count == 2

    assert [
        call[0]
        for call in repository.calls
    ] == [
        "processing",
        "complete",
    ]

    assert all(
        call[0] != "failed"
        for call in repository.calls
    )


def test_failed_audit_failure_preserves_parser_failure(
    tmp_path,
):
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"test")

    request = make_request()
    target = make_target(request)
    repository = RecordingRepository(
        target
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated parser failure",
    ):
        service(
            repository=repository,
            storage=FakeStorage(file_path),
            parser=FailingParser(),
            audit=FakeAuditService(
                fail_on_event=(
                    AuditEventType
                    .DOCUMENT_PROCESSING_FAILED
                )
            ),
        ).process(
            request
        )

    assert [
        call[0]
        for call in repository.calls
    ] == [
        "processing",
        "failed",
    ]
