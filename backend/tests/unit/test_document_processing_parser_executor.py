from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.application.ports.document_processing_repository import (
    ProcessingTarget,
)
from app.application.ports.parser_execution import (
    ParserExecutionResult,
)
from app.application.services.document_processing import (
    DocumentProcessingService,
    ProcessDocumentRequest,
)
from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
)
from app.domain.exceptions.document_processing import (
    ParsedDocumentValidationError,
)


def valid_document() -> ParsedDocument:
    return ParsedDocument(
        pages=(
            ParsedPage(
                page_number=1,
                text="Policy text.",
                metadata={},
            ),
        ),
        chunks=(
            ParsedChunk(
                page_number=1,
                chunk_index=0,
                content="Policy text.",
                section_title=None,
                language="en",
                metadata={},
            ),
        ),
    )


def invalid_document() -> ParsedDocument:
    return ParsedDocument(
        pages=(),
        chunks=(),
    )


class FakeStorage:
    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path

    def resolve(
        self,
        storage_path: str,
    ) -> Path:
        return self.path


class FakeAudit:
    def __init__(self) -> None:
        self.events = []

    def record(
        self,
        event,
    ) -> None:
        self.events.append(event)


class FakeExecutor:
    def __init__(
        self,
        result: ParserExecutionResult,
    ) -> None:
        self.result = result
        self.calls = 0

    def execute(
        self,
        file_path: Path,
        *,
        language: str,
    ) -> ParserExecutionResult:
        self.calls += 1
        return self.result


class FakeRepository:
    def __init__(self) -> None:
        self.processing_job_id = uuid4()
        self.version_id = uuid4()
        self.document_id = uuid4()

        self.target = ProcessingTarget(
            processing_job_id=(
                self.processing_job_id
            ),
            version_id=self.version_id,
            document_id=self.document_id,
            storage_path="documents/source.pdf",
            language="en",
        )

        self.overall_status = "PENDING"
        self.parsing_status = None
        self.failure_class = None
        self.parser_lineage = None

        self.completed_parsed = None
        self.failure_error_code = None
        self.failure_error_message = None

    def mark_processing(
        self,
        *,
        processing_job_id: UUID,
    ) -> ProcessingTarget:
        assert (
            processing_job_id
            == self.processing_job_id
        )

        self.overall_status = "PROCESSING"

        return self.target

    def record_parser_outcome(
        self,
        *,
        version_id: UUID,
        processing_job_id: UUID,
        parsing_status: ParsingStatus,
        failure_class: ParserFailureClass | None,
        parser_lineage: dict[str, object],
    ) -> None:
        assert version_id == self.version_id
        assert (
            processing_job_id
            == self.processing_job_id
        )

        self.parsing_status = parsing_status
        self.failure_class = failure_class
        self.parser_lineage = dict(
            parser_lineage
        )

    def complete_processing(
        self,
        *,
        document_id: UUID,
        version_id: UUID,
        processing_job_id: UUID,
        parsed: ParsedDocument,
    ) -> None:
        assert document_id == self.document_id
        assert version_id == self.version_id
        assert (
            processing_job_id
            == self.processing_job_id
        )

        self.completed_parsed = parsed
        self.overall_status = "SUCCEEDED"

    def fail_processing(
        self,
        *,
        version_id: UUID,
        processing_job_id: UUID,
        error_code: str,
        error_message: str,
    ) -> None:
        assert version_id == self.version_id
        assert (
            processing_job_id
            == self.processing_job_id
        )

        self.overall_status = "FAILED"
        self.failure_error_code = error_code
        self.failure_error_message = (
            error_message
        )


def make_service(
    *,
    tmp_path: Path,
    repository: FakeRepository,
    execution: ParserExecutionResult,
) -> tuple[
    DocumentProcessingService,
    FakeExecutor,
]:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-test")

    executor = FakeExecutor(execution)

    service = DocumentProcessingService(
        repository=repository,
        storage=FakeStorage(source),
        parser_executor=executor,
        audit=FakeAudit(),
    )

    return service, executor


def test_executor_success_persists_parser_success(
    tmp_path: Path,
) -> None:
    repository = FakeRepository()

    lineage = {
        "execution_mode": "ISOLATED_WORKER",
        "original_sha256": "abc123",
    }

    service, executor = make_service(
        tmp_path=tmp_path,
        repository=repository,
        execution=ParserExecutionResult(
            parsing_status=(
                ParsingStatus.SUCCESS
            ),
            failure_class=None,
            parsed_document=valid_document(),
            lineage=lineage,
        ),
    )

    result = service.process(
        ProcessDocumentRequest(
            processing_job_id=(
                repository.processing_job_id
            ),
        )
    )

    assert executor.calls == 1
    assert repository.overall_status == (
        "SUCCEEDED"
    )
    assert (
        repository.parsing_status
        is ParsingStatus.SUCCESS
    )
    assert repository.failure_class is None
    assert repository.parser_lineage == lineage
    assert repository.completed_parsed is not None

    assert result.page_count == 1
    assert result.chunk_count == 1


def test_executor_failure_persists_parser_failure(
    tmp_path: Path,
) -> None:
    repository = FakeRepository()

    lineage = {
        "execution_mode": "ISOLATED_WORKER",
        "timeout_seconds": 0.75,
    }

    service, executor = make_service(
        tmp_path=tmp_path,
        repository=repository,
        execution=ParserExecutionResult(
            parsing_status=(
                ParsingStatus.FAILED
            ),
            failure_class=(
                ParserFailureClass.TIMEOUT
            ),
            parsed_document=None,
            lineage=lineage,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="parser execution failed",
    ):
        service.process(
            ProcessDocumentRequest(
                processing_job_id=(
                    repository
                    .processing_job_id
                ),
            )
        )

    assert executor.calls == 1
    assert repository.overall_status == "FAILED"
    assert (
        repository.parsing_status
        is ParsingStatus.FAILED
    )
    assert (
        repository.failure_class
        is ParserFailureClass.TIMEOUT
    )
    assert repository.parser_lineage == lineage
    assert (
        repository.completed_parsed
        is None
    )


def test_parser_success_remains_success_when_validation_fails(
    tmp_path: Path,
) -> None:
    repository = FakeRepository()

    lineage = {
        "execution_mode": "ISOLATED_WORKER",
        "original_sha256": "abc123",
    }

    service, executor = make_service(
        tmp_path=tmp_path,
        repository=repository,
        execution=ParserExecutionResult(
            parsing_status=(
                ParsingStatus.SUCCESS
            ),
            failure_class=None,
            parsed_document=invalid_document(),
            lineage=lineage,
        ),
    )

    with pytest.raises(
        ParsedDocumentValidationError,
        match="contains no pages",
    ):
        service.process(
            ProcessDocumentRequest(
                processing_job_id=(
                    repository
                    .processing_job_id
                ),
            )
        )

    assert executor.calls == 1

    # Overall processing failed after parsing.
    assert repository.overall_status == "FAILED"

    # Parser truth is retained independently.
    assert (
        repository.parsing_status
        is ParsingStatus.SUCCESS
    )
    assert repository.failure_class is None
    assert repository.parser_lineage == lineage

    assert (
        repository.completed_parsed
        is None
    )
