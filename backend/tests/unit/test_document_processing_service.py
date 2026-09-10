from pathlib import Path
from uuid import uuid4

import pytest

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.application.services.document_processing import (
    DocumentProcessingService,
    ProcessDocumentRequest,
)


class FakeStorage:
    def __init__(self, path: Path):
        self.path = path

    def resolve(self, storage_path: str) -> Path:
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
    def __init__(self):
        self.calls = []

    def mark_processing(
        self,
        *,
        version_id,
        processing_job_id,
    ):
        self.calls.append(
            ("processing", version_id, processing_job_id)
        )

    def complete_processing(
        self,
        *,
        document_id,
        version_id,
        processing_job_id,
        parsed,
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


def make_request():
    return ProcessDocumentRequest(
        document_id=uuid4(),
        version_id=uuid4(),
        processing_job_id=uuid4(),
        storage_path="document.pdf",
        language="fr",
    )


def test_processing_service_completes_document(
    tmp_path,
):
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"test")

    repository = RecordingRepository()

    service = DocumentProcessingService(
        repository=repository,
        storage=FakeStorage(file_path),
        parser=SuccessfulParser(),
    )

    result = service.process(
        make_request()
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


def test_processing_service_marks_failure(
    tmp_path,
):
    file_path = tmp_path / "document.pdf"
    file_path.write_bytes(b"test")

    repository = RecordingRepository()

    service = DocumentProcessingService(
        repository=repository,
        storage=FakeStorage(file_path),
        parser=FailingParser(),
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated parser failure",
    ):
        service.process(
            make_request()
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
