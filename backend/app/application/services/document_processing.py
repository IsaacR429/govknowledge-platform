from dataclasses import dataclass
from uuid import UUID

from app.application.ports.document_parser import DocumentParser
from app.application.ports.document_processing_repository import (
    DocumentProcessingRepository,
)
from app.application.ports.file_storage import FileStorage


@dataclass(frozen=True)
class ProcessDocumentRequest:
    document_id: UUID
    version_id: UUID
    processing_job_id: UUID
    storage_path: str
    language: str


@dataclass(frozen=True)
class ProcessDocumentResult:
    document_id: UUID
    version_id: UUID
    processing_job_id: UUID
    page_count: int
    chunk_count: int


class DocumentProcessingService:
    def __init__(
        self,
        *,
        repository: DocumentProcessingRepository,
        storage: FileStorage,
        parser: DocumentParser,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._parser = parser

    def process(
        self,
        request: ProcessDocumentRequest,
    ) -> ProcessDocumentResult:
        self._repository.mark_processing(
            version_id=request.version_id,
            processing_job_id=request.processing_job_id,
        )

        try:
            file_path = self._storage.resolve(
                request.storage_path
            )

            parsed = self._parser.parse(
                file_path,
                language=request.language,
            )

            if not parsed.pages:
                raise ValueError(
                    "Parsed document contains no pages."
                )

            self._repository.complete_processing(
                document_id=request.document_id,
                version_id=request.version_id,
                processing_job_id=request.processing_job_id,
                parsed=parsed,
            )

        except Exception as exc:
            self._repository.fail_processing(
                version_id=request.version_id,
                processing_job_id=request.processing_job_id,
                error_code="DOCUMENT_PROCESSING_FAILED",
                error_message=str(exc),
            )
            raise

        return ProcessDocumentResult(
            document_id=request.document_id,
            version_id=request.version_id,
            processing_job_id=request.processing_job_id,
            page_count=len(parsed.pages),
            chunk_count=len(parsed.chunks),
        )
