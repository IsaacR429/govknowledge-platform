from dataclasses import dataclass
from uuid import UUID

from app.application.ports.document_processing_repository import (
    DocumentProcessingRepository,
)


@dataclass(frozen=True)
class RetryDocumentProcessingResult:
    version_id: UUID
    processing_job_id: UUID
    attempt_number: int


class DocumentProcessingRetryService:
    def __init__(
        self,
        *,
        repository: DocumentProcessingRepository,
    ) -> None:
        self._repository = repository

    def create_retry(
        self,
        *,
        version_id: UUID,
    ) -> RetryDocumentProcessingResult:
        attempt = (
            self._repository.create_retry_job(
                version_id=version_id
            )
        )

        return RetryDocumentProcessingResult(
            version_id=attempt.version_id,
            processing_job_id=(
                attempt.processing_job_id
            ),
            attempt_number=(
                attempt.attempt_number
            ),
        )
