from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.application.ports.document_parser import (
    ParsedDocument,
)


@dataclass(frozen=True)
class ProcessingJobAttempt:
    processing_job_id: UUID
    version_id: UUID
    attempt_number: int


@dataclass(frozen=True)
class ProcessingTarget:
    processing_job_id: UUID
    version_id: UUID
    document_id: UUID
    storage_path: str
    language: str


class DocumentProcessingRepository(Protocol):
    def create_retry_job(
        self,
        *,
        version_id: UUID,
    ) -> ProcessingJobAttempt:
        ...

    def mark_processing(
        self,
        *,
        processing_job_id: UUID,
    ) -> ProcessingTarget:
        ...

    def complete_processing(
        self,
        *,
        document_id: UUID,
        version_id: UUID,
        processing_job_id: UUID,
        parsed: ParsedDocument,
    ) -> None:
        ...

    def fail_processing(
        self,
        *,
        version_id: UUID,
        processing_job_id: UUID,
        error_code: str,
        error_message: str,
    ) -> None:
        ...
