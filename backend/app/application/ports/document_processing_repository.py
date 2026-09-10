from typing import Protocol
from uuid import UUID

from app.application.ports.document_parser import (
    ParsedDocument,
)


class DocumentProcessingRepository(Protocol):
    def mark_processing(
        self,
        *,
        version_id: UUID,
        processing_job_id: UUID,
    ) -> None:
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
