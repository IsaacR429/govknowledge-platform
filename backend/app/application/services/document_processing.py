from dataclasses import dataclass
from uuid import UUID

from app.application.ports.audit import (
    AuditEvent,
)
from app.application.ports.document_parser import (
    DocumentParser,
)
from app.application.ports.document_processing_repository import (
    DocumentProcessingRepository,
    ProcessingTarget,
)
from app.application.ports.file_storage import (
    FileStorage,
)
from app.application.services.audit import (
    AuditPersistenceError,
    AuditService,
)
from app.application.services.parsed_document_validation import (
    validate_parsed_document,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)


@dataclass(frozen=True)
class ProcessDocumentRequest:
    processing_job_id: UUID


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
        audit: AuditService,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._parser = parser
        self._audit = audit

    def _record_lifecycle_event(
        self,
        *,
        event_type: AuditEventType,
        outcome: AuditOutcome,
        target: ProcessingTarget,
        error_code: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        try:
            self._audit.record(
                AuditEvent(
                    event_type=event_type,
                    outcome=outcome,
                    actor_user_id=None,
                    resource_type=(
                        "document_processing_job"
                    ),
                    resource_id=(
                        target.processing_job_id
                    ),
                    document_id=(
                        target.document_id
                    ),
                    correlation_id=(
                        target.processing_job_id
                    ),
                    error_code=error_code,
                    metadata={
                        "document_version_id": str(
                            target.version_id
                        ),
                        **(metadata or {}),
                    },
                )
            )

        except AuditPersistenceError:
            # Processing state is independently
            # durable. Audit outage must not
            # corrupt an already committed state.
            pass

    def process(
        self,
        request: ProcessDocumentRequest,
    ) -> ProcessDocumentResult:
        target = (
            self._repository.mark_processing(
                processing_job_id=(
                    request.processing_job_id
                ),
            )
        )

        self._record_lifecycle_event(
            event_type=(
                AuditEventType
                .DOCUMENT_PROCESSING_STARTED
            ),
            outcome=AuditOutcome.SUCCEEDED,
            target=target,
        )

        try:
            file_path = self._storage.resolve(
                target.storage_path
            )

            parsed = self._parser.parse(
                file_path,
                language=target.language,
            )

            validate_parsed_document(
                parsed,
                expected_language=(
                    target.language
                ),
            )

            self._repository.complete_processing(
                document_id=target.document_id,
                version_id=target.version_id,
                processing_job_id=(
                    target.processing_job_id
                ),
                parsed=parsed,
            )

        except Exception:
            self._repository.fail_processing(
                version_id=target.version_id,
                processing_job_id=(
                    target.processing_job_id
                ),
                error_code=(
                    "DOCUMENT_PROCESSING_FAILED"
                ),
                error_message=(
                    "Document processing failed."
                ),
            )

            self._record_lifecycle_event(
                event_type=(
                    AuditEventType
                    .DOCUMENT_PROCESSING_FAILED
                ),
                outcome=AuditOutcome.FAILED,
                target=target,
                error_code=(
                    "DOCUMENT_PROCESSING_FAILED"
                ),
            )

            raise

        result = ProcessDocumentResult(
            document_id=target.document_id,
            version_id=target.version_id,
            processing_job_id=(
                target.processing_job_id
            ),
            page_count=len(parsed.pages),
            chunk_count=len(parsed.chunks),
        )

        self._record_lifecycle_event(
            event_type=(
                AuditEventType
                .DOCUMENT_PROCESSING_COMPLETED
            ),
            outcome=AuditOutcome.SUCCEEDED,
            target=target,
            metadata={
                "page_count": result.page_count,
                "chunk_count": result.chunk_count,
            },
        )

        return result
