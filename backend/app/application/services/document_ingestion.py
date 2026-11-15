from dataclasses import dataclass
from datetime import date
from typing import BinaryIO
from uuid import UUID, uuid4

from app.application.ports.audit import AuditEvent
from app.application.ports.document_ingestion_repository import (
    DocumentIngestionRepository,
    InitialDocumentUpload,
)
from app.application.ports.file_storage import FileStorage
from app.application.services.audit import AuditService
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)


@dataclass(frozen=True)
class CreateDocumentRequest:
    title: str
    description: str | None
    document_type: str | None
    issuing_organization: str
    jurisdiction: str
    language: str
    publication_date: date | None
    source_url: str | None
    confidentiality_level: str
    original_filename: str


@dataclass(frozen=True)
class CreateDocumentResult:
    document_id: UUID
    version_id: UUID
    processing_job_id: UUID
    file_hash: str
    size_bytes: int


class DocumentIngestionService:
    def __init__(
        self,
        *,
        repository: DocumentIngestionRepository,
        storage: FileStorage,
        audit: AuditService,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._audit = audit

    def create_initial_upload(
        self,
        request: CreateDocumentRequest,
        source: BinaryIO,
        *,
        actor_user_id: UUID,
        correlation_id: UUID,
        content_type: str = "application/pdf",
    ) -> CreateDocumentResult:
        document_id = uuid4()
        version_id = uuid4()
        processing_job_id = uuid4()

        stored = self._storage.store_document_version(
            document_id=document_id,
            version_id=version_id,
            original_filename=(
                request.original_filename
            ),
            source=source,
        )

        try:
            self._repository.create_initial_upload(
                InitialDocumentUpload(
                    document_id=document_id,
                    version_id=version_id,
                    processing_job_id=(
                        processing_job_id
                    ),
                    title=request.title,
                    description=request.description,
                    document_type=(
                        request.document_type
                    ),
                    issuing_organization=(
                        request.issuing_organization
                    ),
                    jurisdiction=request.jurisdiction,
                    language=request.language,
                    publication_date=(
                        request.publication_date
                    ),
                    source_url=request.source_url,
                    confidentiality_level=(
                        request.confidentiality_level
                    ),
                    original_filename=(
                        request.original_filename
                    ),
                    storage_path=str(
                        stored.relative_path
                    ),
                    file_hash=stored.sha256,
                )
            )

            self._audit.record(
                AuditEvent(
                    event_type=(
                        AuditEventType.DOCUMENT_INGESTED
                    ),
                    outcome=AuditOutcome.SUCCEEDED,
                    actor_user_id=actor_user_id,
                    resource_type="DOCUMENT",
                    resource_id=document_id,
                    document_id=document_id,
                    correlation_id=correlation_id,
                    metadata={
                        "document_version_id": str(
                            version_id
                        ),
                        "processing_job_id": str(
                            processing_job_id
                        ),
                        "file_hash": stored.sha256,
                        "size_bytes": stored.size_bytes,
                        "content_type": content_type,
                    },
                )
            )

        except Exception:
            self._storage.delete(
                str(stored.relative_path)
            )
            raise

        return CreateDocumentResult(
            document_id=document_id,
            version_id=version_id,
            processing_job_id=processing_job_id,
            file_hash=stored.sha256,
            size_bytes=stored.size_bytes,
        )
