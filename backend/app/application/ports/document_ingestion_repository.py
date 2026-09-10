from dataclasses import dataclass
from datetime import date
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class InitialDocumentUpload:
    document_id: UUID
    version_id: UUID
    processing_job_id: UUID

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
    storage_path: str
    file_hash: str


class DocumentIngestionRepository(Protocol):
    def create_initial_upload(
        self,
        upload: InitialDocumentUpload,
    ) -> None:
        ...
