from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    version_id: UUID
    processing_job_id: UUID
    file_hash: str
    size_bytes: int
    processing_status: Literal["PENDING"] = "PENDING"


class DocumentVersionSummary(BaseModel):
    version_id: UUID
    version_number: int
    original_filename: str
    processing_status: str
    effective_date: date | None
    superseded_at: datetime | None
    created_at: datetime


class DocumentInventoryItem(BaseModel):
    document_id: UUID
    title: str
    description: str | None
    document_type: str | None
    issuing_organization: str
    jurisdiction: str
    language: str
    publication_date: date | None
    source_url: str | None
    status: str
    confidentiality_level: str
    created_at: datetime
    updated_at: datetime
    latest_version: DocumentVersionSummary | None


class DocumentDetailResponse(DocumentInventoryItem):
    versions: list[DocumentVersionSummary]


class DocumentMetadataPatch(BaseModel):
    title: str | None = None
    description: str | None = None
    document_type: str | None = None
    issuing_organization: str | None = None
    jurisdiction: str | None = None
    language: str | None = None
    publication_date: date | None = None
    source_url: str | None = None
    confidentiality_level: str | None = None


class DocumentMetadataUpdateResponse(BaseModel):
    changed: bool
    changed_fields: list[str]
    document: DocumentDetailResponse


class DocumentArchiveResponse(BaseModel):
    document_id: UUID
    status: str
    changed: bool


class AccessibleDocumentItem(BaseModel):
    document_id: UUID
    title: str
    description: str | None
    document_type: str | None
    issuing_organization: str
    jurisdiction: str
    language: str
    publication_date: date | None
    source_url: str | None


class AccessibleDocumentDetailResponse(
    AccessibleDocumentItem
):
    pass
