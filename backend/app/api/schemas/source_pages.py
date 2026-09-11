from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class SourcePageResponse(BaseModel):
    page_id: UUID
    document_id: UUID
    document_version_id: UUID

    page_number: int
    extracted_text: str
    metadata: dict[str, Any]

    document_title: str
    issuing_organization: str
    jurisdiction: str
    document_language: str
    publication_date: date | None
    source_url: str | None

    version_number: int
    original_filename: str
