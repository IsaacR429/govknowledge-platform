from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, Protocol
from uuid import UUID


@dataclass(frozen=True)
class SourcePage:
    page_id: UUID
    document_id: UUID
    document_version_id: UUID

    page_number: int
    extracted_text: str
    metadata: Mapping[str, object]

    document_title: str
    issuing_organization: str
    jurisdiction: str
    document_language: str
    publication_date: date | None
    source_url: str | None

    version_number: int
    original_filename: str


class SourcePageRepository(Protocol):
    def get_by_id(
        self,
        page_id: UUID,
    ) -> SourcePage | None:
        ...
