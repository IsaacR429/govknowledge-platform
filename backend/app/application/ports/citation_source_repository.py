from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class CitationSource:
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    page_id: UUID
    page_number: int
    content: str


class CitationSourceRepository(Protocol):
    def get_by_chunk_id(
        self,
        chunk_id: UUID,
    ) -> CitationSource | None:
        ...
