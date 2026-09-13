from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class CitationPresentation:
    page_id: UUID
    document_id: UUID

    document_title: str
    issuing_organization: str
    jurisdiction: str


class CitationPresentationRepository(
    Protocol
):
    def get_by_page_id(
        self,
        *,
        page_id: UUID,
        document_id: UUID,
    ) -> CitationPresentation | None:
        ...
