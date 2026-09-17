from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.application.ports.authorization import (
    AuthorizationScope,
)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    page_id: UUID
    page_number: int
    chunk_index: int
    content: str
    section_title: str | None
    language: str
    score: float
    # None means page-range provenance is unavailable.
    page_end: int | None = None


class RetrievalPort(Protocol):
    def search_authorized(
        self,
        *,
        scope: AuthorizationScope,
        query_vector: tuple[float, ...],
        model_name: str,
        model_revision: str,
        top_k: int,
    ) -> tuple[RetrievedChunk, ...]:
        ...
