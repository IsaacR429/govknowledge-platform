from uuid import UUID

from pydantic import BaseModel


class SearchResultItem(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    version_number: int
    page_id: UUID
    page_number: int
    page_end: int | None
    chunk_index: int

    document_title: str
    issuing_organization: str
    jurisdiction: str

    section_title: str | None
    language: str
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[SearchResultItem]
