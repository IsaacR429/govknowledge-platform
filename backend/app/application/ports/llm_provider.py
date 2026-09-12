from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class LLMGroundingEvidence:
    """
    Evidence allowed to cross the LLM boundary.

    Every item must originate from a validated
    canonical citation.
    """

    evidence_id: str
    claim: str

    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    page_id: UUID
    page_number: int

    start_char: int
    end_char: int
    cited_text: str


@dataclass(frozen=True)
class LLMGenerationRequest:
    """
    Provider-independent grounded generation request.

    Raw retrieval results must never appear here.
    """

    correlation_id: UUID
    question: str
    language: str

    evidence: tuple[
        LLMGroundingEvidence,
        ...,
    ]


@dataclass(frozen=True)
class LLMGenerationResponse:
    """
    Provider response before application validation.

    citation_ids must refer only to evidence_id values
    supplied in the corresponding request.
    """

    answer: str

    citation_ids: tuple[
        str,
        ...,
    ]


class LLMProvider(Protocol):
    def generate(
        self,
        request: LLMGenerationRequest,
    ) -> LLMGenerationResponse:
        ...
