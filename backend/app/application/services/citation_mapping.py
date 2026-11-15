from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.application.services.claim_provenance import (
    ClaimOrigin,
)
from app.application.services.sentence_evidence_verification import (
    EvidenceSpan,
)


@dataclass(frozen=True)
class Citation:
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    page_id: UUID
    page_number: int

    start_char: int
    end_char: int
    cited_text: str


class EvidenceSpanCitationMapper:
    def map(
        self,
        span: EvidenceSpan,
    ) -> Citation:
        chunk = span.chunk

        if span.start_char < 0:
            raise ValueError(
                "Citation start offset cannot be negative."
            )

        if span.end_char <= span.start_char:
            raise ValueError(
                "Citation end offset must follow start offset."
            )

        if span.end_char > len(chunk.content):
            raise ValueError(
                "Citation offsets exceed chunk content."
            )

        canonical_text = chunk.content[
            span.start_char:span.end_char
        ]

        if canonical_text != span.text:
            raise ValueError(
                "Evidence span does not match chunk content."
            )

        if chunk.page_number < 1:
            raise ValueError(
                "Citation page number must be positive."
            )

        return Citation(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_version_id=(
                chunk.document_version_id
            ),
            page_id=chunk.page_id,
            page_number=chunk.page_number,
            start_char=span.start_char,
            end_char=span.end_char,
            cited_text=span.text,
        )


class ClaimOriginCitationMapper:
    def map(
        self,
        origin: ClaimOrigin,
    ) -> Citation:
        if origin.start_char < 0:
            raise ValueError(
                "Citation start offset cannot be negative."
            )

        if origin.end_char <= origin.start_char:
            raise ValueError(
                "Citation end offset must follow start offset."
            )

        if (
            origin.end_char - origin.start_char
            != len(origin.text)
        ):
            raise ValueError(
                "Claim origin offsets do not match its text."
            )

        return Citation(
            chunk_id=origin.chunk_id,
            document_id=origin.document_id,
            document_version_id=(
                origin.document_version_id
            ),
            page_id=origin.page_id,
            page_number=origin.page_number,
            start_char=origin.start_char,
            end_char=origin.end_char,
            cited_text=origin.text,
        )
