from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class ClaimOrigin:
    """
    Backend-validated provenance for an atomic claim.

    This object must never be constructed directly
    from untrusted planner-supplied provenance fields.
    """

    evidence_id: str

    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    page_id: UUID
    page_number: int

    start_char: int
    end_char: int
    text: str
    language: str

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError(
                "evidence_id cannot be empty."
            )

        if self.page_number < 1:
            raise ValueError(
                "page_number must be positive."
            )

        if self.start_char < 0:
            raise ValueError(
                "start_char cannot be negative."
            )

        if self.end_char <= self.start_char:
            raise ValueError(
                "end_char must follow start_char."
            )

        if not self.text:
            raise ValueError(
                "Claim origin text cannot be empty."
            )

        if not self.language.strip():
            raise ValueError(
                "Claim origin language cannot be empty."
            )
