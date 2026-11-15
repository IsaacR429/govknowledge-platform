from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class ClaimPlanningEvidence:
    """
    Authorized candidate evidence available to the
    untrusted claim planner.

    evidence_id is request-local and is the only
    provenance identifier the planner may select.

    chunk_id remains backend-owned provenance.
    """

    evidence_id: str
    chunk_id: UUID
    language: str
    content: str

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError(
                "evidence_id cannot be empty."
            )


@dataclass(frozen=True)
class QuestionClaimPlanningRequest:
    question: str
    question_language: str
    source_language: str

    evidence: tuple[
        ClaimPlanningEvidence,
        ...,
    ]


@dataclass(frozen=True)
class PlannedClaim:
    """
    Untrusted planner selection.

    The planner identifies authorized evidence and
    quotes exact text from it. The backend must
    independently validate both fields before the
    claim becomes trusted provenance.
    """

    evidence_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError(
                "evidence_id cannot be empty."
            )

        if not self.text.strip():
            raise ValueError(
                "Planned claim text cannot be empty."
            )


class QuestionClaimPlanner(Protocol):
    def plan(
        self,
        request: QuestionClaimPlanningRequest,
    ) -> tuple[
        PlannedClaim,
        ...,
    ]:
        ...
