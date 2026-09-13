from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class ClaimPlanningEvidence:
    """
    Authorized candidate evidence that may be used
    only for planning atomic claims.

    This evidence has not yet passed the final
    entailment/evidence-policy boundary.
    """

    chunk_id: UUID
    language: str
    content: str


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
    Untrusted candidate claim produced by a planner.

    It must still pass authorized retrieval,
    same-language NLI, and AtomicEvidencePolicy.
    """

    text: str


class QuestionClaimPlanner(Protocol):
    def plan(
        self,
        request: QuestionClaimPlanningRequest,
    ) -> tuple[
        PlannedClaim,
        ...,
    ]:
        ...
