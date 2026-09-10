from dataclasses import dataclass
from typing import Protocol, Sequence

from app.application.ports.retrieval import (
    RetrievedChunk,
)
from app.domain.enums.evidence import (
    EvidenceDecision,
)


@dataclass(frozen=True)
class EvidenceAssessment:
    decision: EvidenceDecision
    evidence: tuple[RetrievedChunk, ...]
    top_score: float | None
    supporting_count: int

    @property
    def is_sufficient(self) -> bool:
        return (
            self.decision
            == EvidenceDecision.SUFFICIENT
        )


class EvidencePolicy(Protocol):
    def assess(
        self,
        results: Sequence[RetrievedChunk],
    ) -> EvidenceAssessment:
        ...
