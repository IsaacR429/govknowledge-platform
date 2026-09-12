from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from app.application.ports.retrieval import (
    RetrievedChunk,
)


@dataclass(frozen=True)
class EvidenceSupportScore:
    chunk: RetrievedChunk
    support_score: float


class EvidenceSupportVerifierPort(Protocol):
    @property
    def model_name(self) -> str:
        ...

    @property
    def model_revision(self) -> str:
        ...

    def score(
        self,
        *,
        query: str,
        evidence: Sequence[RetrievedChunk],
    ) -> tuple[EvidenceSupportScore, ...]:
        ...
