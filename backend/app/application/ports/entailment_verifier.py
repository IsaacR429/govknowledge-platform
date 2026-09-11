from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from app.domain.enums.evidence import (
    EntailmentLabel,
)


@dataclass(frozen=True)
class EntailmentScore:
    label: EntailmentLabel
    entailment_probability: float
    neutral_probability: float
    contradiction_probability: float


class EntailmentVerifierPort(Protocol):
    @property
    def model_name(self) -> str:
        ...

    @property
    def model_revision(self) -> str:
        ...

    def score(
        self,
        *,
        claim: str,
        premises: Sequence[str],
    ) -> tuple[EntailmentScore, ...]:
        ...
