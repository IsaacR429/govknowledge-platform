from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from app.application.ports.evidence_policy import (
    EvidenceAssessment,
    EvidencePolicy,
)
from app.application.ports.retrieval import (
    RetrievedChunk,
)
from app.domain.enums.evidence import (
    EvidenceDecision,
)


@dataclass(frozen=True)
class EvidencePolicyConfig:
    min_top_score: float
    min_supporting_score: float
    min_supporting_chunks: int
    max_evidence_chunks: int

    def __post_init__(self) -> None:
        if not (
            -1.0
            <= self.min_top_score
            <= 1.0
        ):
            raise ValueError(
                "min_top_score must be "
                "between -1 and 1."
            )

        if not (
            -1.0
            <= self.min_supporting_score
            <= 1.0
        ):
            raise ValueError(
                "min_supporting_score must be "
                "between -1 and 1."
            )

        if (
            self.min_supporting_score
            > self.min_top_score
        ):
            raise ValueError(
                "min_supporting_score cannot "
                "exceed min_top_score."
            )

        if self.min_supporting_chunks < 1:
            raise ValueError(
                "min_supporting_chunks must "
                "be positive."
            )

        if (
            self.max_evidence_chunks
            < self.min_supporting_chunks
        ):
            raise ValueError(
                "max_evidence_chunks must be "
                "at least min_supporting_chunks."
            )


class ThresholdEvidencePolicy(
    EvidencePolicy
):
    """Deterministic P0 evidence sufficiency policy."""

    def __init__(
        self,
        config: EvidencePolicyConfig,
    ) -> None:
        self._config = config

    def assess(
        self,
        results: Sequence[RetrievedChunk],
    ) -> EvidenceAssessment:
        if not results:
            return EvidenceAssessment(
                decision=(
                    EvidenceDecision.NO_EVIDENCE
                ),
                evidence=(),
                top_score=None,
                supporting_count=0,
            )

        for result in results:
            if not math.isfinite(
                result.score
            ):
                raise ValueError(
                    "Evidence score must be finite."
                )

        ranked = sorted(
            results,
            key=lambda item: item.score,
            reverse=True,
        )

        top_score = ranked[0].score

        if (
            top_score
            < self._config.min_top_score
        ):
            return EvidenceAssessment(
                decision=(
                    EvidenceDecision
                    .TOP_SCORE_TOO_LOW
                ),
                evidence=(),
                top_score=top_score,
                supporting_count=0,
            )

        supporting = tuple(
            result
            for result in ranked
            if (
                result.score
                >= self._config
                .min_supporting_score
            )
        )

        if (
            len(supporting)
            < self._config
            .min_supporting_chunks
        ):
            return EvidenceAssessment(
                decision=(
                    EvidenceDecision
                    .INSUFFICIENT_SUPPORT
                ),
                evidence=(),
                top_score=top_score,
                supporting_count=len(
                    supporting
                ),
            )

        selected = supporting[
            : self._config.max_evidence_chunks
        ]

        return EvidenceAssessment(
            decision=(
                EvidenceDecision.SUFFICIENT
            ),
            evidence=selected,
            top_score=top_score,
            supporting_count=len(
                supporting
            ),
        )
