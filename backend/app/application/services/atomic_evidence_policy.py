from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.application.services.sentence_evidence_verification import (
    EvidenceSpan,
    EvidenceVerificationResult,
)
from app.domain.enums.evidence import (
    EvidenceDecision,
)


@dataclass(frozen=True)
class AtomicClaimEvidence:
    claim_id: str
    verification: EvidenceVerificationResult
    required: bool = True

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise ValueError(
                "claim_id cannot be empty."
            )


@dataclass(frozen=True)
class SelectedClaimEvidence:
    claim_id: str
    claim: str
    span: EvidenceSpan


@dataclass(frozen=True)
class AtomicEvidenceAssessment:
    decision: EvidenceDecision
    selected_evidence: tuple[
        SelectedClaimEvidence,
        ...,
    ]
    unsupported_required_claim_ids: tuple[
        str,
        ...,
    ]
    weakest_required_support_probability: (
        float | None
    )

    @property
    def is_sufficient(self) -> bool:
        return (
            self.decision
            == EvidenceDecision.SUFFICIENT
        )


class AtomicEvidencePolicy:
    """
    Conservative P0 factual-support policy.

    Every required atomic claim must have at least
    one independently entailing evidence sentence.

    No cross-sentence inference is performed here.
    No retrieval-score threshold is used here.
    """

    def assess(
        self,
        claims: Sequence[
            AtomicClaimEvidence
        ],
    ) -> AtomicEvidenceAssessment:
        items = tuple(claims)

        if not items:
            return AtomicEvidenceAssessment(
                decision=(
                    EvidenceDecision.NO_EVIDENCE
                ),
                selected_evidence=(),
                unsupported_required_claim_ids=(),
                weakest_required_support_probability=None,
            )

        claim_ids = tuple(
            item.claim_id
            for item in items
        )

        if len(set(claim_ids)) != len(
            claim_ids
        ):
            raise ValueError(
                "claim_id values must be unique."
            )

        required = tuple(
            item
            for item in items
            if item.required
        )

        if not required:
            raise ValueError(
                "At least one required atomic "
                "claim is necessary."
            )

        has_any_required_evidence = any(
            item.verification.spans
            for item in required
        )

        if not has_any_required_evidence:
            return AtomicEvidenceAssessment(
                decision=(
                    EvidenceDecision.NO_EVIDENCE
                ),
                selected_evidence=(),
                unsupported_required_claim_ids=tuple(
                    item.claim_id
                    for item in required
                ),
                weakest_required_support_probability=None,
            )

        selected: list[
            SelectedClaimEvidence
        ] = []

        unsupported: list[str] = []

        for item in required:
            supporting = (
                item.verification
                .supporting_spans
            )

            if not supporting:
                unsupported.append(
                    item.claim_id
                )

                continue

            # One exact source sentence is enough
            # for P0 support. If several sentences
            # independently entail the claim, retain
            # the strongest one for the evidence trace.
            best = max(
                supporting,
                key=lambda span: (
                    span.verification
                    .entailment_probability
                ),
            )

            selected.append(
                SelectedClaimEvidence(
                    claim_id=item.claim_id,
                    claim=(
                        item.verification
                        .claim
                    ),
                    span=best,
                )
            )

        if unsupported:
            return AtomicEvidenceAssessment(
                decision=(
                    EvidenceDecision
                    .INSUFFICIENT_SUPPORT
                ),
                selected_evidence=tuple(
                    selected
                ),
                unsupported_required_claim_ids=tuple(
                    unsupported
                ),
                weakest_required_support_probability=None,
            )

        weakest = min(
            item.span.verification
            .entailment_probability
            for item in selected
        )

        return AtomicEvidenceAssessment(
            decision=(
                EvidenceDecision.SUFFICIENT
            ),
            selected_evidence=tuple(
                selected
            ),
            unsupported_required_claim_ids=(),
            weakest_required_support_probability=(
                weakest
            ),
        )
