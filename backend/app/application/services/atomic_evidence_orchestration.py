from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.retrieval import (
    RetrievedChunk,
)
from app.application.services.atomic_evidence_policy import (
    AtomicClaimEvidence,
    AtomicEvidenceAssessment,
    AtomicEvidencePolicy,
)
from app.application.services.authorized_search import (
    AuthorizedSearchService,
)
from app.application.services.sentence_evidence_verification import (
    EvidenceVerificationResult,
    SentenceEvidenceVerificationService,
)


@dataclass(frozen=True)
class AtomicClaim:
    claim_id: str
    text: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise ValueError(
                "claim_id cannot be empty."
            )

        if not self.text.strip():
            raise ValueError(
                "Atomic claim text cannot be empty."
            )


@dataclass(frozen=True)
class AtomicClaimResult:
    claim: AtomicClaim
    retrieved_evidence: tuple[
        RetrievedChunk,
        ...,
    ]
    verification: EvidenceVerificationResult


@dataclass(frozen=True)
class AtomicEvidenceOrchestrationResult:
    claim_results: tuple[
        AtomicClaimResult,
        ...,
    ]
    assessment: AtomicEvidenceAssessment

    @property
    def is_sufficient(self) -> bool:
        return self.assessment.is_sufficient


class AtomicEvidenceOrchestrationService:
    """
    P0 evidence-grounding orchestration.

    Each atomic claim receives its own
    authorization-first retrieval operation.

    Retrieved evidence is never shared between
    claims before verification.

    Only evidence returned by AuthorizedSearchService
    may enter sentence-level entailment verification.
    """

    def __init__(
        self,
        *,
        search: AuthorizedSearchService,
        verification: (
            SentenceEvidenceVerificationService
        ),
        policy: AtomicEvidencePolicy,
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        self._search = search
        self._verification = verification
        self._policy = policy
        self._top_k = top_k

    def evaluate(
        self,
        *,
        identity: CurrentIdentity,
        claims: Sequence[AtomicClaim],
    ) -> AtomicEvidenceOrchestrationResult:
        items = tuple(claims)

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

        claim_results: list[
            AtomicClaimResult
        ] = []

        policy_inputs: list[
            AtomicClaimEvidence
        ] = []

        for claim in items:
            claim_text = (
                claim.text.strip()
            )

            # Security boundary:
            # authorization happens inside
            # AuthorizedSearchService BEFORE
            # embedding-backed retrieval.
            search_result = (
                self._search.search(
                    identity=identity,
                    query=claim_text,
                    top_k=self._top_k,
                )
            )

            # Only the already-authorized
            # retrieval result can enter NLI.
            verification = (
                self._verification.verify(
                    claim=claim_text,
                    evidence=(
                        search_result.results
                    ),
                )
            )

            claim_results.append(
                AtomicClaimResult(
                    claim=claim,
                    retrieved_evidence=(
                        search_result.results
                    ),
                    verification=(
                        verification
                    ),
                )
            )

            policy_inputs.append(
                AtomicClaimEvidence(
                    claim_id=(
                        claim.claim_id
                    ),
                    verification=(
                        verification
                    ),
                    required=(
                        claim.required
                    ),
                )
            )

        assessment = (
            self._policy.assess(
                policy_inputs
            )
        )

        return (
            AtomicEvidenceOrchestrationResult(
                claim_results=tuple(
                    claim_results
                ),
                assessment=assessment,
            )
        )
