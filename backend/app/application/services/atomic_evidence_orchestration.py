from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from uuid import UUID, uuid4

from app.application.ports.audit import (
    AuditEvent,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.retrieval import (
    RetrievedChunk,
)
from app.application.services.audit import (
    AuditService,
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
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)


@dataclass(frozen=True)
class AtomicClaim:
    claim_id: str
    text: str
    language: str = "fr"
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

        if not self.language.strip():
            raise ValueError(
                "Atomic claim language cannot be empty."
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
    correlation_id: UUID
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
        audit: AuditService,
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        self._search = search
        self._verification = verification
        self._policy = policy
        self._audit = audit
        self._top_k = top_k

    def evaluate(
        self,
        *,
        identity: CurrentIdentity,
        claims: Sequence[AtomicClaim],
        correlation_id: UUID | None = None,
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

        operation_id = (
            correlation_id
            if correlation_id is not None
            else uuid4()
        )

        required_claim_count = sum(
            1
            for item in items
            if item.required
        )

        # Deliberately contains counts only.
        # Raw query/claim text must never enter
        # the audit store.
        self._audit.record(
            AuditEvent(
                event_type=(
                    AuditEventType
                    .QUERY_SUBMITTED
                ),
                outcome=(
                    AuditOutcome.SUCCEEDED
                ),
                actor_user_id=(
                    identity.user_id
                ),
                resource_type="qa_evidence",
                correlation_id=operation_id,
                metadata={
                    "claim_count": len(items),
                    "required_claim_count": (
                        required_claim_count
                    ),
                },
            )
        )

        claim_results: list[
            AtomicClaimResult
        ] = []

        policy_inputs: list[
            AtomicClaimEvidence
        ] = []

        try:
            for claim in items:
                claim_text = (
                    claim.text.strip()
                )

                # Authorization happens inside
                # AuthorizedSearchService BEFORE
                # embedding-backed retrieval.
                search_result = (
                    self._search.search(
                        identity=identity,
                        query=claim_text,
                        top_k=self._top_k,
                    )
                )

                # Retrieval may be multilingual, but
                # entailment verification is strictly
                # same-language.
                claim_language = (
                    claim.language
                    .strip()
                    .lower()
                )

                same_language_evidence = tuple(
                    item
                    for item in search_result.results
                    if (
                        item.language
                        .strip()
                        .lower()
                        == claim_language
                    )
                )

                # Only authorized, same-language
                # retrieval results may enter NLI.
                verification = (
                    self._verification.verify(
                        claim=claim_text,
                        evidence=(
                            same_language_evidence
                        ),
                    )
                )

                claim_results.append(
                    AtomicClaimResult(
                        claim=claim,
                        retrieved_evidence=(
                            search_result.results
                        ),
                        verification=verification,
                    )
                )

                policy_inputs.append(
                    AtomicClaimEvidence(
                        claim_id=claim.claim_id,
                        verification=verification,
                        required=claim.required,
                    )
                )

            assessment = self._policy.assess(
                policy_inputs
            )

        except Exception:
            # A system/security failure is
            # categorically different from
            # insufficient evidence.
            #
            # Do not persist raw claims, exception
            # messages, prompts, or evidence text.
            self._audit.record(
                AuditEvent(
                    event_type=(
                        AuditEventType
                        .EVIDENCE_FAILED
                    ),
                    outcome=(
                        AuditOutcome.FAILED
                    ),
                    actor_user_id=(
                        identity.user_id
                    ),
                    resource_type="qa_evidence",
                    correlation_id=operation_id,
                    error_code=(
                        "EVIDENCE_EVALUATION_FAILED"
                    ),
                    metadata={
                        "claim_count": len(items),
                        "required_claim_count": (
                            required_claim_count
                        ),
                        "completed_claim_count": (
                            len(claim_results)
                        ),
                    },
                )
            )

            raise

        total_retrieved = sum(
            len(
                item.retrieved_evidence
            )
            for item in claim_results
        )

        event_type = (
            AuditEventType.EVIDENCE_SUFFICIENT
            if assessment.is_sufficient
            else AuditEventType
            .EVIDENCE_INSUFFICIENT
        )

        # Evaluation itself completed normally.
        # Sufficiency is represented by event_type,
        # not by treating a refusal as a system error.
        self._audit.record(
            AuditEvent(
                event_type=event_type,
                outcome=(
                    AuditOutcome.SUCCEEDED
                ),
                actor_user_id=(
                    identity.user_id
                ),
                resource_type="qa_evidence",
                correlation_id=operation_id,
                metadata={
                    "claim_count": len(items),
                    "required_claim_count": (
                        required_claim_count
                    ),
                    "retrieved_result_count": (
                        total_retrieved
                    ),
                    "unsupported_required_claim_count": (
                        len(
                            assessment
                            .unsupported_required_claim_ids
                        )
                    ),
                    "decision": (
                        assessment.decision.value
                    ),
                },
            )
        )

        return AtomicEvidenceOrchestrationResult(
            correlation_id=operation_id,
            claim_results=tuple(
                claim_results
            ),
            assessment=assessment,
        )
