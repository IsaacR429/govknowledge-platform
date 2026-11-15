from __future__ import annotations

from uuid import (
    UUID,
    uuid4,
)

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.grounded_answer_generation import (
    FRENCH_INSUFFICIENT_EVIDENCE_REFUSAL,
    GroundedAnswerResult,
)
from app.application.services.table_citation_validation import (
    TableClaimCitationValidator,
)
from app.application.services.table_claim_provenance import (
    TableAtomicClaim,
)
from app.application.services.validated_citations import (
    ClaimCitation,
    ValidatedCitationSet,
)


class TableGroundedAnswerGenerationService:
    """
    Deterministic P0 answer generation for an exact,
    already-selected structured table cell.

    The answer comes only from the canonical citation.
    No LLM is used for this exact-cell path.
    """

    def __init__(
        self,
        *,
        citation_validator: (
            TableClaimCitationValidator
        ),
    ) -> None:
        self._citation_validator = (
            citation_validator
        )

    def generate(
        self,
        *,
        identity: CurrentIdentity,
        question: str,
        claim: TableAtomicClaim | None,
        correlation_id: UUID | None = None,
        language: str = "fr",
    ) -> GroundedAnswerResult:

        clean_question = (
            question.strip()
        )

        clean_language = (
            language.strip()
        )

        if not clean_question:
            raise ValueError(
                "question cannot be empty."
            )

        if not clean_language:
            raise ValueError(
                "language cannot be empty."
            )

        operation_id = (
            correlation_id
            if correlation_id is not None
            else uuid4()
        )

        # --------------------------------------------
        # No canonical table cell => refusal.
        # --------------------------------------------

        if claim is None:
            return GroundedAnswerResult(
                correlation_id=(
                    operation_id
                ),
                status=(
                    "INSUFFICIENT_EVIDENCE"
                ),
                answer=(
                    FRENCH_INSUFFICIENT_EVIDENCE_REFUSAL
                ),
                citations=(
                    ValidatedCitationSet(
                        citations=()
                    )
                ),
                cited_evidence_ids=(),
            )

        # --------------------------------------------
        # Re-read canonical evidence under ACL.
        # --------------------------------------------

        citation = (
            self._citation_validator
            .validate(
                identity=identity,
                claim=claim,
            )
        )

        validated = (
            ValidatedCitationSet(
                citations=(
                    ClaimCitation(
                        claim_id=(
                            claim.claim_id
                        ),
                        claim=(
                            claim.text
                        ),
                        citation=(
                            citation
                        ),
                    ),
                )
            )
        )

        # --------------------------------------------
        # Exact-cell answer.
        #
        # Use the revalidated canonical citation,
        # not the original in-memory claim.
        # --------------------------------------------

        return GroundedAnswerResult(
            correlation_id=(
                operation_id
            ),
            status="SUPPORTED",
            answer=(
                citation.cited_text
            ),
            citations=validated,
            cited_evidence_ids=(
                claim.claim_id,
            ),
        )
