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

    def generate_comparison(
        self,
        *,
        identity: CurrentIdentity,
        question: str,
        claims: tuple[
            TableAtomicClaim,
            ...,
        ],
        correlation_id: UUID | None = None,
        language: str = "fr",
    ) -> GroundedAnswerResult:
        """
        Deterministic comparison answer from multiple
        independently validated table claims.

        Disclosure is built only from revalidated
        canonical citations, never directly from the
        incoming in-memory claim text.
        """

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

        if (
            not isinstance(
                claims,
                tuple,
            )
            or len(claims) < 2
        ):
            raise ValueError(
                "Comparison requires at least "
                "two table claims."
            )

        for claim in claims:
            if not isinstance(
                claim,
                TableAtomicClaim,
            ):
                raise ValueError(
                    "Comparison contains an "
                    "invalid table claim."
                )

        operation_id = (
            correlation_id
            if correlation_id is not None
            else uuid4()
        )

        # --------------------------------------------
        # Revalidate every claim independently under
        # the current identity / ACL.
        # --------------------------------------------

        validated_pairs = []

        for claim in claims:

            citation = (
                self._citation_validator
                .validate(
                    identity=identity,
                    claim=claim,
                )
            )

            validated_pairs.append(
                (
                    claim,
                    citation,
                )
            )

        # --------------------------------------------
        # Comparison integrity:
        # all citations must still describe one
        # canonical logical row in one document
        # version, with distinct named columns.
        # --------------------------------------------

        row_identities = {
            (
                citation.document_id,
                citation.document_version_id,
                citation.logical_key,
            )
            for _, citation
            in validated_pairs
        }

        if len(row_identities) != 1:
            raise ValueError(
                "Comparison citations do not "
                "belong to one logical row."
            )

        column_names = tuple(
            citation.column_name
            for _, citation
            in validated_pairs
        )

        if any(
            (
                not isinstance(
                    column_name,
                    str,
                )
                or not column_name.strip()
            )
            for column_name
            in column_names
        ):
            raise ValueError(
                "Comparison citation is missing "
                "an authorized column name."
            )

        normalized_columns = tuple(
            column_name.strip().casefold()
            for column_name
            in column_names
        )

        if (
            len(set(normalized_columns))
            != len(normalized_columns)
        ):
            raise ValueError(
                "Comparison citations must use "
                "distinct columns."
            )

        cited_texts = tuple(
            citation.cited_text
            for _, citation
            in validated_pairs
        )

        if any(
            (
                not isinstance(
                    cited_text,
                    str,
                )
                or not cited_text.strip()
            )
            for cited_text
            in cited_texts
        ):
            raise ValueError(
                "Comparison citation contains "
                "empty canonical text."
            )

        validated = (
            ValidatedCitationSet(
                citations=tuple(
                    ClaimCitation(
                        claim_id=claim.claim_id,
                        claim=claim.text,
                        citation=citation,
                    )
                    for claim, citation
                    in validated_pairs
                )
            )
        )

        # --------------------------------------------
        # Compose only from revalidated canonical
        # citation fields.
        # --------------------------------------------

        answer = "; ".join(
            (
                f"{citation.column_name}: "
                f"{citation.cited_text}"
            )
            for _, citation
            in validated_pairs
        )

        return GroundedAnswerResult(
            correlation_id=operation_id,
            status="SUPPORTED",
            answer=answer,
            citations=validated,
            cited_evidence_ids=tuple(
                claim.claim_id
                for claim, _
                in validated_pairs
            ),
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
