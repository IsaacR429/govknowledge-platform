from __future__ import annotations

from dataclasses import dataclass

from app.application.services.atomic_evidence_orchestration import (
    AtomicEvidenceOrchestrationResult,
)
from app.application.services.citation_mapping import (
    Citation,
    ClaimOriginCitationMapper,
)
from app.application.services.citation_validation import (
    CitationValidator,
    InvalidCitationError,
)
from app.application.services.table_citation_validation import (
    TableCitation,
)

class InsufficientEvidenceForCitationsError(
    Exception
):
    pass


@dataclass(frozen=True)
class ClaimCitation:
    claim_id: str
    claim: str
    citation: Citation | TableCitation
@dataclass(frozen=True)
class ValidatedCitationSet:
    citations: tuple[
        ClaimCitation,
        ...,
    ]


class ValidatedCitationService:
    def __init__(
        self,
        *,
        mapper: ClaimOriginCitationMapper,
        validator: CitationValidator,
    ) -> None:
        self._mapper = mapper
        self._validator = validator

    def build(
        self,
        evidence_result: AtomicEvidenceOrchestrationResult,
    ) -> ValidatedCitationSet:
        assessment = evidence_result.assessment

        if not assessment.is_sufficient:
            raise (
                InsufficientEvidenceForCitationsError(
                    "Validated answer citations "
                    "require sufficient evidence."
                )
            )

        claims_by_id = {
            result.claim.claim_id: result.claim
            for result
            in evidence_result.claim_results
        }

        if (
            len(claims_by_id)
            != len(evidence_result.claim_results)
        ):
            raise InvalidCitationError(
                "Citation provenance is invalid."
            )

        citations: list[
            ClaimCitation
        ] = []

        for selected in (
            assessment.selected_evidence
        ):
            claim = claims_by_id.get(
                selected.claim_id
            )

            if claim is None:
                raise InvalidCitationError(
                    "Citation provenance is invalid."
                )

            # Defense in depth:
            # the policy must have assessed the same
            # claim whose origin is being cited.
            if selected.claim != claim.text:
                raise InvalidCitationError(
                    "Citation provenance is invalid."
                )

            if claim.origin.text != claim.text:
                raise InvalidCitationError(
                    "Citation provenance is invalid."
                )

            citation = self._mapper.map(
                claim.origin
            )

            validated = self._validator.validate(
                citation
            )

            citations.append(
                ClaimCitation(
                    claim_id=selected.claim_id,
                    claim=claim.text,
                    citation=validated,
                )
            )

        return ValidatedCitationSet(
            citations=tuple(citations)
        )
