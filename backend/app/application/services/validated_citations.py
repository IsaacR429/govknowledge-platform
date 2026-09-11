from __future__ import annotations

from dataclasses import dataclass

from app.application.services.atomic_evidence_policy import (
    AtomicEvidenceAssessment,
)
from app.application.services.citation_mapping import (
    Citation,
    EvidenceSpanCitationMapper,
)
from app.application.services.citation_validation import (
    CitationValidator,
)


class InsufficientEvidenceForCitationsError(
    Exception
):
    pass


@dataclass(frozen=True)
class ClaimCitation:
    claim_id: str
    claim: str
    citation: Citation


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
        mapper: EvidenceSpanCitationMapper,
        validator: CitationValidator,
    ) -> None:
        self._mapper = mapper
        self._validator = validator

    def build(
        self,
        assessment: AtomicEvidenceAssessment,
    ) -> ValidatedCitationSet:
        if not assessment.is_sufficient:
            raise (
                InsufficientEvidenceForCitationsError(
                    "Validated answer citations "
                    "require sufficient evidence."
                )
            )

        citations: list[
            ClaimCitation
        ] = []

        for selected in (
            assessment.selected_evidence
        ):
            citation = self._mapper.map(
                selected.span
            )

            validated = (
                self._validator.validate(
                    citation
                )
            )

            citations.append(
                ClaimCitation(
                    claim_id=(
                        selected.claim_id
                    ),
                    claim=selected.claim,
                    citation=validated,
                )
            )

        return ValidatedCitationSet(
            citations=tuple(citations)
        )
