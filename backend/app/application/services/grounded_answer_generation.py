from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from app.application.ports.llm_provider import (
    LLMGenerationRequest,
    LLMGenerationResponse,
    LLMGroundingEvidence,
    LLMProvider,
)
from app.application.services.atomic_evidence_orchestration import (
    AtomicEvidenceOrchestrationResult,
)
from app.application.services.validated_citations import (
    ValidatedCitationService,
    ValidatedCitationSet,
)


FRENCH_INSUFFICIENT_EVIDENCE_REFUSAL = (
    "Les sources autorisées disponibles ne permettent "
    "pas de répondre de manière fiable à cette question."
)


class InvalidGeneratedCitationError(
    ValueError
):
    pass


class MissingGeneratedCitationError(
    ValueError
):
    pass


class EmptyGeneratedAnswerError(
    ValueError
):
    pass


@dataclass(frozen=True)
class GroundedAnswerResult:
    correlation_id: UUID

    status: Literal[
        "SUPPORTED",
        "INSUFFICIENT_EVIDENCE",
    ]

    answer: str

    citations: ValidatedCitationSet

    cited_evidence_ids: tuple[
        str,
        ...,
    ]


class GroundedAnswerGenerationService:
    """
    Security boundary between evidence verification
    and an external LLM provider.

    Rules:

    1. Insufficient evidence never reaches the LLM.
    2. Only canonical validated citations may become
       LLM grounding evidence.
    3. Raw retrieval/top-k results never enter the
       provider request.
    4. Generated citation identifiers must refer only
       to evidence supplied to the provider.
    """

    def __init__(
        self,
        *,
        citation_service: (
            ValidatedCitationService
        ),
        provider: LLMProvider,
    ) -> None:
        self._citation_service = (
            citation_service
        )
        self._provider = provider

    def generate(
        self,
        *,
        question: str,
        evidence_result: (
            AtomicEvidenceOrchestrationResult
        ),
        language: str = "fr",
    ) -> GroundedAnswerResult:
        clean_question = question.strip()

        if not clean_question:
            raise ValueError(
                "question cannot be empty."
            )

        clean_language = language.strip()

        if not clean_language:
            raise ValueError(
                "language cannot be empty."
            )

        # --------------------------------------------------
        # Fail closed BEFORE citation construction or LLM.
        # --------------------------------------------------

        if not evidence_result.is_sufficient:
            return GroundedAnswerResult(
                correlation_id=(
                    evidence_result
                    .correlation_id
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

        # --------------------------------------------------
        # Canonical citation validation is mandatory.
        # --------------------------------------------------

        validated = (
            self._citation_service.build(
                evidence_result
            )
        )

        if not validated.citations:
            raise RuntimeError(
                "Sufficient evidence produced no "
                "validated citations."
            )

        # --------------------------------------------------
        # Construct the ONLY evidence allowed to cross
        # the provider boundary.
        #
        # Deliberately built from ValidatedCitationSet,
        # never from claim_results/retrieved_evidence.
        # --------------------------------------------------

        grounding_evidence = tuple(
            LLMGroundingEvidence(
                evidence_id=item.claim_id,
                claim=item.claim,
                chunk_id=(
                    item.citation.chunk_id
                ),
                document_id=(
                    item.citation.document_id
                ),
                document_version_id=(
                    item.citation
                    .document_version_id
                ),
                page_id=(
                    item.citation.page_id
                ),
                page_number=(
                    item.citation.page_number
                ),
                start_char=(
                    item.citation.start_char
                ),
                end_char=(
                    item.citation.end_char
                ),
                cited_text=(
                    item.citation.cited_text
                ),
            )
            for item in validated.citations
        )

        request = LLMGenerationRequest(
            correlation_id=(
                evidence_result.correlation_id
            ),
            question=clean_question,
            language=clean_language,
            evidence=grounding_evidence,
        )

        response = self._provider.generate(
            request
        )

        self._validate_response(
            response=response,
            request=request,
        )

        return GroundedAnswerResult(
            correlation_id=(
                evidence_result.correlation_id
            ),
            status="SUPPORTED",
            answer=response.answer.strip(),
            citations=validated,
            cited_evidence_ids=(
                response.citation_ids
            ),
        )

    @staticmethod
    def _validate_response(
        *,
        response: LLMGenerationResponse,
        request: LLMGenerationRequest,
    ) -> None:
        if not response.answer.strip():
            raise EmptyGeneratedAnswerError(
                "LLM returned an empty answer."
            )

        if not response.citation_ids:
            raise MissingGeneratedCitationError(
                "Grounded answers require at least "
                "one validated citation."
            )

        allowed = {
            item.evidence_id
            for item in request.evidence
        }

        supplied = set(
            response.citation_ids
        )

        unknown = supplied - allowed

        if unknown:
            raise InvalidGeneratedCitationError(
                "LLM returned citation identifiers "
                "outside the validated citation set: "
                + ", ".join(
                    sorted(unknown)
                )
            )
