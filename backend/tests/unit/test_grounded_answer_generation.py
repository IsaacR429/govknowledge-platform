from __future__ import annotations

from uuid import uuid4

import pytest

from app.application.ports.llm_provider import (
    LLMGenerationResponse,
)
from app.application.services.atomic_evidence_orchestration import (
    AtomicEvidenceOrchestrationResult,
)
from app.application.services.atomic_evidence_policy import (
    AtomicEvidenceAssessment,
)
from app.application.services.citation_mapping import (
    Citation,
)
from app.application.services.grounded_answer_generation import (
    EmptyGeneratedAnswerError,
    FRENCH_INSUFFICIENT_EVIDENCE_REFUSAL,
    GroundedAnswerGenerationService,
    InvalidGeneratedCitationError,
    MissingGeneratedCitationError,
)
from app.application.services.validated_citations import (
    ClaimCitation,
    ValidatedCitationSet,
)
from app.domain.enums.evidence import (
    EvidenceDecision,
)


class FakeProvider:
    def __init__(
        self,
        response,
    ):
        self.response = response
        self.calls = []

    def generate(
        self,
        request,
    ):
        self.calls.append(request)

        return self.response


class FakeCitationService:
    def __init__(
        self,
        validated,
    ):
        self.validated = validated
        self.calls = []

    def build(
        self,
        assessment,
    ):
        self.calls.append(
            assessment
        )

        return self.validated


def sufficient_result():
    return (
        AtomicEvidenceOrchestrationResult(
            correlation_id=uuid4(),
            claim_results=(),
            assessment=(
                AtomicEvidenceAssessment(
                    decision=(
                        EvidenceDecision.SUFFICIENT
                    ),
                    selected_evidence=(),
                    unsupported_required_claim_ids=(),
                    weakest_required_support_probability=(
                        0.99
                    ),
                )
            ),
        )
    )


def insufficient_result():
    return (
        AtomicEvidenceOrchestrationResult(
            correlation_id=uuid4(),
            claim_results=(),
            assessment=(
                AtomicEvidenceAssessment(
                    decision=(
                        EvidenceDecision
                        .INSUFFICIENT_SUPPORT
                    ),
                    selected_evidence=(),
                    unsupported_required_claim_ids=(
                        "claim-a",
                    ),
                    weakest_required_support_probability=None,
                )
            ),
        )
    )


def validated_citations():
    citation = Citation(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=19,
        start_char=100,
        end_char=140,
        cited_text=(
            "Validated evidence only."
        ),
    )

    return ValidatedCitationSet(
        citations=(
            ClaimCitation(
                claim_id="claim-a",
                claim=(
                    "The validated claim."
                ),
                citation=citation,
            ),
        )
    )


def test_insufficient_evidence_never_calls_provider():
    provider = FakeProvider(
        LLMGenerationResponse(
            answer="Must not be used.",
            citation_ids=("claim-a",),
        )
    )

    citation_service = (
        FakeCitationService(
            validated_citations()
        )
    )

    result = (
        GroundedAnswerGenerationService(
            citation_service=(
                citation_service
            ),
            provider=provider,
        ).generate(
            question=(
                "What does the policy say?"
            ),
            evidence_result=(
                insufficient_result()
            ),
        )
    )

    assert (
        result.status
        == "INSUFFICIENT_EVIDENCE"
    )

    assert (
        result.answer
        == FRENCH_INSUFFICIENT_EVIDENCE_REFUSAL
    )

    assert result.citations.citations == ()

    assert (
        result.cited_evidence_ids
        == ()
    )

    # Critical security invariant.
    assert provider.calls == []

    # We do not even attempt answer citations.
    assert citation_service.calls == []


def test_sufficient_evidence_sends_only_validated_citations():
    validated = validated_citations()

    provider = FakeProvider(
        LLMGenerationResponse(
            answer=(
                "La source confirme la règle."
            ),
            citation_ids=("claim-a",),
        )
    )

    citation_service = (
        FakeCitationService(
            validated
        )
    )

    evidence_result = (
        sufficient_result()
    )

    result = (
        GroundedAnswerGenerationService(
            citation_service=(
                citation_service
            ),
            provider=provider,
        ).generate(
            question=(
                "Quelle est la règle ?"
            ),
            evidence_result=(
                evidence_result
            ),
            language="fr",
        )
    )

    assert result.status == "SUPPORTED"

    assert len(provider.calls) == 1

    request = provider.calls[0]

    assert (
        request.correlation_id
        == evidence_result.correlation_id
    )

    assert (
        request.question
        == "Quelle est la règle ?"
    )

    assert request.language == "fr"

    assert len(request.evidence) == 1

    evidence = request.evidence[0]

    assert (
        evidence.evidence_id
        == "claim-a"
    )

    assert (
        evidence.claim
        == "The validated claim."
    )

    assert (
        evidence.cited_text
        == "Validated evidence only."
    )

    assert evidence.page_number == 19

    # Provider evidence comes from the exact
    # canonical validated citation.
    assert (
        evidence.chunk_id
        == validated.citations[
            0
        ].citation.chunk_id
    )

    assert result.citations == validated

    assert (
        result.cited_evidence_ids
        == ("claim-a",)
    )


def test_generated_unknown_citation_is_rejected():
    provider = FakeProvider(
        LLMGenerationResponse(
            answer="Generated answer.",
            citation_ids=(
                "fabricated-citation",
            ),
        )
    )

    service = (
        GroundedAnswerGenerationService(
            citation_service=(
                FakeCitationService(
                    validated_citations()
                )
            ),
            provider=provider,
        )
    )

    with pytest.raises(
        InvalidGeneratedCitationError
    ):
        service.generate(
            question="Question",
            evidence_result=(
                sufficient_result()
            ),
        )


def test_generated_answer_requires_citation():
    provider = FakeProvider(
        LLMGenerationResponse(
            answer="Generated answer.",
            citation_ids=(),
        )
    )

    service = (
        GroundedAnswerGenerationService(
            citation_service=(
                FakeCitationService(
                    validated_citations()
                )
            ),
            provider=provider,
        )
    )

    with pytest.raises(
        MissingGeneratedCitationError
    ):
        service.generate(
            question="Question",
            evidence_result=(
                sufficient_result()
            ),
        )


def test_empty_generated_answer_is_rejected():
    provider = FakeProvider(
        LLMGenerationResponse(
            answer="   ",
            citation_ids=("claim-a",),
        )
    )

    service = (
        GroundedAnswerGenerationService(
            citation_service=(
                FakeCitationService(
                    validated_citations()
                )
            ),
            provider=provider,
        )
    )

    with pytest.raises(
        EmptyGeneratedAnswerError
    ):
        service.generate(
            question="Question",
            evidence_result=(
                sufficient_result()
            ),
        )
