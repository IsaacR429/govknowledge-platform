from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from app.application.ports.citation_source_repository import (
    CitationSource,
)
from app.application.ports.entailment_verifier import (
    EntailmentScore,
)
from app.application.ports.retrieval import (
    RetrievedChunk,
)
from app.application.services.atomic_evidence_orchestration import (
    AtomicClaim,
    AtomicClaimResult,
    AtomicEvidenceOrchestrationResult,
)
from app.application.services.atomic_evidence_policy import (
    AtomicClaimEvidence,
    AtomicEvidencePolicy,
)
from app.application.services.citation_mapping import (
    ClaimOriginCitationMapper,
    EvidenceSpanCitationMapper,
)
from app.application.services.citation_validation import (
    CitationValidator,
    InvalidCitationError,
)
from app.application.services.claim_provenance import ClaimOrigin
from app.application.services.sentence_evidence_verification import (
    EvidenceSpan,
    EvidenceVerificationResult,
)
from app.application.services.validated_citations import (
    InsufficientEvidenceForCitationsError,
    ValidatedCitationService,
)
from app.domain.enums.evidence import (
    EntailmentLabel,
)


class FakeCitationSourceRepository:
    def __init__(
        self,
        sources,
    ):
        self.sources = {
            source.chunk_id: source
            for source in sources
        }

        self.calls = []

    def get_by_chunk_id(
        self,
        chunk_id,
    ):
        self.calls.append(
            chunk_id
        )

        return self.sources.get(
            chunk_id
        )


def retrieved_chunk():
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=8,
        chunk_index=3,
        content=(
            "First sentence. "
            "Canonical supporting sentence."
        ),
        section_title="Article 4",
        language="fr",
        score=0.91,
    )


def evidence_span():
    chunk = retrieved_chunk()

    text = (
        "Canonical supporting sentence."
    )

    start = chunk.content.index(
        text
    )

    return EvidenceSpan(
        chunk=chunk,
        sentence_index=1,
        start_char=start,
        end_char=start + len(text),
        text=text,
        verification=EntailmentScore(
            label=(
                EntailmentLabel.ENTAILMENT
            ),
            entailment_probability=0.99,
            neutral_probability=0.009,
            contradiction_probability=0.001,
        ),
    )


def validator_for(
    citation,
    content="First sentence. Canonical supporting sentence.",
):
    source = CitationSource(
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        document_version_id=(
            citation.document_version_id
        ),
        page_id=citation.page_id,
        page_number=citation.page_number,
        content=content,
    )

    return CitationValidator(
        repository=(
            FakeCitationSourceRepository(
                (source,)
            )
        )
    )


def test_maps_exact_evidence_span_to_citation():
    span = evidence_span()

    citation = (
        EvidenceSpanCitationMapper()
        .map(span)
    )

    assert (
        citation.chunk_id
        == span.chunk.chunk_id
    )

    assert (
        citation.document_id
        == span.chunk.document_id
    )

    assert (
        citation.document_version_id
        == span.chunk.document_version_id
    )

    assert (
        citation.page_id
        == span.chunk.page_id
    )

    assert (
        citation.page_number
        == span.chunk.page_number
    )

    assert (
        citation.cited_text
        == span.text
    )

    assert (
        span.chunk.content[
            citation.start_char:
            citation.end_char
        ]
        == citation.cited_text
    )


def test_maps_claim_origin_to_citation():
    chunk_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    page_id = uuid4()

    text = "Les informations doivent être transmises."
    origin = ClaimOrigin(
        evidence_id="evidence-1",
        chunk_id=chunk_id,
        document_id=document_id,
        document_version_id=version_id,
        page_id=page_id,
        page_number=5,
        start_char=0,
        end_char=len(text),
        text=text,
        language="fr",
    )

    citation = ClaimOriginCitationMapper().map(origin)

    assert citation.chunk_id == chunk_id
    assert citation.document_id == document_id
    assert citation.document_version_id == version_id
    assert citation.page_id == page_id
    assert citation.page_number == 5
    assert citation.start_char == 0
    assert citation.end_char == len(text)
    assert citation.cited_text == text


def test_valid_citation_passes():
    citation = (
        EvidenceSpanCitationMapper()
        .map(
            evidence_span()
        )
    )

    result = (
        validator_for(
            citation
        )
        .validate(citation)
    )

    assert result == citation


@pytest.mark.parametrize(
    "field",
    (
        "chunk_id",
        "document_id",
        "document_version_id",
        "page_id",
        "page_number",
    ),
)
def test_tampered_lineage_is_rejected(
    field,
):
    citation = (
        EvidenceSpanCitationMapper()
        .map(
            evidence_span()
        )
    )

    if field == "page_number":
        tampered = replace(
            citation,
            page_number=999,
        )
    else:
        tampered = replace(
            citation,
            **{
                field: uuid4()
            },
        )

    with pytest.raises(
        InvalidCitationError
    ):
        validator_for(
            citation
        ).validate(
            tampered
        )


def test_tampered_text_is_rejected():
    citation = (
        EvidenceSpanCitationMapper()
        .map(
            evidence_span()
        )
    )

    tampered = replace(
        citation,
        cited_text=(
            "Invented policy statement."
        ),
    )

    with pytest.raises(
        InvalidCitationError
    ):
        validator_for(
            citation
        ).validate(
            tampered
        )


def test_tampered_offsets_are_rejected():
    citation = (
        EvidenceSpanCitationMapper()
        .map(
            evidence_span()
        )
    )

    tampered = replace(
        citation,
        start_char=0,
    )

    with pytest.raises(
        InvalidCitationError
    ):
        validator_for(
            citation
        ).validate(
            tampered
        )


def test_missing_canonical_chunk_is_rejected():
    citation = (
        EvidenceSpanCitationMapper()
        .map(
            evidence_span()
        )
    )

    validator = CitationValidator(
        repository=(
            FakeCitationSourceRepository([])
        )
    )

    with pytest.raises(
        InvalidCitationError
    ):
        validator.validate(
            citation
        )


def test_mapper_rejects_noncanonical_span_text():
    span = evidence_span()

    tampered_span = replace(
        span,
        text="Different text.",
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        (
            EvidenceSpanCitationMapper()
            .map(tampered_span)
        )


def supporting_claim_fixture(
    *,
    claim_id: str,
    claim_text: str,
    page_number: int,
):
    chunk_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    page_id = uuid4()

    origin = ClaimOrigin(
        evidence_id="evidence-1",
        chunk_id=chunk_id,
        document_id=document_id,
        document_version_id=version_id,
        page_id=page_id,
        page_number=page_number,
        start_char=0,
        end_char=len(claim_text),
        text=claim_text,
        language="fr",
    )

    claim = AtomicClaim(
        claim_id=claim_id,
        text=claim_text,
        origin=origin,
    )

    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version_id=version_id,
        page_id=page_id,
        page_number=page_number,
        chunk_index=page_number,
        content=claim_text,
        section_title=None,
        language="fr",
        score=0.90,
    )

    span = EvidenceSpan(
        chunk=chunk,
        sentence_index=0,
        start_char=0,
        end_char=len(claim_text),
        text=claim_text,
        verification=EntailmentScore(
            label=EntailmentLabel.ENTAILMENT,
            entailment_probability=0.99,
            neutral_probability=0.009,
            contradiction_probability=0.001,
        ),
    )

    verification = EvidenceVerificationResult(
        claim=claim_text,
        spans=(span,),
    )

    source = CitationSource(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version_id=version_id,
        page_id=page_id,
        page_number=page_number,
        content=claim_text,
    )

    return claim, verification, source


def service(sources):
    repository = FakeCitationSourceRepository(sources)
    return (
        ValidatedCitationService(
            mapper=ClaimOriginCitationMapper(),
            validator=CitationValidator(repository=repository),
        ),
        repository,
    )


def test_builds_validated_citation_for_each_claim():
    claim_a, verification_a, source_a = supporting_claim_fixture(
        claim_id="A",
        claim_text="Evidence A.",
        page_number=3,
    )

    claim_b, verification_b, source_b = supporting_claim_fixture(
        claim_id="B",
        claim_text="Evidence B.",
        page_number=7,
    )

    assessment = AtomicEvidencePolicy().assess(
        (
            AtomicClaimEvidence(
                claim_id="A",
                verification=verification_a,
            ),
            AtomicClaimEvidence(
                claim_id="B",
                verification=verification_b,
            ),
        )
    )

    evidence_result = AtomicEvidenceOrchestrationResult(
        correlation_id=uuid4(),
        claim_results=(
            AtomicClaimResult(
                claim=claim_a,
                retrieved_evidence=(),
                verification=verification_a,
            ),
            AtomicClaimResult(
                claim=claim_b,
                retrieved_evidence=(),
                verification=verification_b,
            ),
        ),
        assessment=assessment,
    )

    citation_service, repository = service((source_a, source_b))

    result = citation_service.build(evidence_result)

    assert [item.claim_id for item in result.citations] == ["A", "B"]
    assert [item.claim for item in result.citations] == ["Evidence A.", "Evidence B."]
    assert [item.citation.cited_text for item in result.citations] == [
        "Evidence A.",
        "Evidence B.",
    ]
    assert repository.calls == [source_a.chunk_id, source_b.chunk_id]


def test_insufficient_assessment_cannot_build_answer_citations():
    claim_a, verification_a, source_a = supporting_claim_fixture(
        claim_id="A",
        claim_text="Evidence A.",
        page_number=3,
    )

    claim_b = AtomicClaim(
        claim_id="B",
        text="Unsupported claim B.",
        origin=ClaimOrigin(
            evidence_id="evidence-2",
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_version_id=uuid4(),
            page_id=uuid4(),
            page_number=1,
            start_char=0,
            end_char=len("Unsupported claim B."),
            text="Unsupported claim B.",
            language="fr",
        ),
    )

    unsupported = EvidenceVerificationResult(
        claim="Unsupported claim B.",
        spans=(),
    )

    assessment = AtomicEvidencePolicy().assess(
        (
            AtomicClaimEvidence(
                claim_id="A",
                verification=verification_a,
            ),
            AtomicClaimEvidence(
                claim_id="B",
                verification=unsupported,
            ),
        )
    )

    evidence_result = AtomicEvidenceOrchestrationResult(
        correlation_id=uuid4(),
        claim_results=(
            AtomicClaimResult(
                claim=claim_a,
                retrieved_evidence=(),
                verification=verification_a,
            ),
            AtomicClaimResult(
                claim=claim_b,
                retrieved_evidence=(),
                verification=unsupported,
            ),
        ),
        assessment=assessment,
    )

    citation_service, repository = service((source_a,))

    with pytest.raises(InsufficientEvidenceForCitationsError):
        citation_service.build(evidence_result)

    assert repository.calls == []


def test_canonical_validation_failure_aborts_set():
    claim_a, verification_a, source_a = supporting_claim_fixture(
        claim_id="A",
        claim_text="Evidence A.",
        page_number=3,
    )

    assessment = AtomicEvidencePolicy().assess(
        (
            AtomicClaimEvidence(
                claim_id="A",
                verification=verification_a,
            ),
        )
    )

    evidence_result = AtomicEvidenceOrchestrationResult(
        correlation_id=uuid4(),
        claim_results=(
            AtomicClaimResult(
                claim=claim_a,
                retrieved_evidence=(),
                verification=verification_a,
            ),
        ),
        assessment=assessment,
    )

    tampered_source = CitationSource(
        chunk_id=source_a.chunk_id,
        document_id=uuid4(),
        document_version_id=source_a.document_version_id,
        page_id=source_a.page_id,
        page_number=source_a.page_number,
        content=source_a.content,
    )

    citation_service, _ = service((tampered_source,))

    with pytest.raises(InvalidCitationError):
        citation_service.build(evidence_result)


def test_citation_points_to_claim_origin_not_nli_verification_span():
    """
    Explicitly reproduces G28's failure class:
    origin text: "Les informations ... dans les 24 heures si possible."
    verification winner: "Toute trace écrite doit être conservée en toute sécurité..."

    The assessment should still be SUFFICIENT, because NLI verification is a separate concern.
    But the resulting citation must be:
    "Les informations ... dans les 24 heures si possible."
    and never the storage sentence.
    """
    origin_text = "Les informations doivent être transmises dans les 24 heures si possible."
    storage_text = "Toute trace écrite doit être conservée en toute sécurité."
    chunk_content = f"{origin_text} {storage_text}"

    chunk_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    page_id = uuid4()

    origin = ClaimOrigin(
        evidence_id="evidence-1",
        chunk_id=chunk_id,
        document_id=document_id,
        document_version_id=version_id,
        page_id=page_id,
        page_number=1,
        start_char=0,
        end_char=len(origin_text),
        text=origin_text,
        language="fr",
    )

    claim = AtomicClaim(
        claim_id="claim-g28",
        text=origin_text,
        origin=origin,
    )

    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version_id=version_id,
        page_id=page_id,
        page_number=1,
        chunk_index=1,
        content=chunk_content,
        section_title=None,
        language="fr",
        score=0.95,
    )

    # NLI verification picked the storage sentence span
    storage_start = chunk_content.index(storage_text)
    storage_span = EvidenceSpan(
        chunk=chunk,
        sentence_index=1,
        start_char=storage_start,
        end_char=storage_start + len(storage_text),
        text=storage_text,
        verification=EntailmentScore(
            label=EntailmentLabel.ENTAILMENT,
            entailment_probability=0.98,
            neutral_probability=0.01,
            contradiction_probability=0.01,
        ),
    )

    verification = EvidenceVerificationResult(
        claim=origin_text,
        spans=(storage_span,),
    )

    assessment = AtomicEvidencePolicy().assess(
        (
            AtomicClaimEvidence(
                claim_id="claim-g28",
                verification=verification,
            ),
        )
    )

    # Assessment is SUFFICIENT because NLI entailment score was high on storage_span
    assert assessment.is_sufficient

    evidence_result = AtomicEvidenceOrchestrationResult(
        correlation_id=uuid4(),
        claim_results=(
            AtomicClaimResult(
                claim=claim,
                retrieved_evidence=(),
                verification=verification,
            ),
        ),
        assessment=assessment,
    )

    source = CitationSource(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version_id=version_id,
        page_id=page_id,
        page_number=1,
        content=chunk_content,
    )

    citation_service, _ = service((source,))
    result = citation_service.build(evidence_result)

    assert len(result.citations) == 1
    item = result.citations[0]
    assert item.claim_id == "claim-g28"
    assert item.claim == origin_text

    # The resulting citation must be origin_text and NOT storage_text
    assert item.citation.cited_text == origin_text
    assert item.citation.cited_text != storage_text
    assert item.citation.start_char == 0
    assert item.citation.end_char == len(origin_text)
