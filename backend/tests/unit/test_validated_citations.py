from __future__ import annotations

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
from app.application.services.atomic_evidence_policy import (
    AtomicClaimEvidence,
    AtomicEvidencePolicy,
)
from app.application.services.citation_mapping import (
    EvidenceSpanCitationMapper,
)
from app.application.services.citation_validation import (
    CitationValidator,
    InvalidCitationError,
)
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


def supporting_verification(
    *,
    claim,
    text,
    page_number,
):
    chunk = RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=page_number,
        chunk_index=page_number,
        content=text,
        section_title=None,
        language="fr",
        score=0.90,
    )

    span = EvidenceSpan(
        chunk=chunk,
        sentence_index=0,
        start_char=0,
        end_char=len(text),
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

    return (
        EvidenceVerificationResult(
            claim=claim,
            spans=(span,),
        ),
        CitationSource(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_version_id=(
                chunk.document_version_id
            ),
            page_id=chunk.page_id,
            page_number=chunk.page_number,
            content=chunk.content,
        ),
    )


def service(
    sources,
):
    repository = (
        FakeCitationSourceRepository(
            sources
        )
    )

    return (
        ValidatedCitationService(
            mapper=(
                EvidenceSpanCitationMapper()
            ),
            validator=(
                CitationValidator(
                    repository=repository
                )
            ),
        ),
        repository,
    )


def test_builds_validated_citation_for_each_claim():
    verification_a, source_a = (
        supporting_verification(
            claim="Claim A",
            text="Evidence A.",
            page_number=3,
        )
    )

    verification_b, source_b = (
        supporting_verification(
            claim="Claim B",
            text="Evidence B.",
            page_number=7,
        )
    )

    assessment = (
        AtomicEvidencePolicy()
        .assess(
            (
                AtomicClaimEvidence(
                    claim_id="A",
                    verification=(
                        verification_a
                    ),
                ),
                AtomicClaimEvidence(
                    claim_id="B",
                    verification=(
                        verification_b
                    ),
                ),
            )
        )
    )

    citation_service, repository = (
        service(
            (
                source_a,
                source_b,
            )
        )
    )

    result = citation_service.build(
        assessment
    )

    assert [
        item.claim_id
        for item in result.citations
    ] == [
        "A",
        "B",
    ]

    assert [
        item.claim
        for item in result.citations
    ] == [
        "Claim A",
        "Claim B",
    ]

    assert [
        item.citation.cited_text
        for item in result.citations
    ] == [
        "Evidence A.",
        "Evidence B.",
    ]

    assert repository.calls == [
        source_a.chunk_id,
        source_b.chunk_id,
    ]


def test_insufficient_assessment_cannot_build_answer_citations():
    verification, source = (
        supporting_verification(
            claim="Claim A",
            text="Evidence A.",
            page_number=3,
        )
    )

    unsupported = (
        EvidenceVerificationResult(
            claim="Claim B",
            spans=(),
        )
    )

    assessment = (
        AtomicEvidencePolicy()
        .assess(
            (
                AtomicClaimEvidence(
                    claim_id="A",
                    verification=(
                        verification
                    ),
                ),
                AtomicClaimEvidence(
                    claim_id="B",
                    verification=(
                        unsupported
                    ),
                ),
            )
        )
    )

    citation_service, repository = (
        service((source,))
    )

    with pytest.raises(
        InsufficientEvidenceForCitationsError
    ):
        citation_service.build(
            assessment
        )

    assert repository.calls == []


def test_canonical_validation_failure_aborts_set():
    verification, source = (
        supporting_verification(
            claim="Claim A",
            text="Evidence A.",
            page_number=3,
        )
    )

    assessment = (
        AtomicEvidencePolicy()
        .assess(
            (
                AtomicClaimEvidence(
                    claim_id="A",
                    verification=(
                        verification
                    ),
                ),
            )
        )
    )

    tampered_source = CitationSource(
        chunk_id=source.chunk_id,
        document_id=uuid4(),
        document_version_id=(
            source.document_version_id
        ),
        page_id=source.page_id,
        page_number=source.page_number,
        content=source.content,
    )

    citation_service, _ = (
        service(
            (
                tampered_source,
            )
        )
    )

    with pytest.raises(
        InvalidCitationError
    ) as exc_info:
        citation_service.build(
            assessment
        )

    print(
        "\n===== CITATION LINEAGE TAMPERING ====="
    )

    print(
        "Original document_id:",
        source.document_id,
    )

    print(
        "Tampered document_id:",
        tampered_source.document_id,
    )

    print(
        "Chunk ID preserved:",
        tampered_source.chunk_id
        == source.chunk_id,
    )

    print(
        "Version ID preserved:",
        (
            tampered_source.document_version_id
            == source.document_version_id
        ),
    )

    print(
        "Page ID preserved:",
        (
            tampered_source.page_id
            == source.page_id
        ),
    )

    print(
        "Document ID changed:",
        (
            tampered_source.document_id
            != source.document_id
        ),
    )

    print(
        "Validator rejection:",
        type(exc_info.value).__name__,
    )

    print(
        "Citation lineage tampering blocked: PASS"
    )

    assert (
        tampered_source.document_id
        != source.document_id
    )
