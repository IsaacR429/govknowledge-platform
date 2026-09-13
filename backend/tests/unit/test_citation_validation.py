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
from app.application.services.citation_mapping import (
    EvidenceSpanCitationMapper,
)
from app.application.services.citation_validation import (
    CitationValidator,
    InvalidCitationError,
)
from app.application.services.sentence_evidence_verification import (
    EvidenceSpan,
)
from app.domain.enums.evidence import (
    EntailmentLabel,
)


class FakeCitationSourceRepository:
    def __init__(
        self,
        source=None,
    ):
        self.source = source
        self.calls = []

    def get_by_chunk_id(
        self,
        chunk_id,
    ):
        self.calls.append(
            chunk_id
        )

        if (
            self.source is not None
            and self.source.chunk_id
            == chunk_id
        ):
            return self.source

        return None


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
):
    source = CitationSource(
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        document_version_id=(
            citation.document_version_id
        ),
        page_id=citation.page_id,
        page_number=citation.page_number,
        content=(
            "First sentence. "
            "Canonical supporting sentence."
        ),
    )

    return CitationValidator(
        repository=(
            FakeCitationSourceRepository(
                source
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
            FakeCitationSourceRepository()
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
