from __future__ import annotations

from uuid import uuid4

import pytest

from app.application.ports.entailment_verifier import (
    EntailmentScore,
)
from app.application.ports.retrieval import (
    RetrievedChunk,
)
from app.application.services.sentence_evidence_verification import (
    SentenceEvidenceVerificationService,
)
from app.domain.enums.evidence import (
    EntailmentLabel,
)


def score(
    label: EntailmentLabel,
) -> EntailmentScore:
    if (
        label
        == EntailmentLabel.ENTAILMENT
    ):
        values = (
            0.99,
            0.009,
            0.001,
        )

    elif (
        label
        == EntailmentLabel.NEUTRAL
    ):
        values = (
            0.01,
            0.98,
            0.01,
        )

    else:
        values = (
            0.001,
            0.009,
            0.99,
        )

    return EntailmentScore(
        label=label,
        entailment_probability=(
            values[0]
        ),
        neutral_probability=(
            values[1]
        ),
        contradiction_probability=(
            values[2]
        ),
    )


class FakeVerifier:
    model_name = "fake-nli"
    model_revision = "test"

    def __init__(
        self,
        labels,
    ):
        self._labels = labels
        self.calls = []

    def score(
        self,
        *,
        claim,
        premises,
    ):
        self.calls.append(
            (
                claim,
                tuple(premises),
            )
        )

        return tuple(
            score(label)
            for label
            in self._labels
        )


def chunk(
    content: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=7,
        chunk_index=2,
        content=content,
        section_title="Test",
        language="fr",
        score=0.84,
    )


def test_verifies_sentences_in_one_batch():
    verifier = FakeVerifier(
        (
            EntailmentLabel.NEUTRAL,
            EntailmentLabel.ENTAILMENT,
        )
    )

    evidence = chunk(
        "Premiere phrase. "
        "Deuxieme phrase."
    )

    result = (
        SentenceEvidenceVerificationService(
            verifier=verifier
        )
        .verify(
            claim="Une affirmation",
            evidence=(evidence,),
        )
    )

    assert len(
        verifier.calls
    ) == 1

    assert verifier.calls[0][1] == (
        "Premiere phrase.",
        "Deuxieme phrase.",
    )

    assert result.is_supported

    assert len(
        result.supporting_spans
    ) == 1

    assert (
        result.supporting_spans[0]
        .sentence_index
        == 1
    )


def test_preserves_exact_source_offsets():
    verifier = FakeVerifier(
        (
            EntailmentLabel.ENTAILMENT,
            EntailmentLabel.NEUTRAL,
        )
    )

    evidence = chunk(
        "  First sentence. "
        "Second sentence.  "
    )

    result = (
        SentenceEvidenceVerificationService(
            verifier=verifier
        )
        .verify(
            claim="Claim",
            evidence=(evidence,),
        )
    )

    for span in result.spans:
        assert (
            evidence.content[
                span.start_char:
                span.end_char
            ]
            == span.text
        )

        assert (
            span.chunk.chunk_id
            == evidence.chunk_id
        )

        assert (
            span.chunk.page_id
            == evidence.page_id
        )


def test_only_entailment_supports_claim():
    verifier = FakeVerifier(
        (
            EntailmentLabel.NEUTRAL,
            EntailmentLabel.CONTRADICTION,
        )
    )

    result = (
        SentenceEvidenceVerificationService(
            verifier=verifier
        )
        .verify(
            claim="Claim",
            evidence=(
                chunk(
                    "One sentence. "
                    "Another sentence."
                ),
            ),
        )
    )

    assert not result.is_supported
    assert (
        result.supporting_spans
        == ()
    )


def test_no_evidence_skips_verifier():
    verifier = FakeVerifier(())

    result = (
        SentenceEvidenceVerificationService(
            verifier=verifier
        )
        .verify(
            claim="Claim",
            evidence=(),
        )
    )

    assert not result.is_supported
    assert result.spans == ()
    assert verifier.calls == []


def test_empty_claim_is_rejected():
    verifier = FakeVerifier(())

    with pytest.raises(
        ValueError,
        match="Claim cannot be empty",
    ):
        (
            SentenceEvidenceVerificationService(
                verifier=verifier
            )
            .verify(
                claim="   ",
                evidence=(),
            )
        )


def test_score_count_mismatch_fails_closed():
    verifier = FakeVerifier(
        (
            EntailmentLabel.ENTAILMENT,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected number",
    ):
        (
            SentenceEvidenceVerificationService(
                verifier=verifier
            )
            .verify(
                claim="Claim",
                evidence=(
                    chunk(
                        "One. Two."
                    ),
                ),
            )
        )
