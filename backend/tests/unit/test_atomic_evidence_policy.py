from __future__ import annotations

from uuid import uuid4

import pytest

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
from app.application.services.sentence_evidence_verification import (
    EvidenceSpan,
    EvidenceVerificationResult,
)
from app.domain.enums.evidence import (
    EntailmentLabel,
    EvidenceDecision,
)


def chunk(
    *,
    content: str,
    index: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=1,
        chunk_index=index,
        content=content,
        section_title=None,
        language="fr",
        score=0.80,
    )


def nli_score(
    *,
    label: EntailmentLabel,
    entailment: float,
) -> EntailmentScore:
    if label == EntailmentLabel.ENTAILMENT:
        neutral = (
            1.0 - entailment
        ) * 0.8

        contradiction = (
            1.0
            - entailment
            - neutral
        )

    elif label == EntailmentLabel.NEUTRAL:
        neutral = 0.98
        contradiction = (
            1.0
            - entailment
            - neutral
        )

    else:
        contradiction = 0.98
        neutral = (
            1.0
            - entailment
            - contradiction
        )

    return EntailmentScore(
        label=label,
        entailment_probability=(
            entailment
        ),
        neutral_probability=neutral,
        contradiction_probability=(
            contradiction
        ),
    )


def verification(
    *,
    claim: str,
    labels: tuple[
        tuple[
            EntailmentLabel,
            float,
        ],
        ...,
    ],
) -> EvidenceVerificationResult:
    spans = []

    for index, (
        label,
        probability,
    ) in enumerate(labels):
        text = f"Evidence {index}."

        source = chunk(
            content=text,
            index=index,
        )

        spans.append(
            EvidenceSpan(
                chunk=source,
                sentence_index=0,
                start_char=0,
                end_char=len(text),
                text=text,
                verification=nli_score(
                    label=label,
                    entailment=probability,
                ),
            )
        )

    return EvidenceVerificationResult(
        claim=claim,
        spans=tuple(spans),
    )


def test_all_required_claims_supported():
    assessment = (
        AtomicEvidencePolicy()
        .assess(
            (
                AtomicClaimEvidence(
                    claim_id="A",
                    verification=verification(
                        claim="Claim A",
                        labels=(
                            (
                                EntailmentLabel
                                .ENTAILMENT,
                                0.99,
                            ),
                        ),
                    ),
                ),
                AtomicClaimEvidence(
                    claim_id="B",
                    verification=verification(
                        claim="Claim B",
                        labels=(
                            (
                                EntailmentLabel
                                .ENTAILMENT,
                                0.95,
                            ),
                        ),
                    ),
                ),
            )
        )
    )

    assert assessment.is_sufficient

    assert (
        assessment.decision
        == EvidenceDecision.SUFFICIENT
    )

    assert [
        item.claim_id
        for item
        in assessment.selected_evidence
    ] == [
        "A",
        "B",
    ]

    assert (
        assessment
        .weakest_required_support_probability
        == pytest.approx(0.95)
    )


def test_weakest_required_claim_fails_closed():
    assessment = (
        AtomicEvidencePolicy()
        .assess(
            (
                AtomicClaimEvidence(
                    claim_id="A",
                    verification=verification(
                        claim="Claim A",
                        labels=(
                            (
                                EntailmentLabel
                                .ENTAILMENT,
                                0.99,
                            ),
                        ),
                    ),
                ),
                AtomicClaimEvidence(
                    claim_id="B",
                    verification=verification(
                        claim="Claim B",
                        labels=(
                            (
                                EntailmentLabel
                                .NEUTRAL,
                                0.01,
                            ),
                        ),
                    ),
                ),
            )
        )
    )

    assert not assessment.is_sufficient

    assert (
        assessment.decision
        == EvidenceDecision
        .INSUFFICIENT_SUPPORT
    )

    assert (
        assessment
        .unsupported_required_claim_ids
        == ("B",)
    )

    # Supported claims remain available
    # for audit, but the overall answer
    # must still be refused.
    assert [
        item.claim_id
        for item
        in assessment.selected_evidence
    ] == ["A"]

    assert (
        assessment
        .weakest_required_support_probability
        is None
    )


def test_best_independent_sentence_is_selected():
    assessment = (
        AtomicEvidencePolicy()
        .assess(
            (
                AtomicClaimEvidence(
                    claim_id="A",
                    verification=verification(
                        claim="Claim A",
                        labels=(
                            (
                                EntailmentLabel
                                .ENTAILMENT,
                                0.91,
                            ),
                            (
                                EntailmentLabel
                                .ENTAILMENT,
                                0.99,
                            ),
                        ),
                    ),
                ),
            )
        )
    )

    selected = (
        assessment
        .selected_evidence[0]
        .span
    )

    assert (
        selected.verification
        .entailment_probability
        == pytest.approx(0.99)
    )

    assert selected.text == "Evidence 1."


def test_optional_claim_does_not_block_answer():
    assessment = (
        AtomicEvidencePolicy()
        .assess(
            (
                AtomicClaimEvidence(
                    claim_id="required",
                    verification=verification(
                        claim="Required",
                        labels=(
                            (
                                EntailmentLabel
                                .ENTAILMENT,
                                0.98,
                            ),
                        ),
                    ),
                    required=True,
                ),
                AtomicClaimEvidence(
                    claim_id="optional",
                    verification=verification(
                        claim="Optional",
                        labels=(
                            (
                                EntailmentLabel
                                .NEUTRAL,
                                0.01,
                            ),
                        ),
                    ),
                    required=False,
                ),
            )
        )
    )

    assert assessment.is_sufficient


def test_no_required_evidence_is_no_evidence():
    assessment = (
        AtomicEvidencePolicy()
        .assess(
            (
                AtomicClaimEvidence(
                    claim_id="A",
                    verification=(
                        EvidenceVerificationResult(
                            claim="Claim A",
                            spans=(),
                        )
                    ),
                ),
                AtomicClaimEvidence(
                    claim_id="B",
                    verification=(
                        EvidenceVerificationResult(
                            claim="Claim B",
                            spans=(),
                        )
                    ),
                ),
            )
        )
    )

    assert (
        assessment.decision
        == EvidenceDecision.NO_EVIDENCE
    )

    assert (
        assessment
        .unsupported_required_claim_ids
        == (
            "A",
            "B",
        )
    )


def test_some_evidence_but_missing_claim_is_insufficient():
    assessment = (
        AtomicEvidencePolicy()
        .assess(
            (
                AtomicClaimEvidence(
                    claim_id="A",
                    verification=verification(
                        claim="Claim A",
                        labels=(
                            (
                                EntailmentLabel
                                .ENTAILMENT,
                                0.99,
                            ),
                        ),
                    ),
                ),
                AtomicClaimEvidence(
                    claim_id="B",
                    verification=(
                        EvidenceVerificationResult(
                            claim="Claim B",
                            spans=(),
                        )
                    ),
                ),
            )
        )
    )

    assert (
        assessment.decision
        == EvidenceDecision
        .INSUFFICIENT_SUPPORT
    )

    assert (
        assessment
        .unsupported_required_claim_ids
        == ("B",)
    )


def test_duplicate_claim_ids_are_rejected():
    item = AtomicClaimEvidence(
        claim_id="same",
        verification=(
            EvidenceVerificationResult(
                claim="Claim",
                spans=(),
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        AtomicEvidencePolicy().assess(
            (
                item,
                item,
            )
        )


def test_empty_claim_id_is_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        AtomicClaimEvidence(
            claim_id="   ",
            verification=(
                EvidenceVerificationResult(
                    claim="Claim",
                    spans=(),
                )
            ),
        )


def test_at_least_one_required_claim_is_mandatory():
    with pytest.raises(
        ValueError,
        match="At least one required",
    ):
        AtomicEvidencePolicy().assess(
            (
                AtomicClaimEvidence(
                    claim_id="optional",
                    verification=(
                        EvidenceVerificationResult(
                            claim="Optional",
                            spans=(),
                        )
                    ),
                    required=False,
                ),
            )
        )


def test_empty_claim_collection_is_no_evidence():
    assessment = (
        AtomicEvidencePolicy()
        .assess(())
    )

    assert (
        assessment.decision
        == EvidenceDecision.NO_EVIDENCE
    )

    assert not assessment.is_sufficient
