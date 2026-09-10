from uuid import uuid4

import pytest

from app.application.ports.retrieval import (
    RetrievedChunk,
)
from app.application.services.evidence_policy import (
    EvidencePolicyConfig,
    ThresholdEvidencePolicy,
)
from app.domain.enums.evidence import (
    EvidenceDecision,
)


def chunk(
    *,
    score: float,
    index: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=1,
        chunk_index=index,
        content=f"Evidence {index}",
        section_title=None,
        language="en",
        score=score,
    )


def policy(
    *,
    min_top_score: float = 0.80,
    min_supporting_score: float = 0.75,
    min_supporting_chunks: int = 2,
    max_evidence_chunks: int = 3,
):
    return ThresholdEvidencePolicy(
        EvidencePolicyConfig(
            min_top_score=min_top_score,
            min_supporting_score=(
                min_supporting_score
            ),
            min_supporting_chunks=(
                min_supporting_chunks
            ),
            max_evidence_chunks=(
                max_evidence_chunks
            ),
        )
    )


def test_no_results_is_insufficient():
    assessment = policy().assess(())

    assert not assessment.is_sufficient

    assert (
        assessment.decision
        == EvidenceDecision.NO_EVIDENCE
    )

    assert assessment.evidence == ()
    assert assessment.top_score is None


def test_low_top_score_is_rejected():
    assessment = policy().assess(
        (
            chunk(
                score=0.79,
                index=0,
            ),
            chunk(
                score=0.78,
                index=1,
            ),
        )
    )

    assert not assessment.is_sufficient

    assert (
        assessment.decision
        == EvidenceDecision.TOP_SCORE_TOO_LOW
    )

    assert assessment.evidence == ()


def test_strong_top_result_without_support_is_rejected():
    assessment = policy().assess(
        (
            chunk(
                score=0.91,
                index=0,
            ),
            chunk(
                score=0.70,
                index=1,
            ),
        )
    )

    assert not assessment.is_sufficient

    assert (
        assessment.decision
        == EvidenceDecision.INSUFFICIENT_SUPPORT
    )

    assert assessment.supporting_count == 1


def test_sufficient_evidence_is_ranked_and_bounded():
    assessment = policy(
        min_supporting_chunks=2,
        max_evidence_chunks=2,
    ).assess(
        (
            chunk(
                score=0.81,
                index=0,
            ),
            chunk(
                score=0.92,
                index=1,
            ),
            chunk(
                score=0.86,
                index=2,
            ),
        )
    )

    assert assessment.is_sufficient

    assert (
        assessment.decision
        == EvidenceDecision.SUFFICIENT
    )

    assert len(
        assessment.evidence
    ) == 2

    assert [
        item.score
        for item in assessment.evidence
    ] == [
        0.92,
        0.86,
    ]

    assert assessment.supporting_count == 3


def test_non_finite_score_fails_closed():
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        policy().assess(
            (
                chunk(
                    score=float("nan")
                ),
            )
        )


def test_invalid_policy_configuration_is_rejected():
    with pytest.raises(
        ValueError
    ):
        EvidencePolicyConfig(
            min_top_score=0.80,
            min_supporting_score=0.90,
            min_supporting_chunks=1,
            max_evidence_chunks=3,
        )
