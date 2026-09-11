from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.application.ports.entailment_verifier import (
    EntailmentScore,
    EntailmentVerifierPort,
)
from app.application.ports.retrieval import (
    RetrievedChunk,
)
from app.domain.enums.evidence import (
    EntailmentLabel,
)


_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+"
)


@dataclass(frozen=True)
class EvidenceSpan:
    chunk: RetrievedChunk
    sentence_index: int
    start_char: int
    end_char: int
    text: str
    verification: EntailmentScore

    @property
    def is_supporting(self) -> bool:
        return (
            self.verification.label
            == EntailmentLabel.ENTAILMENT
        )


@dataclass(frozen=True)
class EvidenceVerificationResult:
    claim: str
    spans: tuple[EvidenceSpan, ...]

    @property
    def supporting_spans(
        self,
    ) -> tuple[EvidenceSpan, ...]:
        return tuple(
            span
            for span in self.spans
            if span.is_supporting
        )

    @property
    def is_supported(self) -> bool:
        return bool(
            self.supporting_spans
        )


@dataclass(frozen=True)
class _SentenceCandidate:
    chunk: RetrievedChunk
    sentence_index: int
    start_char: int
    end_char: int
    text: str


def _sentence_candidates(
    chunk: RetrievedChunk,
) -> tuple[_SentenceCandidate, ...]:
    content = chunk.content

    if not content.strip():
        return ()

    stripped = content.strip()

    base_offset = content.find(
        stripped
    )

    pieces = [
        piece.strip()
        for piece in _SENTENCE_BOUNDARY.split(
            stripped
        )
        if piece.strip()
    ]

    candidates: list[
        _SentenceCandidate
    ] = []

    cursor = base_offset

    for sentence_index, text in enumerate(
        pieces
    ):
        start_char = content.find(
            text,
            cursor,
        )

        if start_char < 0:
            raise RuntimeError(
                "Sentence could not be mapped "
                "back to its source chunk."
            )

        end_char = (
            start_char
            + len(text)
        )

        candidates.append(
            _SentenceCandidate(
                chunk=chunk,
                sentence_index=(
                    sentence_index
                ),
                start_char=start_char,
                end_char=end_char,
                text=text,
            )
        )

        cursor = end_char

    return tuple(
        candidates
    )


class SentenceEvidenceVerificationService:
    def __init__(
        self,
        *,
        verifier: EntailmentVerifierPort,
    ) -> None:
        self._verifier = verifier

    def verify(
        self,
        *,
        claim: str,
        evidence: Sequence[
            RetrievedChunk
        ],
    ) -> EvidenceVerificationResult:
        cleaned_claim = claim.strip()

        if not cleaned_claim:
            raise ValueError(
                "Claim cannot be empty."
            )

        candidates = tuple(
            candidate
            for chunk in evidence
            for candidate in (
                _sentence_candidates(
                    chunk
                )
            )
        )

        if not candidates:
            return EvidenceVerificationResult(
                claim=cleaned_claim,
                spans=(),
            )

        scores = self._verifier.score(
            claim=cleaned_claim,
            premises=tuple(
                candidate.text
                for candidate
                in candidates
            ),
        )

        if len(scores) != len(
            candidates
        ):
            raise RuntimeError(
                "Entailment verifier returned "
                "an unexpected number of scores."
            )

        spans = tuple(
            EvidenceSpan(
                chunk=candidate.chunk,
                sentence_index=(
                    candidate
                    .sentence_index
                ),
                start_char=(
                    candidate.start_char
                ),
                end_char=(
                    candidate.end_char
                ),
                text=candidate.text,
                verification=score,
            )
            for candidate, score
            in zip(
                candidates,
                scores,
                strict=True,
            )
        )

        return EvidenceVerificationResult(
            claim=cleaned_claim,
            spans=spans,
        )
