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


def _markdown_test_row_candidates(
    chunk: RetrievedChunk,
    claim: str,
) -> tuple[_SentenceCandidate, ...] | None:
    """Return exact source rows for a T-### claim.

    None means this chunk has no recognized Markdown table,
    so ordinary sentence verification may proceed.

    An empty tuple means a table was found, but no single
    unambiguous row supports checking this claim.
    """
    lines: list[tuple[int, str]] = []
    offset = 0

    for raw_line in chunk.content.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        lines.append((offset, line))
        offset += len(raw_line)

    table_found = False
    matches: list[_SentenceCandidate] = []

    for index in range(len(lines) - 1):
        _, header = lines[index]
        _, divider = lines[index + 1]

        if not header.strip().startswith("|"):
            continue

        # Recognize a Markdown header followed by its separator.
        separator_cells = divider.strip().strip("|").split("|")
        if (
            len(separator_cells) < 2
            or not all(
                re.fullmatch(r":?-{3,}:?", cell.strip())
                for cell in separator_cells
            )
        ):
            continue

        table_found = True
        row_index = index + 2

        while row_index < len(lines):
            line_start, line = lines[row_index]
            row_text = line.strip()

            if not (
                row_text.startswith("|")
                and row_text.endswith("|")
            ):
                break

            first_cell = (
                row_text.strip("|").split("|", 1)[0]
                .strip()
                .strip("*_` ")
            )

            claim_ids = {
                value.upper()
                for value in re.findall(
                    r"(?<![\w-])T-\d+(?![\w-])",
                    claim,
                    flags=re.IGNORECASE,
                )
            }

            if (
                len(claim_ids) == 1
                and first_cell.upper() in claim_ids
            ):
                start = line_start + line.index(row_text)
                end = start + len(row_text)

                matches.append(
                    _SentenceCandidate(
                        chunk=chunk,
                        sentence_index=row_index,
                        start_char=start,
                        end_char=end,
                        text=chunk.content[start:end],
                    )
                )

            row_index += 1

    if not table_found:
        return None

    # No matching ID, no ID in the claim, or ambiguous duplicates:
    # never pass the whole table to NLI as a fallback.
    if len(matches) != 1:
        return ()

    return (matches[0],)


# Experimental, opt-in fallback. Context must remain a single,
# exact, contiguous source-chunk slice.
_MAX_CONTEXT_CHARS = 1000
_HEADING_LINE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]+\S[^\r\n]*|article[ \t]+\d+[ \t]*[:.]?)[ \t]*$"
)


def _looks_like_document_title(content: str) -> bool:
    first_line = content.splitlines()[0].strip()
    return (
        20 <= len(first_line) <= 120
        and first_line == first_line.upper()
        and any(character.isalpha() for character in first_line)
        and not first_line.endswith(".")
    )


def _contextual_candidates(
    chunk: RetrievedChunk,
) -> tuple[_SentenceCandidate, ...]:
    # Never infer single-page provenance from page_number alone.
    if (
        chunk.page_number < 1
        or chunk.page_end is None
        or chunk.page_end != chunk.page_number
    ):
        return ()

    sentences = _sentence_candidates(chunk)
    expanded: list[_SentenceCandidate] = []
    seen: set[tuple[int, int]] = set()

    def add(start: int, end: int, sentence_index: int) -> None:
        if not (0 <= start < end <= len(chunk.content)):
            return
        if end - start > _MAX_CONTEXT_CHARS:
            return
        if (start, end) in seen:
            return

        # This text is a canonical substring, not a reconstructed premise.
        source_text = chunk.content[start:end]
        if not source_text.strip():
            return

        seen.add((start, end))
        expanded.append(
            _SentenceCandidate(
                chunk=chunk,
                sentence_index=sentence_index,
                start_char=start,
                end_char=end,
                text=source_text,
            )
        )

    for fact_index in range(1, len(sentences)):
        heading = sentences[fact_index - 1]
        fact = sentences[fact_index]

        # Require a short, recognizable heading immediately before
        # the factual sentence; never merge ordinary adjacent sentences.
        if len(heading.text) > 180:
            continue
        if _HEADING_LINE.search(heading.text) is None:
            continue
        if not any(character.isalpha() for character in fact.text):
            continue

        add(
            heading.start_char,
            fact.end_char,
            fact.sentence_index,
        )

        # An initial document title may resolve a later "the Court"-
        # style reference. The entire intervening text stays in the
        # exact cited slice, and the 1000-character bound still applies.
        if (
            fact_index >= 2
            and sentences[0].start_char == 0
            and _looks_like_document_title(chunk.content)
        ):
            add(0, fact.end_char, fact.sentence_index)

    return tuple(expanded)


class SentenceEvidenceVerificationService:
    def __init__(
        self,
        *,
        verifier: EntailmentVerifierPort,
        contextual_fallback: bool = False,
    ) -> None:
        self._verifier = verifier
        self._contextual_fallback = contextual_fallback

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
                _markdown_test_row_candidates(chunk, cleaned_claim)
                if _markdown_test_row_candidates(
                    chunk, cleaned_claim
                ) is not None
                else _sentence_candidates(chunk)
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

        # Preserve the original single-sentence decision whenever
        # at least one original sentence independently entails the claim.
        if (
            self._contextual_fallback
            and not any(span.is_supporting for span in spans)
        ):
            contextual = tuple(
                candidate
                for chunk in evidence
                if _markdown_test_row_candidates(
                    chunk, cleaned_claim
                ) is None
                for candidate in _contextual_candidates(chunk)
            )

            if contextual:
                contextual_scores = self._verifier.score(
                    claim=cleaned_claim,
                    premises=tuple(
                        candidate.text for candidate in contextual
                    ),
                )

                if len(contextual_scores) != len(contextual):
                    raise RuntimeError(
                        "Entailment verifier returned "
                        "an unexpected number of scores."
                    )

                contextual_spans = tuple(
                    EvidenceSpan(
                        chunk=candidate.chunk,
                        sentence_index=candidate.sentence_index,
                        start_char=candidate.start_char,
                        end_char=candidate.end_char,
                        text=candidate.text,
                        verification=score,
                    )
                    for candidate, score in zip(
                        contextual,
                        contextual_scores,
                        strict=True,
                    )
                )

                spans = spans + contextual_spans

        return EvidenceVerificationResult(
            claim=cleaned_claim,
            spans=spans,
        )
