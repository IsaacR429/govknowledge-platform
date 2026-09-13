from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.application.ports.question_claim_planner import (
    PlannedClaim,
)
from app.application.ports.retrieval import (
    RetrievedChunk,
)
from app.application.services.authorized_search import (
    AuthorizedSearchResult,
)
from app.application.services.question_answer import (
    InvalidPlannedClaimError,
    QuestionAnswerService,
)


SOURCE_SENTENCE = (
    "If the Checked Baggage is damaged, "
    "the person authorized to take delivery "
    "of the Baggage must notify immediately, "
    "and in any case, within not later than "
    "seven days following receipt of the Baggage."
)


def chunk(
    *,
    text: str = SOURCE_SENTENCE,
    language: str = "en",
    index: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=19,
        chunk_index=index,
        content=text,
        section_title=None,
        language=language,
        score=0.90,
    )


class FakeSearch:
    def __init__(
        self,
        results,
    ):
        self.results = tuple(results)
        self.calls = []

    def search(
        self,
        *,
        identity,
        query,
        top_k=5,
    ):
        self.calls.append(
            (
                identity,
                query,
                top_k,
            )
        )

        return AuthorizedSearchResult(
            query=query,
            results=self.results,
        )


class FakePlanner:
    def __init__(
        self,
        claims,
    ):
        self.claims = tuple(
            PlannedClaim(text=item)
            for item in claims
        )
        self.calls = []

    def plan(
        self,
        request,
    ):
        self.calls.append(request)

        return self.claims


class FakeEvidenceService:
    def __init__(self):
        self.calls = []

    def evaluate(
        self,
        *,
        identity,
        claims,
        correlation_id=None,
    ):
        claims = tuple(claims)

        self.calls.append(
            (
                identity,
                claims,
                correlation_id,
            )
        )

        return SimpleNamespace(
            correlation_id=correlation_id,
            claims=claims,
        )


class FakeAnswerService:
    def __init__(self):
        self.calls = []

    def generate(
        self,
        *,
        question,
        evidence_result,
        language="fr",
    ):
        self.calls.append(
            (
                question,
                evidence_result,
                language,
            )
        )

        return SimpleNamespace(
            status="TEST",
            question=question,
            evidence_result=evidence_result,
            language=language,
        )


def identity():
    return SimpleNamespace(
        user_id=uuid4(),
        language_preference="fr",
    )


def service(
    *,
    search,
    planner,
    evidence=None,
    answer=None,
):
    evidence = (
        evidence
        if evidence is not None
        else FakeEvidenceService()
    )

    answer = (
        answer
        if answer is not None
        else FakeAnswerService()
    )

    return (
        QuestionAnswerService(
            search=search,
            planner=planner,
            evidence=evidence,
            answer=answer,
        ),
        evidence,
        answer,
    )


def test_exact_source_claim_becomes_atomic_claim():
    search = FakeSearch(
        (
            chunk(),
        )
    )

    planner = FakePlanner(
        (
            SOURCE_SENTENCE,
        )
    )

    qa, evidence, answer = service(
        search=search,
        planner=planner,
    )

    result = qa.answer(
        identity=identity(),
        question=(
            "Quel est le délai pour "
            "un bagage endommagé ?"
        ),
        language="fr",
    )

    assert result.status == "TEST"

    assert len(
        evidence.calls
    ) == 1

    _, claims, correlation_id = (
        evidence.calls[0]
    )

    assert correlation_id is not None
    assert len(claims) == 1

    atomic = claims[0]

    assert atomic.claim_id == "claim-1"
    assert atomic.text == SOURCE_SENTENCE
    assert atomic.language == "en"
    assert atomic.required is True

    assert len(
        answer.calls
    ) == 1


def test_claim_not_present_in_authorized_evidence_is_rejected():
    search = FakeSearch(
        (
            chunk(),
        )
    )

    planner = FakePlanner(
        (
            "Passengers always receive "
            "a complimentary hotel.",
        )
    )

    qa, evidence, answer = service(
        search=search,
        planner=planner,
    )

    with pytest.raises(
        InvalidPlannedClaimError,
        match=(
            "not present in authorized "
            "source evidence"
        ),
    ):
        qa.answer(
            identity=identity(),
            question="Question",
            language="fr",
        )

    assert evidence.calls == []
    assert answer.calls == []


def test_only_top_source_language_is_sent_to_planner():
    english = chunk(
        text=SOURCE_SENTENCE,
        language="en",
        index=0,
    )

    french = chunk(
        text=(
            "Le délai est de sept jours."
        ),
        language="fr",
        index=1,
    )

    search = FakeSearch(
        (
            english,
            french,
        )
    )

    planner = FakePlanner(
        (
            SOURCE_SENTENCE,
        )
    )

    qa, _, _ = service(
        search=search,
        planner=planner,
    )

    qa.answer(
        identity=identity(),
        question="Quel est le délai ?",
        language="fr",
    )

    assert len(
        planner.calls
    ) == 1

    request = planner.calls[0]

    assert (
        request.source_language
        == "en"
    )

    assert len(
        request.evidence
    ) == 1

    assert (
        request.evidence[0].language
        == "en"
    )

    assert (
        request.evidence[0].content
        == SOURCE_SENTENCE
    )


def test_no_authorized_results_skips_planner():
    search = FakeSearch(())
    planner = FakePlanner(
        (
            SOURCE_SENTENCE,
        )
    )

    qa, evidence, answer = service(
        search=search,
        planner=planner,
    )

    qa.answer(
        identity=identity(),
        question="Question",
        language="fr",
    )

    assert planner.calls == []

    assert len(
        evidence.calls
    ) == 1

    _, claims, correlation_id = (
        evidence.calls[0]
    )

    assert claims == ()
    assert correlation_id is not None

    assert len(
        answer.calls
    ) == 1


def test_empty_planner_result_enters_insufficient_path():
    search = FakeSearch(
        (
            chunk(),
        )
    )

    planner = FakePlanner(())

    qa, evidence, answer = service(
        search=search,
        planner=planner,
    )

    qa.answer(
        identity=identity(),
        question="Question",
        language="fr",
    )

    assert len(
        evidence.calls
    ) == 1

    _, claims, _ = evidence.calls[0]

    assert claims == ()

    assert len(
        answer.calls
    ) == 1


def test_duplicate_planned_claims_are_rejected():
    search = FakeSearch(
        (
            chunk(),
        )
    )

    planner = FakePlanner(
        (
            SOURCE_SENTENCE,
            SOURCE_SENTENCE,
        )
    )

    qa, evidence, answer = service(
        search=search,
        planner=planner,
    )

    with pytest.raises(
        InvalidPlannedClaimError,
        match="duplicate claims",
    ):
        qa.answer(
            identity=identity(),
            question="Question",
            language="fr",
        )

    assert evidence.calls == []
    assert answer.calls == []


def test_question_and_language_are_validated():
    qa, _, _ = service(
        search=FakeSearch(()),
        planner=FakePlanner(()),
    )

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        qa.answer(
            identity=identity(),
            question="   ",
            language="fr",
        )

    with pytest.raises(
        ValueError,
        match="language cannot be empty",
    ):
        qa.answer(
            identity=identity(),
            question="Question",
            language="   ",
        )
