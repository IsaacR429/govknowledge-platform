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
from app.application.services.atomic_evidence_orchestration import (
    AtomicEvidenceOrchestrationService,
)
from app.application.services.atomic_evidence_policy import (
    AtomicEvidencePolicy,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.authorized_search import (
    AuthorizedSearchResult,
    AuthorizedSearchService,
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


class FakeAuditService:
    def __init__(self):
        self.calls = []

    def record(
        self,
        event,
    ):
        self.calls.append(event)
        return uuid4()


class FailingAuthorizationRepository:
    def __init__(self):
        self.calls = []

    def document_ids_for_user(
        self,
        *,
        user_id,
        action,
        active_only,
    ):
        self.calls.append(
            (
                user_id,
                action,
                active_only,
            )
        )

        raise RuntimeError(
            "SECRET_AUTHORIZATION_DATABASE_FAILURE"
        )

    def has_document_permission(
        self,
        *,
        user_id,
        document_id,
        action,
    ):
        raise AssertionError(
            "Document permission lookup "
            "must not be called."
        )


class NeverEmbeddingProvider:
    def __init__(self):
        self.calls = []

    def embed_query(
        self,
        text,
    ):
        self.calls.append(text)

        raise AssertionError(
            "Embedding must not run after "
            "authorization failure."
        )


class NeverRetrieval:
    def __init__(self):
        self.calls = []

    def search_authorized(
        self,
        **kwargs,
    ):
        self.calls.append(kwargs)

        raise AssertionError(
            "Vector retrieval must not run after "
            "authorization failure."
        )


class FailSecondAuthorizationRepository:
    def __init__(
        self,
        *,
        allowed_document_id,
    ):
        self.allowed_document_id = (
            allowed_document_id
        )
        self.calls = []

    def document_ids_for_user(
        self,
        *,
        user_id,
        action,
        active_only,
    ):
        self.calls.append(
            (
                user_id,
                action,
                active_only,
            )
        )

        if len(self.calls) == 1:
            return frozenset(
                {
                    self.allowed_document_id,
                }
            )

        raise RuntimeError(
            "SECRET_SECOND_AUTHORIZATION_FAILURE"
        )

    def has_document_permission(
        self,
        *,
        user_id,
        document_id,
        action,
    ):
        raise AssertionError(
            "Direct document permission lookup "
            "must not be called."
        )


class RecordingEmbeddingProvider:
    def __init__(self):
        self.calls = []

    def embed_query(
        self,
        text,
    ):
        self.calls.append(text)

        return SimpleNamespace(
            values=(1.0,),
            model_name="test-model",
            model_revision="test-revision",
        )


class RecordingRetrieval:
    def __init__(
        self,
        *,
        result,
    ):
        self.result = result
        self.calls = []

    def search_authorized(
        self,
        **kwargs,
    ):
        self.calls.append(kwargs)
        return (self.result,)


class NeverVerificationService:
    def __init__(self):
        self.calls = []

    def verify(
        self,
        *,
        claim,
        evidence,
    ):
        self.calls.append(
            (
                claim,
                tuple(evidence),
            )
        )

        raise AssertionError(
            "NLI must not run after "
            "second authorization failure."
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
    audit=None,
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

    audit = (
        audit
        if audit is not None
        else FakeAuditService()
    )

    return (
        QuestionAnswerService(
            search=search,
            planner=planner,
            evidence=evidence,
            answer=answer,
            audit=audit,
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


def test_initial_discovery_authorization_failure_stops_qa():
    repository = (
        FailingAuthorizationRepository()
    )

    embedding = NeverEmbeddingProvider()
    retrieval = NeverRetrieval()

    search = AuthorizedSearchService(
        authorization=AuthorizationService(
            repository
        ),
        embedding_provider=embedding,
        retrieval=retrieval,
    )

    planner = FakePlanner(
        (
            SOURCE_SENTENCE,
        )
    )

    evidence = FakeEvidenceService()
    answer = FakeAnswerService()

    qa = QuestionAnswerService(
        search=search,
        planner=planner,
        evidence=evidence,
        answer=answer,
        audit=FakeAuditService(),
        discovery_top_k=10,
    )

    test_identity = identity()

    with pytest.raises(
        RuntimeError,
        match=(
            "SECRET_AUTHORIZATION_DATABASE_FAILURE"
        ),
    ):
        qa.answer(
            identity=test_identity,
            question=(
                "Quel est le délai pour "
                "un bagage endommagé ?"
            ),
            language="fr",
        )

    # The real authorization repository was
    # actually entered once.
    assert len(
        repository.calls
    ) == 1

    # Authorization failed BEFORE any model
    # or corpus work.
    assert embedding.calls == []
    assert retrieval.calls == []

    # The natural-question pipeline must stop
    # before planner, evidence verification,
    # or final answer generation.
    assert planner.calls == []
    assert evidence.calls == []
    assert answer.calls == []


def test_initial_discovery_authorization_failure_is_audited_as_failed():
    from app.domain.enums.audit import (
        AuditEventType,
        AuditOutcome,
    )

    repository = (
        FailingAuthorizationRepository()
    )

    embedding = NeverEmbeddingProvider()
    retrieval = NeverRetrieval()

    search = AuthorizedSearchService(
        authorization=AuthorizationService(
            repository
        ),
        embedding_provider=embedding,
        retrieval=retrieval,
    )

    planner = FakePlanner(
        (
            SOURCE_SENTENCE,
        )
    )

    evidence = FakeEvidenceService()
    answer = FakeAnswerService()
    audit = FakeAuditService()

    qa = QuestionAnswerService(
        search=search,
        planner=planner,
        evidence=evidence,
        answer=answer,
        audit=audit,
        discovery_top_k=10,
    )

    test_identity = identity()

    with pytest.raises(
        RuntimeError,
        match=(
            "SECRET_AUTHORIZATION_DATABASE_FAILURE"
        ),
    ):
        qa.answer(
            identity=test_identity,
            question=(
                "Quel est le délai pour "
                "un bagage endommagé ?"
            ),
            language="fr",
        )

    # Authorization failure must stop every
    # downstream stage.
    assert embedding.calls == []
    assert retrieval.calls == []
    assert planner.calls == []
    assert evidence.calls == []
    assert answer.calls == []

    # Initial discovery needs its own
    # categorical terminal failure audit.
    assert len(audit.calls) == 1

    failed = audit.calls[0]

    assert (
        failed.event_type
        == AuditEventType.EVIDENCE_FAILED
    )

    assert (
        failed.outcome
        == AuditOutcome.FAILED
    )

    assert (
        failed.actor_user_id
        == test_identity.user_id
    )

    assert (
        failed.resource_type
        == "qa_evidence"
    )

    assert (
        failed.error_code
        == "EVIDENCE_DISCOVERY_FAILED"
    )

    assert (
        failed.metadata["stage"]
        == "authorized_discovery"
    )

    # The secret-bearing infrastructure error
    # must never enter audit metadata.
    audit_text = str(
        failed.metadata
    )

    assert (
        "SECRET_AUTHORIZATION_DATABASE_FAILURE"
        not in audit_text
    )

    assert all(
        event.event_type
        != AuditEventType.EVIDENCE_INSUFFICIENT
        for event in audit.calls
    )


def test_atomic_reauthorization_failure_is_terminal_and_not_double_audited():
    from app.domain.enums.audit import (
        AuditEventType,
        AuditOutcome,
    )

    discovery_chunk = chunk(
        text=SOURCE_SENTENCE,
        language="en",
    )

    repository = (
        FailSecondAuthorizationRepository(
            allowed_document_id=(
                discovery_chunk.document_id
            )
        )
    )

    embedding = RecordingEmbeddingProvider()

    retrieval = RecordingRetrieval(
        result=discovery_chunk
    )

    search = AuthorizedSearchService(
        authorization=AuthorizationService(
            repository
        ),
        embedding_provider=embedding,
        retrieval=retrieval,
    )

    planner = FakePlanner(
        (
            SOURCE_SENTENCE,
        )
    )

    verification = NeverVerificationService()
    audit = FakeAuditService()

    evidence = (
        AtomicEvidenceOrchestrationService(
            search=search,
            verification=verification,
            policy=AtomicEvidencePolicy(),
            audit=audit,
            top_k=5,
        )
    )

    answer = FakeAnswerService()

    qa = QuestionAnswerService(
        search=search,
        planner=planner,
        evidence=evidence,
        answer=answer,
        audit=audit,
        discovery_top_k=10,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "SECRET_SECOND_AUTHORIZATION_FAILURE"
        ),
    ):
        qa.answer(
            identity=identity(),
            question=(
                "Quel est le délai pour "
                "un bagage endommagé ?"
            ),
            language="fr",
        )

    # Authorization happened once for discovery
    # and once again for atomic evidence.
    assert len(repository.calls) == 2

    # Embedding/retrieval happened only for
    # successful discovery. The second auth
    # failure stopped the atomic search first.
    assert len(embedding.calls) == 1
    assert len(retrieval.calls) == 1

    # Planner legitimately ran after successful
    # discovery.
    assert len(planner.calls) == 1

    # But re-authorization failed before NLI.
    assert verification.calls == []

    # Final grounded answer generation is blocked.
    assert answer.calls == []

    # Exactly two evidence audit events:
    # opening query + terminal failure.
    # No duplicate failure from QuestionAnswerService.
    assert len(audit.calls) == 2

    submitted, failed = audit.calls

    assert (
        submitted.event_type
        == AuditEventType.QUERY_SUBMITTED
    )

    assert (
        failed.event_type
        == AuditEventType.EVIDENCE_FAILED
    )

    assert (
        failed.outcome
        == AuditOutcome.FAILED
    )

    assert (
        failed.error_code
        == "EVIDENCE_EVALUATION_FAILED"
    )

    assert (
        submitted.correlation_id
        == failed.correlation_id
    )

    audit_text = str(
        failed.metadata
    )

    assert (
        "SECRET_SECOND_AUTHORIZATION_FAILURE"
        not in audit_text
    )

    assert all(
        event.event_type
        != AuditEventType.EVIDENCE_INSUFFICIENT
        for event in audit.calls
    )
