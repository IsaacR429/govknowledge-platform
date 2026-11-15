from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.application.ports.question_claim_planner import PlannedClaim
from app.application.services.audit import AuditService
from app.application.services.question_answer import (
    InvalidPlannedClaimError,
    QuestionAnswerService,
)
from app.domain.enums.audit import AuditEventType


SOURCE_FACT = "The Court shall consist of fifteen members."


class MemoryAuditRepository:
    def __init__(self):
        self.events = []

    def append(self, event):
        self.events.append(event)
        return uuid4()


class SpySearch:
    def __init__(self, results=(), error=None):
        self.results = results
        self.error = error
        self.calls = []

    def search(self, *, identity, query, top_k):
        self.calls.append((identity, query, top_k))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(results=self.results)


class SpyPlanner:
    def __init__(self, claim_text, evidence_id="evidence-1"):
        self.claim_text = claim_text
        self.evidence_id = evidence_id
        self.calls = []

    def plan(self, request):
        self.calls.append(request)
        return (
            PlannedClaim(
                evidence_id=self.evidence_id,
                text=self.claim_text,
            ),
        )


class SpyEvidence:
    def __init__(self):
        self.calls = []
        self.result = SimpleNamespace(
            assessment=SimpleNamespace(
                is_sufficient=False,
                selected_evidence=(),
            ),
            claim_results=(),
        )

    def evaluate(self, *, identity, claims, correlation_id):
        self.calls.append((identity, claims, correlation_id))
        return self.result


class SpyAnswer:
    def __init__(self):
        self.calls = []
        self.result = SimpleNamespace(
            correlation_id=uuid4(),
            status="SUPPORTED",
            citations=SimpleNamespace(citations=()),
        )

    def generate(self, *, question, evidence_result, language):
        self.calls.append((question, evidence_result, language))
        return self.result


class SpyAuthorization:
    def require_document_read(self, identity, document_id):
        pass


def fixture(*, results=(), claim_text=SOURCE_FACT, error=None):
    identity = SimpleNamespace(user_id=uuid4())
    search = SpySearch(results=results, error=error)
    planner = SpyPlanner(claim_text)
    evidence = SpyEvidence()
    answer = SpyAnswer()
    audit_repository = MemoryAuditRepository()

    service = QuestionAnswerService(
        search=search,
        planner=planner,
        evidence=evidence,
        answer=answer,
        authorization=SpyAuthorization(),
        audit=AuditService(audit_repository),
        discovery_top_k=5,
    )

    return SimpleNamespace(
        service=service,
        identity=identity,
        search=search,
        planner=planner,
        evidence=evidence,
        answer=answer,
        audit=audit_repository,
    )


def authorized_hit(content=SOURCE_FACT):
    return SimpleNamespace(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=1,
        language="en",
        content=content,
    )


def test_empty_discovery_skips_planner_and_passes_empty_claims():
    case = fixture()

    result = case.service.answer(
        identity=case.identity,
        question="How many members?",
        language="fr",
    )

    assert result is case.answer.result
    assert case.planner.calls == []
    assert len(case.evidence.calls) == 1
    assert case.evidence.calls[0][1] == ()
    assert case.answer.calls == [
        ("How many members?", case.evidence.result, "fr")
    ]


def test_invented_planner_claim_is_rejected_before_evidence():
    case = fixture(
        results=(authorized_hit(),),
        claim_text="The Court consists of sixteen members.",
    )

    with pytest.raises(
        InvalidPlannedClaimError,
        match="Planner quote is not present in the selected authorized evidence.",
    ):
        case.service.answer(
            identity=case.identity,
            question="How many members?",
            language="fr",
        )

    assert len(case.planner.calls) == 1
    assert case.evidence.calls == []
    assert case.answer.calls == []


def test_verbatim_source_claim_reaches_evidence_evaluation():
    case = fixture(
        results=(
            authorized_hit(
                "Article 3\n\n1. " + SOURCE_FACT
            ),
        ),
        claim_text=SOURCE_FACT,
    )

    result = case.service.answer(
        identity=case.identity,
        question="How many members?",
        language="fr",
    )

    assert result is case.answer.result

    planning_request = case.planner.calls[0]
    assert planning_request.question_language == "fr"
    assert planning_request.source_language == "en"
    assert len(planning_request.evidence) == 1
    assert SOURCE_FACT in planning_request.evidence[0].content

    assert len(case.evidence.calls) == 1
    identity, claims, correlation_id = case.evidence.calls[0]
    assert identity is case.identity
    assert correlation_id is not None
    assert len(claims) == 1
    assert claims[0].text == SOURCE_FACT
    assert claims[0].language == "en"
    assert claims[0].required is True

    assert case.answer.calls == [
        ("How many members?", case.evidence.result, "fr")
    ]


def test_discovery_failure_audits_without_question_text():
    case = fixture(error=RuntimeError("discovery blocked"))
    question = "Private test question that must not enter audit metadata"

    with pytest.raises(RuntimeError, match="discovery blocked"):
        case.service.answer(
            identity=case.identity,
            question=question,
            language="fr",
        )

    assert case.planner.calls == []
    assert case.evidence.calls == []
    assert case.answer.calls == []

    assert len(case.audit.events) == 1
    event = case.audit.events[0]
    assert event.event_type == AuditEventType.EVIDENCE_FAILED
    assert event.error_code == "EVIDENCE_DISCOVERY_FAILED"
    assert event.actor_user_id == case.identity.user_id
    assert event.metadata == {"stage": "authorized_discovery"}
    assert question not in repr(event.metadata)
