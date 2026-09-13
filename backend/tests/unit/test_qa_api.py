from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import (
    TestClient,
)

from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.api.dependencies.qa import (
    get_citation_presentation_repository,
    get_question_answer_service,
)
from app.application.ports.citation_presentation_repository import (
    CitationPresentation,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.authorized_search import (
    AuthorizedSearchService,
)
from app.application.services.citation_mapping import (
    Citation,
)
from app.application.services.grounded_answer_generation import (
    GroundedAnswerResult,
)
from app.application.services.question_answer import (
    QuestionAnswerService,
)
from app.application.services.validated_citations import (
    ClaimCitation,
    ValidatedCitationSet,
)
from app.core.config import (
    get_settings,
)
from app.main import create_app


class FakeQuestionAnswerService:
    def __init__(
        self,
        *,
        result,
    ):
        self.result = result
        self.calls = []

    def answer(
        self,
        *,
        identity,
        question,
        language,
    ):
        self.calls.append(
            (
                identity,
                question,
                language,
            )
        )

        return self.result


class FakeCitationPresentationRepository:
    def __init__(
        self,
        presentations,
    ):
        self.presentations = {
            (
                item.page_id,
                item.document_id,
            ): item
            for item in presentations
        }

        self.calls = []

    def get_by_page_id(
        self,
        *,
        page_id,
        document_id,
    ):
        self.calls.append(
            (
                page_id,
                document_id,
            )
        )

        return self.presentations.get(
            (
                page_id,
                document_id,
            )
        )


class FailingAuthorizationRepository:
    def document_ids_for_user(
        self,
        *,
        user_id,
        action,
        active_only,
    ):
        raise RuntimeError(
            "SECRET_DATABASE_HOST=db.internal "
            "password=super-secret"
        )

    def has_document_permission(
        self,
        *,
        user_id,
        document_id,
        action,
    ):
        raise AssertionError(
            "Unexpected permission lookup."
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
            "Embedding must not run."
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
            "Retrieval must not run."
        )


class NeverPlanner:
    def __init__(self):
        self.calls = []

    def plan(
        self,
        request,
    ):
        self.calls.append(request)
        raise AssertionError(
            "Planner must not run."
        )


class NeverEvidenceService:
    def __init__(self):
        self.calls = []

    def evaluate(
        self,
        **kwargs,
    ):
        self.calls.append(kwargs)
        raise AssertionError(
            "Evidence evaluation must not run."
        )


class NeverAnswerService:
    def __init__(self):
        self.calls = []

    def generate(
        self,
        **kwargs,
    ):
        self.calls.append(kwargs)
        raise AssertionError(
            "Final answer LLM must not run."
        )


class RecordingAuditService:
    def __init__(self):
        self.calls = []

    def record(
        self,
        event,
    ):
        self.calls.append(event)
        return uuid4()


def citation_fixture(
    *,
    claim_id,
):
    chunk_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    page_id = uuid4()

    text = (
        "Le delai applicable est "
        "de sept jours."
    )

    citation = Citation(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version_id=version_id,
        page_id=page_id,
        page_number=19,
        start_char=0,
        end_char=len(text),
        cited_text=text,
    )

    claim_citation = ClaimCitation(
        claim_id=claim_id,
        claim=text,
        citation=citation,
    )

    presentation = CitationPresentation(
        page_id=page_id,
        document_id=document_id,
        document_title=(
            "Conditions generales"
        ),
        issuing_organization=(
            "Organisation Exemple"
        ),
        jurisdiction="International",
    )

    return (
        claim_citation,
        presentation,
    )


def endpoint():
    settings = get_settings()

    return (
        f"{settings.api_prefix}"
        "/qa"
    )


def client_with(
    *,
    service,
    presentation_repository,
):
    app = create_app()

    identity = object()

    app.dependency_overrides[
        get_current_identity
    ] = lambda: identity

    app.dependency_overrides[
        get_question_answer_service
    ] = lambda: service

    app.dependency_overrides[
        get_citation_presentation_repository
    ] = (
        lambda: presentation_repository
    )

    return (
        TestClient(app),
        identity,
    )


def test_unauthenticated_qa_is_rejected():
    result = GroundedAnswerResult(
        correlation_id=uuid4(),
        status="INSUFFICIENT_EVIDENCE",
        answer=(
            "Les sources autorisees "
            "disponibles ne permettent "
            "pas de repondre."
        ),
        citations=ValidatedCitationSet(
            citations=()
        ),
        cited_evidence_ids=(),
    )

    service = FakeQuestionAnswerService(
        result=result
    )

    repository = (
        FakeCitationPresentationRepository(
            []
        )
    )

    app = create_app()

    # Override protected business dependencies,
    # but keep the real authentication dependency.
    app.dependency_overrides[
        get_question_answer_service
    ] = lambda: service

    app.dependency_overrides[
        get_citation_presentation_repository
    ] = lambda: repository

    client = TestClient(app)

    response = client.post(
        endpoint(),
        json={
            "question": (
                "Quel est le delai ?"
            ),
            "language": "fr",
        },
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": (
            "Authentication required."
        )
    }

    assert (
        response.headers[
            "www-authenticate"
        ]
        == "Bearer"
    )

    # Authentication must stop the request before
    # any Q&A business logic is executed.
    assert service.calls == []

    assert repository.calls == []


def test_supported_qa_returns_only_cited_evidence():
    used, used_presentation = (
        citation_fixture(
            claim_id="claim-1"
        )
    )

    unused, unused_presentation = (
        citation_fixture(
            claim_id="claim-2"
        )
    )

    correlation_id = uuid4()

    result = GroundedAnswerResult(
        correlation_id=correlation_id,
        status="SUPPORTED",
        answer=(
            "Le delai est de sept jours."
        ),
        citations=ValidatedCitationSet(
            citations=(
                used,
                unused,
            )
        ),
        cited_evidence_ids=(
            "claim-1",
        ),
    )

    service = FakeQuestionAnswerService(
        result=result
    )

    repository = (
        FakeCitationPresentationRepository(
            [
                used_presentation,
                unused_presentation,
            ]
        )
    )

    client, identity = client_with(
        service=service,
        presentation_repository=repository,
    )

    response = client.post(
        endpoint(),
        json={
            "question": (
                "Quel est le delai ?"
            ),
            "language": "fr",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "SUPPORTED"

    assert (
        body["answer"]
        == "Le delai est de sept jours."
    )

    assert (
        body["correlation_id"]
        == str(correlation_id)
    )

    assert len(body["citations"]) == 1

    citation = body["citations"][0]

    assert (
        citation["evidence_id"]
        == "claim-1"
    )

    assert (
        citation["page_id"]
        == str(
            used.citation.page_id
        )
    )

    assert citation["page_number"] == 19

    assert (
        citation["document_title"]
        == "Conditions generales"
    )

    assert (
        citation["issuing_organization"]
        == "Organisation Exemple"
    )

    assert (
        citation["jurisdiction"]
        == "International"
    )

    assert (
        citation["cited_text"]
        == used.citation.cited_text
    )

    assert service.calls == [
        (
            identity,
            "Quel est le delai ?",
            "fr",
        )
    ]

    # Metadata lookup occurs only for evidence
    # actually cited by the final answer.
    assert repository.calls == [
        (
            used.citation.page_id,
            used.citation.document_id,
        )
    ]


def test_insufficient_qa_returns_no_citations():
    correlation_id = uuid4()

    refusal = (
        "Les sources autorisees "
        "disponibles ne permettent "
        "pas de repondre de maniere "
        "fiable a cette question."
    )

    result = GroundedAnswerResult(
        correlation_id=correlation_id,
        status="INSUFFICIENT_EVIDENCE",
        answer=refusal,
        citations=ValidatedCitationSet(
            citations=()
        ),
        cited_evidence_ids=(),
    )

    service = FakeQuestionAnswerService(
        result=result
    )

    repository = (
        FakeCitationPresentationRepository(
            []
        )
    )

    client, identity = client_with(
        service=service,
        presentation_repository=repository,
    )

    response = client.post(
        endpoint(),
        json={
            "question": (
                "La politique garantit-elle "
                "un hotel gratuit ?"
            ),
            "language": "fr",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["status"]
        == "INSUFFICIENT_EVIDENCE"
    )

    assert body["answer"] == refusal

    assert (
        body["correlation_id"]
        == str(correlation_id)
    )

    assert body["citations"] == []

    assert service.calls == [
        (
            identity,
            (
                "La politique garantit-elle "
                "un hotel gratuit ?"
            ),
            "fr",
        )
    ]

    # No citation metadata should be queried
    # for an insufficient answer.
    assert repository.calls == []


def test_authorization_infrastructure_failure_is_sanitized_over_http():
    from app.domain.enums.audit import (
        AuditEventType,
        AuditOutcome,
    )

    embedding = NeverEmbeddingProvider()
    retrieval = NeverRetrieval()
    planner = NeverPlanner()
    evidence = NeverEvidenceService()
    answer = NeverAnswerService()
    audit = RecordingAuditService()

    search = AuthorizedSearchService(
        authorization=AuthorizationService(
            FailingAuthorizationRepository()
        ),
        embedding_provider=embedding,
        retrieval=retrieval,
    )

    service = QuestionAnswerService(
        search=search,
        planner=planner,
        evidence=evidence,
        answer=answer,
        audit=audit,
        discovery_top_k=10,
    )

    app = create_app()

    test_identity = SimpleNamespace(
        user_id=uuid4()
    )

    app.dependency_overrides[
        get_current_identity
    ] = lambda: test_identity

    app.dependency_overrides[
        get_question_answer_service
    ] = lambda: service

    app.dependency_overrides[
        get_citation_presentation_repository
    ] = lambda: (
        FakeCitationPresentationRepository([])
    )

    # Critical: observe the real HTTP response rather
    # than allowing TestClient to re-raise the server
    # exception into the test process.
    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        endpoint(),
        json={
            "question": (
                "Quel est le délai ?"
            ),
            "language": "fr",
        },
    )

    assert response.status_code == 500

    response_text = response.text

    # Infrastructure details and secrets must never
    # cross the HTTP boundary.
    assert (
        "SECRET_DATABASE_HOST"
        not in response_text
    )
    assert "db.internal" not in response_text
    assert "super-secret" not in response_text
    assert "password" not in response_text.lower()

    # No downstream business/model stage ran.
    assert embedding.calls == []
    assert retrieval.calls == []
    assert planner.calls == []
    assert evidence.calls == []
    assert answer.calls == []

    # Initial discovery failure is categorized in
    # the audit trail.
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
        failed.error_code
        == "EVIDENCE_DISCOVERY_FAILED"
    )

    audit_text = str(
        failed.metadata
    )

    assert (
        "SECRET_DATABASE_HOST"
        not in audit_text
    )
    assert "super-secret" not in audit_text
