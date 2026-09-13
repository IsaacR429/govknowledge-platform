from __future__ import annotations

from uuid import uuid4

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
from app.application.services.citation_mapping import (
    Citation,
)
from app.application.services.grounded_answer_generation import (
    GroundedAnswerResult,
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
