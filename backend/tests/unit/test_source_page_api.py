from __future__ import annotations

from datetime import date
from uuid import uuid4

from fastapi.testclient import (
    TestClient,
)

from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.api.dependencies.source_pages import (
    get_authorized_source_page_service,
)
from app.application.ports.source_page_repository import (
    SourcePage,
)
from app.core.config import (
    get_settings,
)
from app.domain.exceptions.authorization import (
    AuthorizationDeniedError,
)
from app.domain.exceptions.source_page import (
    SourcePageNotFoundError,
)
from app.main import create_app


class FakeSourcePageService:
    def __init__(
        self,
        *,
        page=None,
        error=None,
    ):
        self.page = page
        self.error = error
        self.calls = []

    def get(
        self,
        *,
        identity,
        page_id,
        correlation_id=None,
    ):
        self.calls.append(
            (
                identity,
                page_id,
                correlation_id,
            )
        )

        if self.error is not None:
            raise self.error

        return self.page


def source_page():
    return SourcePage(
        page_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_number=12,
        extracted_text=(
            "Texte officiel de la page."
        ),
        metadata={
            "width": 612,
            "height": 792,
        },
        document_title=(
            "Reglement des marches"
        ),
        issuing_organization=(
            "Ministere des Finances"
        ),
        jurisdiction=(
            "Burkina Faso"
        ),
        document_language="fr",
        publication_date=(
            date(2026, 2, 1)
        ),
        source_url=(
            "https://example.invalid/document"
        ),
        version_number=1,
        original_filename=(
            "reglement.pdf"
        ),
    )


def client_with(
    service,
):
    app = create_app()

    identity = object()

    app.dependency_overrides[
        get_current_identity
    ] = lambda: identity

    app.dependency_overrides[
        get_authorized_source_page_service
    ] = lambda: service

    return (
        TestClient(app),
        identity,
    )


def endpoint(
    page_id,
):
    settings = get_settings()

    return (
        f"{settings.api_prefix}"
        f"/source-pages/{page_id}"
    )


def test_unauthenticated_source_page_is_rejected():
    service = FakeSourcePageService(
        page=source_page()
    )

    app = create_app()

    # Override only the protected business service.
    # Do NOT override get_current_identity here:
    # we want the real authentication dependency.
    app.dependency_overrides[
        get_authorized_source_page_service
    ] = lambda: service

    client = TestClient(app)

    response = client.get(
        endpoint(uuid4())
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Authentication required."
    }

    assert (
        response.headers[
            "www-authenticate"
        ]
        == "Bearer"
    )

    # Authentication must stop the request before
    # protected source-page logic is executed.
    assert service.calls == []


def test_authorized_source_page_is_returned():
    page = source_page()

    service = FakeSourcePageService(
        page=page
    )

    client, identity = client_with(
        service
    )

    response = client.get(
        endpoint(page.page_id)
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["page_id"]
        == str(page.page_id)
    )

    assert (
        body["document_id"]
        == str(page.document_id)
    )

    assert body["page_number"] == 12

    assert (
        body["issuing_organization"]
        == "Ministere des Finances"
    )

    assert (
        body["jurisdiction"]
        == "Burkina Faso"
    )

    assert (
        body["extracted_text"]
        == "Texte officiel de la page."
    )

    assert "storage_path" not in body
    assert "file_hash" not in body

    assert service.calls == [
        (
            identity,
            page.page_id,
            None,
        )
    ]


def test_missing_page_returns_generic_404():
    page_id = uuid4()

    service = FakeSourcePageService(
        error=SourcePageNotFoundError(
            page_id
        )
    )

    client, _ = client_with(
        service
    )

    response = client.get(
        endpoint(page_id)
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": (
            "Source page not found."
        )
    }


def test_unauthorized_page_uses_same_404():
    page_id = uuid4()
    document_id = uuid4()

    service = FakeSourcePageService(
        error=AuthorizationDeniedError(
            document_id=document_id
        )
    )

    client, _ = client_with(
        service
    )

    response = client.get(
        endpoint(page_id)
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": (
            "Source page not found."
        )
    }


def test_malformed_page_id_is_rejected_before_source_page_lookup():
    service = FakeSourcePageService(
        page=source_page()
    )

    client, _ = client_with(
        service
    )

    response = client.get(
        endpoint(
            "definitely-not-a-uuid"
        )
    )

    print(
        "Malformed UUID HTTP status:",
        response.status_code,
    )

    print(
        "Malformed UUID response:",
        response.text,
    )

    # FastAPI path-parameter validation rejects
    # malformed UUIDs before the protected
    # source-page business method executes.
    assert response.status_code == 422

    assert service.calls == []

    response_text = response.text.lower()

    # Validation details may describe the UUID
    # problem, but must not expose server internals.
    assert "traceback" not in response_text
    assert "stack trace" not in response_text
    assert "sqlalchemy" not in response_text
    assert "postgres" not in response_text
    assert "storage_path" not in response_text
    assert "file_hash" not in response_text


def test_original_pdf_denied_access_returns_generic_404():
    from app.api.dependencies.source_pages import (
        get_source_page_storage,
    )
    from app.infrastructure.database.session import get_db

    page_id = uuid4()
    identity = object()

    class DenyingService:
        def __init__(self):
            self.calls = []

        def authorize(self, *, identity, page_id, correlation_id=None):
            self.calls.append((identity, page_id, correlation_id))
            raise AuthorizationDeniedError(
                document_id=uuid4()
            )

        def record_view(self, **kwargs):
            raise AssertionError(
                "A denied page must not be audited as viewed."
            )

    class NoDatabaseAccess:
        def get(self, *args, **kwargs):
            raise AssertionError(
                "A denied page must not reach database file lookup."
            )

    service = DenyingService()
    app = create_app()

    app.dependency_overrides[get_current_identity] = lambda: identity
    app.dependency_overrides[
        get_authorized_source_page_service
    ] = lambda: service
    app.dependency_overrides[get_db] = lambda: NoDatabaseAccess()
    app.dependency_overrides[
        get_source_page_storage
    ] = lambda: object()

    with TestClient(app) as client:
        response = client.get(
            f"{endpoint(page_id)}/original.pdf"
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Source page not found."
    }
    assert service.calls == [(identity, page_id, None)]


def test_source_page_forwards_correlation_id():
    page = source_page()
    service = FakeSourcePageService(page=page)
    client, identity = client_with(service)
    correlation_id = uuid4()

    response = client.get(
        f"{endpoint(page.page_id)}?correlation_id={correlation_id}"
    )

    assert response.status_code == 200
    assert service.calls == [
        (identity, page.page_id, correlation_id)
    ]


def test_original_pdf_forwards_correlation_id():
    from app.api.dependencies.source_pages import (
        get_source_page_storage,
    )
    from app.infrastructure.database.session import get_db

    page_id = uuid4()
    identity = object()
    correlation_id = uuid4()

    class DenyingService:
        def __init__(self):
            self.calls = []

        def authorize(
            self, *, identity, page_id, correlation_id=None
        ):
            self.calls.append((identity, page_id, correlation_id))
            raise AuthorizationDeniedError(
                document_id=uuid4()
            )

        def record_view(self, **kwargs):
            raise AssertionError(
                "A denied page must not be audited as viewed."
            )

    service = DenyingService()
    app = create_app()

    app.dependency_overrides[get_current_identity] = lambda: identity
    app.dependency_overrides[
        get_authorized_source_page_service
    ] = lambda: service
    app.dependency_overrides[get_db] = lambda: None
    app.dependency_overrides[
        get_source_page_storage
    ] = lambda: object()

    with TestClient(app) as client:
        response = client.get(
            f"{endpoint(page_id)}/original.pdf?correlation_id={correlation_id}"
        )

    assert response.status_code == 404
    assert service.calls == [
        (identity, page_id, correlation_id)
    ]
