from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from uuid import uuid4

from fastapi.testclient import (
    TestClient,
)

from app.api.dependencies.audit_read import (
    get_audit_read_service,
)
from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.application.ports.audit_read_repository import (
    AuditEventPage,
    AuditEventRead,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.main import create_app


class FakeAuditReadService:
    def __init__(
        self,
    ):
        self.calls = []

    def list_events(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        return AuditEventPage(
            items=(
                AuditEventRead(
                    event_id=uuid4(),
                    occurred_at=(
                        datetime.now(
                            timezone.utc
                        )
                    ),
                    event_type=(
                        "SOURCE_PAGE_DENIED"
                    ),
                    outcome="DENIED",
                    actor_user_id=(
                        uuid4()
                    ),
                    actor_username=(
                        "employee-a"
                    ),
                    actor_display_name=(
                        "Employee A"
                    ),
                    resource_type=(
                        "source_page"
                    ),
                    resource_id=(
                        uuid4()
                    ),
                    document_id=(
                        uuid4()
                    ),
                    correlation_id=(
                        uuid4()
                    ),
                    error_code=(
                        "DOCUMENT_READ_DENIED"
                    ),
                    metadata={
                        "permission": (
                            "READ"
                        ),
                    },
                ),
            ),
            total=1,
            limit=kwargs[
                "limit"
            ],
            offset=kwargs[
                "offset"
            ],
        )


def identity(
    role,
):
    return CurrentIdentity(
        user_id=uuid4(),
        username="audit-test",
        display_name=(
            "Audit Test"
        ),
        email=None,
        department="QA",
        language_preference="fr",
        roles=(
            role,
        ),
    )


def test_auditor_can_list_events():

    app = create_app()

    fake = (
        FakeAuditReadService()
    )

    app.dependency_overrides[
        get_current_identity
    ] = lambda: identity(
        "AUDITOR"
    )

    app.dependency_overrides[
        get_audit_read_service
    ] = lambda: fake

    with TestClient(
        app
    ) as client:

        response = client.get(
            "/api/v1/audit/events",
            params={
                "limit": 20,
                "offset": 0,
                "event_type": (
                    "SOURCE_PAGE_DENIED"
                ),
            },
        )

    assert (
        response.status_code
        == 200
    ), response.text

    body = (
        response.json()
    )

    assert (
        body["total"]
        == 1
    )

    assert (
        body["has_more"]
        is False
    )

    assert (
        body["next_offset"]
        is None
    )

    assert (
        body["items"][0][
            "event_type"
        ]
        == "SOURCE_PAGE_DENIED"
    )

    assert (
        body["items"][0][
            "error_code"
        ]
        == "DOCUMENT_READ_DENIED"
    )

    assert (
        fake.calls[0][
            "event_type"
        ].value
        == "SOURCE_PAGE_DENIED"
    )


def test_admin_can_list_events():

    app = create_app()

    fake = (
        FakeAuditReadService()
    )

    app.dependency_overrides[
        get_current_identity
    ] = lambda: identity(
        "ADMIN"
    )

    app.dependency_overrides[
        get_audit_read_service
    ] = lambda: fake

    with TestClient(
        app
    ) as client:

        response = client.get(
            "/api/v1/audit/events"
        )

    assert (
        response.status_code
        == 200
    ), response.text


def test_employee_is_forbidden():

    app = create_app()

    app.dependency_overrides[
        get_current_identity
    ] = lambda: identity(
        "EMPLOYEE"
    )

    app.dependency_overrides[
        get_audit_read_service
    ] = lambda: (
        FakeAuditReadService()
    )

    with TestClient(
        app
    ) as client:

        response = client.get(
            "/api/v1/audit/events"
        )

    assert (
        response.status_code
        == 403
    )


def test_anonymous_is_unauthorized():

    app = create_app()

    # Avoid touching the real database during
    # this unit-level authorization test.
    app.dependency_overrides[
        get_audit_read_service
    ] = lambda: (
        FakeAuditReadService()
    )

    with TestClient(
        app
    ) as client:

        response = client.get(
            "/api/v1/audit/events"
        )

    assert (
        response.status_code
        == 401
    )
