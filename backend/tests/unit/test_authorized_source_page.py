from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.source_page_repository import (
    SourcePage,
)
from app.application.services.audit import (
    AuditPersistenceError,
)
from app.application.services.authorized_source_page import (
    AuthorizedSourcePageService,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)
from app.domain.exceptions.authorization import (
    AuthorizationDeniedError,
)
from app.domain.exceptions.source_page import (
    SourcePageNotFoundError,
)


class FakeSourcePageRepository:
    def __init__(
        self,
        page=None,
    ):
        self.page = page
        self.calls = []

    def get_by_id(
        self,
        page_id,
    ):
        self.calls.append(page_id)

        if (
            self.page is not None
            and self.page.page_id
            == page_id
        ):
            return self.page

        return None


class FakeAuthorizationService:
    def __init__(
        self,
        *,
        denied=False,
    ):
        self.denied = denied
        self.calls = []

    def require_document_read(
        self,
        identity,
        document_id,
    ):
        self.calls.append(
            (
                identity,
                document_id,
            )
        )

        if self.denied:
            raise AuthorizationDeniedError(
                document_id=document_id
            )


class FakeAuditService:
    def __init__(
        self,
        *,
        fail=False,
    ):
        self.fail = fail
        self.calls = []

    def record(
        self,
        event,
    ):
        self.calls.append(event)

        if self.fail:
            raise AuditPersistenceError(
                "Audit unavailable."
            )

        return uuid4()


def current_identity():
    return CurrentIdentity(
        user_id=uuid4(),
        username="employee",
        display_name="Employee",
        email=None,
        department="Finance",
        language_preference="fr",
        roles=("EMPLOYEE",),
    )


def source_page():
    return SourcePage(
        page_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_number=17,
        extracted_text=(
            "Canonical source text."
        ),
        metadata={
            "width": 612,
            "height": 792,
        },
        document_title=(
            "Government Regulation"
        ),
        issuing_organization=(
            "Ministry of Finance"
        ),
        jurisdiction=(
            "Burkina Faso"
        ),
        document_language="fr",
        publication_date=(
            date(2026, 1, 15)
        ),
        source_url=(
            "https://example.invalid/source"
        ),
        version_number=2,
        original_filename=(
            "regulation.pdf"
        ),
    )


def test_returns_page_after_authorization_and_audit():
    page = source_page()

    repository = FakeSourcePageRepository(
        page
    )

    authorization = (
        FakeAuthorizationService()
    )

    audit = FakeAuditService()
    identity = current_identity()

    result = AuthorizedSourcePageService(
        repository=repository,
        authorization=authorization,
        audit=audit,
    ).get(
        identity=identity,
        page_id=page.page_id,
    )

    assert result == page

    assert repository.calls == [
        page.page_id
    ]

    assert authorization.calls == [
        (
            identity,
            page.document_id,
        )
    ]

    assert len(audit.calls) == 1

    event = audit.calls[0]

    assert (
        event.event_type
        == AuditEventType.SOURCE_PAGE_VIEWED
    )
    assert (
        event.outcome
        == AuditOutcome.SUCCEEDED
    )
    assert (
        event.actor_user_id
        == identity.user_id
    )
    assert event.resource_type == "source_page"
    assert event.resource_id == page.page_id
    assert event.document_id == page.document_id
    assert event.error_code is None
    assert event.metadata == {
        "permission": "READ",
    }


def test_missing_page_does_not_authorize_or_audit():
    repository = (
        FakeSourcePageRepository()
    )

    authorization = (
        FakeAuthorizationService()
    )

    audit = FakeAuditService()

    page_id = uuid4()

    with pytest.raises(
        SourcePageNotFoundError
    ):
        AuthorizedSourcePageService(
            repository=repository,
            authorization=authorization,
            audit=audit,
        ).get(
            identity=current_identity(),
            page_id=page_id,
        )

    assert authorization.calls == []
    assert audit.calls == []


def test_unauthorized_page_is_audited_and_not_returned():
    page = source_page()

    repository = (
        FakeSourcePageRepository(
            page
        )
    )

    authorization = (
        FakeAuthorizationService(
            denied=True
        )
    )

    audit = FakeAuditService()
    identity = current_identity()

    with pytest.raises(
        AuthorizationDeniedError
    ):
        AuthorizedSourcePageService(
            repository=repository,
            authorization=authorization,
            audit=audit,
        ).get(
            identity=identity,
            page_id=page.page_id,
        )

    assert len(audit.calls) == 1

    event = audit.calls[0]

    assert (
        event.event_type
        == AuditEventType.SOURCE_PAGE_DENIED
    )
    assert event.outcome == AuditOutcome.DENIED
    assert (
        event.actor_user_id
        == identity.user_id
    )
    assert event.resource_id == page.page_id
    assert event.document_id == page.document_id
    assert (
        event.error_code
        == "DOCUMENT_READ_DENIED"
    )


def test_authorized_page_fails_closed_when_audit_fails():
    page = source_page()

    audit = FakeAuditService(
        fail=True
    )

    with pytest.raises(
        AuditPersistenceError
    ):
        AuthorizedSourcePageService(
            repository=(
                FakeSourcePageRepository(
                    page
                )
            ),
            authorization=(
                FakeAuthorizationService()
            ),
            audit=audit,
        ).get(
            identity=current_identity(),
            page_id=page.page_id,
        )

    assert len(audit.calls) == 1


def test_denied_page_preserves_denial_when_audit_fails():
    page = source_page()

    audit = FakeAuditService(
        fail=True
    )

    with pytest.raises(
        AuthorizationDeniedError
    ):
        AuthorizedSourcePageService(
            repository=(
                FakeSourcePageRepository(
                    page
                )
            ),
            authorization=(
                FakeAuthorizationService(
                    denied=True
                )
            ),
            audit=audit,
        ).get(
            identity=current_identity(),
            page_id=page.page_id,
        )

    assert len(audit.calls) == 1


def test_source_page_contains_no_storage_path():
    page = source_page()

    assert not hasattr(
        page,
        "storage_path",
    )

    assert not hasattr(
        page,
        "file_hash",
    )


def test_source_page_view_preserves_query_correlation():
    page = source_page()
    correlation_id = uuid4()

    audit = FakeAuditService()

    identity = current_identity()

    result = AuthorizedSourcePageService(
        repository=(
            FakeSourcePageRepository(
                page
            )
        ),
        authorization=(
            FakeAuthorizationService()
        ),
        audit=audit,
    ).get(
        identity=identity,
        page_id=page.page_id,
        correlation_id=correlation_id,
    )

    assert result == page

    assert len(audit.calls) == 1

    assert (
        audit.calls[0].correlation_id
        == correlation_id
    )


def test_direct_source_page_view_has_no_correlation():
    page = source_page()

    audit = FakeAuditService()

    identity = current_identity()

    AuthorizedSourcePageService(
        repository=(
            FakeSourcePageRepository(
                page
            )
        ),
        authorization=(
            FakeAuthorizationService()
        ),
        audit=audit,
    ).get(
        identity=identity,
        page_id=page.page_id,
    )

    assert len(audit.calls) == 1

    assert (
        audit.calls[0].correlation_id
        is None
    )
