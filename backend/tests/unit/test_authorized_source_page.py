from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest

from app.application.ports.source_page_repository import (
    SourcePage,
)
from app.application.services.authorized_source_page import (
    AuthorizedSourcePageService,
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
        self.calls.append(
            page_id
        )

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


def test_returns_page_after_authorization():
    page = source_page()

    repository = (
        FakeSourcePageRepository(
            page
        )
    )

    authorization = (
        FakeAuthorizationService()
    )

    identity = object()

    result = (
        AuthorizedSourcePageService(
            repository=repository,
            authorization=authorization,
        )
        .get(
            identity=identity,
            page_id=page.page_id,
        )
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


def test_missing_page_does_not_call_authorization():
    repository = (
        FakeSourcePageRepository()
    )

    authorization = (
        FakeAuthorizationService()
    )

    page_id = uuid4()

    with pytest.raises(
        SourcePageNotFoundError
    ):
        (
            AuthorizedSourcePageService(
                repository=repository,
                authorization=authorization,
            )
            .get(
                identity=object(),
                page_id=page_id,
            )
        )

    assert authorization.calls == []


def test_unauthorized_page_is_not_returned():
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

    with pytest.raises(
        AuthorizationDeniedError
    ):
        (
            AuthorizedSourcePageService(
                repository=repository,
                authorization=authorization,
            )
            .get(
                identity=object(),
                page_id=page.page_id,
            )
        )


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
