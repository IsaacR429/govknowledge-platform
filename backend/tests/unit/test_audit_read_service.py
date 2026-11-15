from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from uuid import uuid4

import pytest

from app.application.ports.audit_read_repository import (
    AuditEventPage,
    AuditEventRead,
)
from app.application.services.audit_read import (
    AuditReadService,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)


class FakeRepository:
    def __init__(
        self,
        items,
    ):
        self.items = tuple(
            items
        )

        self.calls = []

    def list_events(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        return AuditEventPage(
            items=self.items,
            total=len(
                self.items
            ),
            limit=kwargs[
                "limit"
            ],
            offset=kwargs[
                "offset"
            ],
        )


def event(
    *,
    metadata,
):
    return AuditEventRead(
        event_id=uuid4(),
        occurred_at=(
            datetime.now(
                timezone.utc
            )
        ),
        event_type=(
            "QUERY_SUBMITTED"
        ),
        outcome="SUCCEEDED",
        actor_user_id=uuid4(),
        actor_username="audited-user",
        actor_display_name=(
            "Audited User"
        ),
        resource_type="qa_query",
        resource_id=None,
        document_id=None,
        correlation_id=uuid4(),
        error_code=None,
        metadata=metadata,
    )


def test_safe_metadata_is_returned():

    repository = FakeRepository(
        [
            event(
                metadata={
                    "language": "fr",
                    "result_count": 4,
                }
            ),
        ]
    )

    result = AuditReadService(
        repository
    ).list_events()

    assert (
        result.items[0].metadata
        == {
            "language": "fr",
            "result_count": 4,
        }
    )


def test_unsafe_historical_metadata_is_redacted():

    repository = FakeRepository(
        [
            event(
                metadata={
                    "access_token": (
                        "DO-NOT-EXPOSE"
                    ),
                }
            ),
        ]
    )

    result = AuditReadService(
        repository
    ).list_events()

    assert (
        result.items[0].metadata
        == {
            "redacted": True,
        }
    )


def test_filters_are_forwarded_as_canonical_values():

    repository = FakeRepository(
        []
    )

    user_id = uuid4()
    document_id = uuid4()
    correlation_id = uuid4()

    AuditReadService(
        repository
    ).list_events(
        limit=25,
        offset=50,
        event_type=(
            AuditEventType
            .ANSWER_REFUSED
        ),
        outcome=(
            AuditOutcome
            .SUCCEEDED
        ),
        actor_user_id=user_id,
        document_id=document_id,
        correlation_id=(
            correlation_id
        ),
    )

    call = repository.calls[0]

    assert (
        call["event_type"]
        == "ANSWER_REFUSED"
    )

    assert (
        call["outcome"]
        == "SUCCEEDED"
    )

    assert (
        call["actor_user_id"]
        == user_id
    )

    assert (
        call["document_id"]
        == document_id
    )

    assert (
        call["correlation_id"]
        == correlation_id
    )


def test_invalid_limit_is_rejected():

    repository = FakeRepository(
        []
    )

    with pytest.raises(
        ValueError
    ):
        AuditReadService(
            repository
        ).list_events(
            limit=0
        )


def test_negative_offset_is_rejected():

    repository = FakeRepository(
        []
    )

    with pytest.raises(
        ValueError
    ):
        AuditReadService(
            repository
        ).list_events(
            offset=-1
        )
