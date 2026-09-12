from __future__ import annotations

from uuid import uuid4

import pytest

from app.application.ports.audit import (
    AuditEvent,
)
from app.application.services.audit import (
    AuditService,
    UnsafeAuditMetadataError,
    sanitize_audit_metadata,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)


class FakeAuditRepository:
    def __init__(self):
        self.events = []
        self.event_id = uuid4()

    def append(
        self,
        event,
    ):
        self.events.append(event)
        return self.event_id


def test_records_sanitized_audit_event():
    repository = (
        FakeAuditRepository()
    )

    service = AuditService(
        repository
    )

    user_id = uuid4()
    document_id = uuid4()

    event_id = service.record(
        AuditEvent(
            event_type=(
                AuditEventType
                .DOCUMENT_READ_GRANTED
            ),
            outcome=(
                AuditOutcome.SUCCEEDED
            ),
            actor_user_id=user_id,
            resource_type="document",
            resource_id=document_id,
            document_id=document_id,
            metadata={
                "permission": "READ",
                "source": "api",
            },
        )
    )

    assert (
        event_id
        == repository.event_id
    )

    assert len(
        repository.events
    ) == 1

    stored = repository.events[0]

    assert (
        stored.actor_user_id
        == user_id
    )

    assert stored.metadata == {
        "permission": "READ",
        "source": "api",
    }


@pytest.mark.parametrize(
    "key",
    (
        "password",
        "access_token",
        "Authorization",
        "raw-prompt",
        "document text",
        "storage_path",
        "traceback",
    ),
)
def test_sensitive_metadata_is_rejected(
    key,
):
    with pytest.raises(
        UnsafeAuditMetadataError,
        match="Unsafe audit metadata",
    ):
        sanitize_audit_metadata(
            {
                key: "SECRET",
            }
        )


def test_none_metadata_becomes_empty():
    assert (
        sanitize_audit_metadata(
            None
        )
        == {}
    )


def test_audit_repository_not_called_for_unsafe_event():
    repository = (
        FakeAuditRepository()
    )

    service = AuditService(
        repository
    )

    with pytest.raises(
        UnsafeAuditMetadataError
    ):
        service.record(
            AuditEvent(
                event_type=(
                    AuditEventType
                    .AUTH_LOGIN_FAILED
                ),
                outcome=(
                    AuditOutcome.FAILED
                ),
                metadata={
                    "password": "secret"
                },
            )
        )

    assert repository.events == []



def test_repository_failure_is_wrapped():
    class FailingRepository:
        def append(
            self,
            event,
        ):
            raise RuntimeError(
                "database unavailable"
            )

    from app.application.services.audit import (
        AuditPersistenceError,
    )

    service = AuditService(
        FailingRepository()
    )

    with pytest.raises(
        AuditPersistenceError,
        match="could not be persisted",
    ):
        service.record(
            AuditEvent(
                event_type=(
                    AuditEventType
                    .SOURCE_PAGE_VIEWED
                ),
                outcome=(
                    AuditOutcome.SUCCEEDED
                ),
            )
        )



@pytest.mark.parametrize(
    "key",
    (
        "query",
        "question",
        "claim",
        "claim_text",
        "answer",
        "evidence",
        "content",
        "extracted_text",
    ),
)
def test_qa_text_metadata_is_rejected(
    key,
):
    with pytest.raises(
        UnsafeAuditMetadataError
    ):
        sanitize_audit_metadata(
            {
                key: "Sensitive user text",
            }
        )


def test_nested_sensitive_metadata_is_rejected():
    with pytest.raises(
        UnsafeAuditMetadataError,
        match="Unsafe audit metadata",
    ):
        sanitize_audit_metadata(
            {
                "request": {
                    "prompt": "SECRET",
                }
            }
        )


def test_sensitive_metadata_inside_list_is_rejected():
    with pytest.raises(
        UnsafeAuditMetadataError,
        match="Unsafe audit metadata",
    ):
        sanitize_audit_metadata(
            {
                "operations": [
                    {
                        "access-token": "SECRET",
                    }
                ]
            }
        )


def test_safe_nested_metadata_is_preserved():
    metadata = {
        "request": {
            "method": "GET",
            "status": 200,
        },
        "documents": [
            {
                "document_id": "abc",
                "page_number": 4,
            }
        ],
    }

    assert (
        sanitize_audit_metadata(
            metadata
        )
        == metadata
    )


@pytest.mark.parametrize(
    "key",
    (
        "api_key",
        "client-secret",
        "private key",
        "secret",
    ),
)
def test_nested_provider_credentials_are_rejected(
    key,
):
    with pytest.raises(
        UnsafeAuditMetadataError,
        match="Unsafe audit metadata",
    ):
        sanitize_audit_metadata(
            {
                "provider": {
                    key: "SECRET",
                }
            }
        )
