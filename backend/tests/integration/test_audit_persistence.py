from datetime import datetime, timezone
from uuid import uuid4

import pytest

from sqlalchemy import delete, func, select
from sqlalchemy.exc import DataError

from app.application.ports.audit import (
    AuditEvent,
)
from app.application.services.audit import (
    AuditService,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)
from app.infrastructure.database.models.audit import (
    AuditEventRecord,
)
from app.infrastructure.database.repositories.audit import (
    PostgresAuditRepository,
)
from app.infrastructure.database.session import (
    SessionLocal,
)


def cleanup_correlation(
    db,
    correlation_id,
):
    db.rollback()

    db.execute(
        delete(
            AuditEventRecord
        ).where(
            AuditEventRecord.correlation_id
            == correlation_id
        )
    )

    db.commit()


def test_audit_event_persists_complete_record():
    db = SessionLocal()

    correlation_id = uuid4()
    actor_user_id = uuid4()
    document_id = uuid4()
    resource_id = uuid4()

    try:
        service = AuditService(
            PostgresAuditRepository(db)
        )

        event_id = service.record(
            AuditEvent(
                event_type=(
                    AuditEventType
                    .DOCUMENT_READ_GRANTED
                ),
                outcome=(
                    AuditOutcome.SUCCEEDED
                ),
                actor_user_id=actor_user_id,
                resource_type="document",
                resource_id=resource_id,
                document_id=document_id,
                correlation_id=correlation_id,
                metadata={
                    "permission": "READ",
                    "source": (
                        "integration_test"
                    ),
                },
            )
        )

        db.expire_all()

        stored = db.get(
            AuditEventRecord,
            event_id,
        )

        assert stored is not None

        assert (
            stored.event_type
            == "DOCUMENT_READ_GRANTED"
        )

        assert (
            stored.outcome
            == "SUCCEEDED"
        )

        assert (
            stored.actor_user_id
            == actor_user_id
        )

        assert (
            stored.resource_type
            == "document"
        )

        assert (
            stored.resource_id
            == resource_id
        )

        assert (
            stored.document_id
            == document_id
        )

        assert (
            stored.correlation_id
            == correlation_id
        )

        assert stored.error_code is None

        assert stored.metadata_json == {
            "permission": "READ",
            "source": "integration_test",
        }

        assert stored.occurred_at is not None

    finally:
        cleanup_correlation(
            db,
            correlation_id,
        )
        db.close()


def test_explicit_occurred_at_is_preserved():
    db = SessionLocal()

    correlation_id = uuid4()

    occurred_at = datetime(
        2026,
        9,
        11,
        12,
        34,
        56,
        123456,
        tzinfo=timezone.utc,
    )

    try:
        service = AuditService(
            PostgresAuditRepository(db)
        )

        event_id = service.record(
            AuditEvent(
                event_type=(
                    AuditEventType
                    .EVIDENCE_SUFFICIENT
                ),
                outcome=(
                    AuditOutcome.SUCCEEDED
                ),
                correlation_id=(
                    correlation_id
                ),
                occurred_at=occurred_at,
            )
        )

        db.expire_all()

        stored = db.get(
            AuditEventRecord,
            event_id,
        )

        assert stored is not None

        assert (
            stored.occurred_at
            == occurred_at
        )

    finally:
        cleanup_correlation(
            db,
            correlation_id,
        )
        db.close()


def test_repository_rolls_back_failed_insert():
    db = SessionLocal()

    correlation_id = uuid4()

    try:
        repository = (
            PostgresAuditRepository(db)
        )

        # error_code is VARCHAR(120).
        # 121 characters must fail at PostgreSQL,
        # proving repository rollback behavior.
        with pytest.raises(
            DataError
        ):
            repository.append(
                AuditEvent(
                    event_type=(
                        AuditEventType
                        .CITATION_REJECTED
                    ),
                    outcome=(
                        AuditOutcome.FAILED
                    ),
                    correlation_id=(
                        correlation_id
                    ),
                    error_code=(
                        "X" * 121
                    ),
                )
            )

        # Repository must have rolled back the
        # failed transaction so this session
        # remains usable.
        count = db.scalar(
            select(
                func.count(
                    AuditEventRecord.id
                )
            ).where(
                AuditEventRecord.correlation_id
                == correlation_id
            )
        )

        assert count == 0

        # A second valid write through the same
        # session proves rollback recovery.
        valid_id = repository.append(
            AuditEvent(
                event_type=(
                    AuditEventType
                    .CITATION_VALIDATED
                ),
                outcome=(
                    AuditOutcome.SUCCEEDED
                ),
                correlation_id=(
                    correlation_id
                ),
            )
        )

        stored = db.get(
            AuditEventRecord,
            valid_id,
        )

        assert stored is not None

        assert (
            stored.event_type
            == "CITATION_VALIDATED"
        )

    finally:
        cleanup_correlation(
            db,
            correlation_id,
        )
        db.close()
