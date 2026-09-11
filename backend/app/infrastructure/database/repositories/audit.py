from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.application.ports.audit import (
    AuditEvent,
    AuditRepository,
)
from app.infrastructure.database.models.audit import (
    AuditEventRecord,
)


class PostgresAuditRepository(
    AuditRepository
):
    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def append(
        self,
        event: AuditEvent,
    ) -> UUID:
        values = {
            "event_type": (
                event.event_type.value
            ),
            "outcome": (
                event.outcome.value
            ),
            "actor_user_id": (
                event.actor_user_id
            ),
            "resource_type": (
                event.resource_type
            ),
            "resource_id": (
                event.resource_id
            ),
            "document_id": (
                event.document_id
            ),
            "correlation_id": (
                event.correlation_id
            ),
            "error_code": (
                event.error_code
            ),
            "metadata_json": dict(
                event.metadata
                or {}
            ),
        }

        if event.occurred_at is not None:
            values["occurred_at"] = (
                event.occurred_at
            )

        record = AuditEventRecord(
            **values
        )

        self._db.add(record)

        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        return record.id
