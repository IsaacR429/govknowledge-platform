from __future__ import annotations

from uuid import UUID

from sqlalchemy import (
    func,
    select,
)
from sqlalchemy.orm import Session

from app.application.ports.audit_read_repository import (
    AuditEventPage,
    AuditEventRead,
    AuditReadRepository,
)
from app.infrastructure.database.models.audit import (
    AuditEventRecord,
)
from app.infrastructure.database.models.identity import (
    User,
)


class PostgresAuditReadRepository(
    AuditReadRepository
):
    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def list_events(
        self,
        *,
        limit: int,
        offset: int,
        event_type: str | None = None,
        outcome: str | None = None,
        actor_user_id: UUID | None = None,
        document_id: UUID | None = None,
        correlation_id: UUID | None = None,
    ) -> AuditEventPage:

        filters = []

        if event_type is not None:
            filters.append(
                AuditEventRecord.event_type
                == event_type
            )

        if outcome is not None:
            filters.append(
                AuditEventRecord.outcome
                == outcome
            )

        if actor_user_id is not None:
            filters.append(
                AuditEventRecord.actor_user_id
                == actor_user_id
            )

        if document_id is not None:
            filters.append(
                AuditEventRecord.document_id
                == document_id
            )

        if correlation_id is not None:
            filters.append(
                AuditEventRecord.correlation_id
                == correlation_id
            )

        total_statement = (
            select(
                func.count(
                    AuditEventRecord.id
                )
            )
            .where(
                *filters
            )
        )

        total = int(
            self._db.scalar(
                total_statement
            )
            or 0
        )

        statement = (
            select(
                AuditEventRecord,
                User.username,
                User.display_name,
            )
            .outerjoin(
                User,
                User.id
                == AuditEventRecord.actor_user_id,
            )
            .where(
                *filters
            )
            .order_by(
                AuditEventRecord.occurred_at.desc(),
                AuditEventRecord.id.desc(),
            )
            .offset(
                offset
            )
            .limit(
                limit
            )
        )

        rows = (
            self._db.execute(
                statement
            )
            .all()
        )

        items = tuple(
            AuditEventRead(
                event_id=record.id,
                occurred_at=(
                    record.occurred_at
                ),
                event_type=(
                    record.event_type
                ),
                outcome=record.outcome,
                actor_user_id=(
                    record.actor_user_id
                ),
                actor_username=username,
                actor_display_name=(
                    display_name
                ),
                resource_type=(
                    record.resource_type
                ),
                resource_id=(
                    record.resource_id
                ),
                document_id=(
                    record.document_id
                ),
                correlation_id=(
                    record.correlation_id
                ),
                error_code=(
                    record.error_code
                ),
                metadata=dict(
                    record.metadata_json
                    or {}
                ),
            )
            for (
                record,
                username,
                display_name,
            )
            in rows
        )

        return AuditEventPage(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )
