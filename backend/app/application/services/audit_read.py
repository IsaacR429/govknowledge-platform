from __future__ import annotations

from uuid import UUID

from app.application.ports.audit_read_repository import (
    AuditEventPage,
    AuditEventRead,
    AuditReadRepository,
)
from app.application.services.audit import (
    UnsafeAuditMetadataError,
    sanitize_audit_metadata,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)


class AuditReadService:
    """
    Read-only audit-query application service.

    It never mutates audit history.

    Metadata is sanitized again on disclosure as
    defense in depth. Historical malformed metadata
    is redacted instead of disclosed.
    """

    def __init__(
        self,
        repository: AuditReadRepository,
    ) -> None:
        self._repository = repository

    def list_events(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        event_type: AuditEventType | None = None,
        outcome: AuditOutcome | None = None,
        actor_user_id: UUID | None = None,
        document_id: UUID | None = None,
        correlation_id: UUID | None = None,
    ) -> AuditEventPage:

        if not 1 <= limit <= 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        if offset < 0:
            raise ValueError(
                "offset cannot be negative."
            )

        page = self._repository.list_events(
            limit=limit,
            offset=offset,
            event_type=(
                event_type.value
                if event_type is not None
                else None
            ),
            outcome=(
                outcome.value
                if outcome is not None
                else None
            ),
            actor_user_id=actor_user_id,
            document_id=document_id,
            correlation_id=correlation_id,
        )

        safe_items: list[
            AuditEventRead
        ] = []

        for item in page.items:

            try:
                safe_metadata = (
                    sanitize_audit_metadata(
                        item.metadata
                    )
                )

            except UnsafeAuditMetadataError:
                # Historical or externally inserted
                # unsafe rows must not leak content.
                safe_metadata = {
                    "redacted": True,
                }

            safe_items.append(
                AuditEventRead(
                    event_id=item.event_id,
                    occurred_at=item.occurred_at,
                    event_type=item.event_type,
                    outcome=item.outcome,
                    actor_user_id=(
                        item.actor_user_id
                    ),
                    actor_username=(
                        item.actor_username
                    ),
                    actor_display_name=(
                        item.actor_display_name
                    ),
                    resource_type=(
                        item.resource_type
                    ),
                    resource_id=(
                        item.resource_id
                    ),
                    document_id=(
                        item.document_id
                    ),
                    correlation_id=(
                        item.correlation_id
                    ),
                    error_code=(
                        item.error_code
                    ),
                    metadata=safe_metadata,
                )
            )

        return AuditEventPage(
            items=tuple(
                safe_items
            ),
            total=page.total,
            limit=page.limit,
            offset=page.offset,
        )
