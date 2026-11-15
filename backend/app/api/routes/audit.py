from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    Query,
)

from app.api.dependencies.audit_read import (
    get_audit_read_service,
    require_audit_reader_identity,
)
from app.api.schemas.audit import (
    AuditEventListResponse,
    AuditEventResponse,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.audit_read import (
    AuditReadService,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)


router = APIRouter(
    prefix="/audit",
    tags=[
        "audit",
    ],
)


@router.get(
    "/events",
    response_model=(
        AuditEventListResponse
    ),
)
def list_audit_events(
    identity: Annotated[
        CurrentIdentity,
        Depends(
            require_audit_reader_identity
        ),
    ],
    service: Annotated[
        AuditReadService,
        Depends(
            get_audit_read_service
        ),
    ],
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=100,
        ),
    ] = 50,
    offset: Annotated[
        int,
        Query(
            ge=0,
        ),
    ] = 0,
    event_type: (
        AuditEventType | None
    ) = None,
    outcome: (
        AuditOutcome | None
    ) = None,
    actor_user_id: UUID | None = None,
    document_id: UUID | None = None,
    correlation_id: UUID | None = None,
) -> AuditEventListResponse:

    # Explicitly resolve identity before
    # audit data can be disclosed.
    _ = identity

    page = service.list_events(
        limit=limit,
        offset=offset,
        event_type=event_type,
        outcome=outcome,
        actor_user_id=actor_user_id,
        document_id=document_id,
        correlation_id=correlation_id,
    )

    items = [
        AuditEventResponse(
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
            metadata=dict(
                item.metadata
            ),
        )
        for item in page.items
    ]

    consumed = (
        page.offset
        + len(
            page.items
        )
    )

    has_more = (
        consumed
        < page.total
    )

    return AuditEventListResponse(
        items=items,
        total=page.total,
        limit=page.limit,
        offset=page.offset,
        has_more=has_more,
        next_offset=(
            consumed
            if has_more
            else None
        ),
    )
