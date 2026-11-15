from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class AuditEventResponse(
    BaseModel
):
    event_id: UUID
    occurred_at: datetime

    event_type: str
    outcome: str

    actor_user_id: UUID | None
    actor_username: str | None
    actor_display_name: str | None

    resource_type: str | None
    resource_id: UUID | None
    document_id: UUID | None

    correlation_id: UUID | None
    error_code: str | None

    metadata: dict[
        str,
        Any,
    ]


class AuditEventListResponse(
    BaseModel
):
    items: list[
        AuditEventResponse
    ]

    total: int
    limit: int
    offset: int

    has_more: bool
    next_offset: int | None
