from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol
from uuid import UUID


@dataclass(frozen=True)
class AuditEventRead:
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

    metadata: Mapping[
        str,
        object,
    ]


@dataclass(frozen=True)
class AuditEventPage:
    items: tuple[
        AuditEventRead,
        ...,
    ]

    total: int
    limit: int
    offset: int


class AuditReadRepository(
    Protocol
):
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
        ...
