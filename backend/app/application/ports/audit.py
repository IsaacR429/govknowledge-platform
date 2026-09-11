from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol
from uuid import UUID

from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)


@dataclass(frozen=True)
class AuditEvent:
    event_type: AuditEventType
    outcome: AuditOutcome

    actor_user_id: UUID | None = None

    resource_type: str | None = None
    resource_id: UUID | None = None
    document_id: UUID | None = None

    correlation_id: UUID | None = None
    error_code: str | None = None

    metadata: Mapping[
        str,
        object,
    ] | None = None

    occurred_at: datetime | None = None


class AuditRepository(Protocol):
    def append(
        self,
        event: AuditEvent,
    ) -> UUID:
        ...
