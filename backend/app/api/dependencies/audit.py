from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.application.services.audit import (
    AuditService,
)
from app.infrastructure.database.repositories.audit import (
    PostgresAuditRepository,
)
from app.infrastructure.database.session import (
    get_db,
)


def get_audit_service(
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> AuditService:
    return AuditService(
        PostgresAuditRepository(
            db
        )
    )
