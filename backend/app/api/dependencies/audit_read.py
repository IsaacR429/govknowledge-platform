from __future__ import annotations

from typing import Annotated

from fastapi import (
    Depends,
    HTTPException,
    status,
)
from sqlalchemy.orm import Session

from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.audit_read import (
    AuditReadService,
)
from app.domain.enums.role import (
    RoleName,
)
from app.infrastructure.database.repositories.audit_read import (
    PostgresAuditReadRepository,
)
from app.infrastructure.database.session import (
    get_db,
)


_ALLOWED_AUDIT_ROLES = frozenset(
    {
        RoleName.ADMIN.value,
        RoleName.AUDITOR.value,
    }
)


def require_audit_reader_identity(
    identity: Annotated[
        CurrentIdentity,
        Depends(
            get_current_identity
        ),
    ],
) -> CurrentIdentity:

    if not (
        set(
            identity.roles
        )
        & _ALLOWED_AUDIT_ROLES
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_403_FORBIDDEN
            ),
            detail=(
                "Auditor or administrator "
                "role required."
            ),
        )

    return identity


def get_audit_read_service(
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> AuditReadService:

    return AuditReadService(
        PostgresAuditReadRepository(
            db
        )
    )
