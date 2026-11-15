from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.api.dependencies.audit import (
    get_audit_service,
)
from app.application.services.audit import (
    AuditService,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.authorized_source_page import (
    AuthorizedSourcePageService,
)
from app.api.dependencies.authorization import (
    get_authorization_service,
)
from app.infrastructure.database.repositories.source_pages import (
    PostgresSourcePageRepository,
)
from app.infrastructure.database.session import (
    get_db,
)




def get_authorized_source_page_service(
    db: Annotated[
        Session,
        Depends(get_db),
    ],
    authorization: Annotated[
        AuthorizationService,
        Depends(
            get_authorization_service
        ),
    ],
    audit: Annotated[
        AuditService,
        Depends(get_audit_service),
    ],
) -> AuthorizedSourcePageService:
    return AuthorizedSourcePageService(
        repository=(
            PostgresSourcePageRepository(
                db
            )
        ),
        authorization=authorization,
        audit=audit,
    )


from pathlib import Path

from app.core.config import get_settings
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)


def get_source_page_storage() -> LocalFileStorage:
    return LocalFileStorage(
        Path(get_settings().document_storage_root)
    )
