from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.authorized_source_page import (
    AuthorizedSourcePageService,
)
from app.infrastructure.database.repositories.authorization import (
    PostgresAuthorizationRepository,
)
from app.infrastructure.database.repositories.source_pages import (
    PostgresSourcePageRepository,
)
from app.infrastructure.database.session import (
    get_db,
)


def get_authorization_service(
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> AuthorizationService:
    return AuthorizationService(
        PostgresAuthorizationRepository(
            db
        )
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
) -> AuthorizedSourcePageService:
    return AuthorizedSourcePageService(
        repository=(
            PostgresSourcePageRepository(
                db
            )
        ),
        authorization=authorization,
    )
