from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.application.services.authorization import (
    AuthorizationService,
)
from app.infrastructure.database.repositories.authorization import (
    PostgresAuthorizationRepository,
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
