from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from sqlalchemy.orm import Session

from app.application.ports.identity_provider import CurrentIdentity
from app.application.services.authentication import (
    AuthenticationService,
)
from app.domain.exceptions.authentication import (
    InvalidSessionError,
)
from app.infrastructure.database.session import get_db
from app.infrastructure.identity.local_identity_provider import (
    LocalIdentityProvider,
)


bearer_scheme = HTTPBearer(
    auto_error=False,
)


def get_authentication_service(
    db: Annotated[Session, Depends(get_db)],
) -> AuthenticationService:
    provider = LocalIdentityProvider(db)

    return AuthenticationService(provider)


def get_access_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    return credentials.credentials


def get_current_identity(
    access_token: Annotated[
        str,
        Depends(get_access_token),
    ],
    authentication_service: Annotated[
        AuthenticationService,
        Depends(get_authentication_service),
    ],
) -> CurrentIdentity:
    try:
        return authentication_service.current_identity(
            access_token
        )

    except InvalidSessionError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        ) from exc
