from typing import Annotated

from fastapi import Depends, HTTPException, status

from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)


def require_admin_identity(
    identity: Annotated[
        CurrentIdentity,
        Depends(get_current_identity),
    ],
) -> CurrentIdentity:
    if "ADMIN" not in identity.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role required.",
        )

    return identity
