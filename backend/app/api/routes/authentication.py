from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    status,
)

from app.api.dependencies.authentication import (
    get_access_token,
    get_authentication_service,
    get_current_identity,
)
from app.api.schemas.authentication import (
    IdentityResponse,
    LoginRequest,
    LoginResponse,
)
from app.application.ports.identity_provider import CurrentIdentity
from app.application.services.authentication import (
    AuthenticationService,
)
from app.domain.exceptions.authentication import (
    InvalidCredentialsError,
)


router = APIRouter(
    tags=["authentication"],
)


def to_identity_response(
    identity: CurrentIdentity,
) -> IdentityResponse:
    return IdentityResponse(
        user_id=identity.user_id,
        username=identity.username,
        display_name=identity.display_name,
        email=identity.email,
        department=identity.department,
        language_preference=identity.language_preference,
        roles=list(identity.roles),
    )


@router.post(
    "/auth/login",
    response_model=LoginResponse,
)
def login(
    payload: LoginRequest,
    authentication_service: Annotated[
        AuthenticationService,
        Depends(get_authentication_service),
    ],
) -> LoginResponse:
    try:
        result = authentication_service.login(
            username=payload.username,
            password=payload.password,
        )

    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        ) from exc

    return LoginResponse(
        access_token=result.access_token,
        expires_at=result.expires_at,
        user=to_identity_response(
            result.identity
        ),
    )


@router.get(
    "/me",
    response_model=IdentityResponse,
)
def me(
    identity: Annotated[
        CurrentIdentity,
        Depends(get_current_identity),
    ],
) -> IdentityResponse:
    return to_identity_response(identity)


@router.post(
    "/auth/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
def logout(
    access_token: Annotated[
        str,
        Depends(get_access_token),
    ],
    authentication_service: Annotated[
        AuthenticationService,
        Depends(get_authentication_service),
    ],
) -> Response:
    authentication_service.logout(
        access_token
    )

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )
