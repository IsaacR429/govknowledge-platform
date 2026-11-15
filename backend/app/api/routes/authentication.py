from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    status,
)

from app.api.dependencies.audit import (
    get_audit_service,
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
from app.application.ports.audit import (
    AuditEvent,
)
from app.application.ports.identity_provider import CurrentIdentity
from app.application.services.audit import (
    AuditPersistenceError,
    AuditService,
)
from app.application.services.authentication import (
    AuthenticationService,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
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
    audit_service: Annotated[
        AuditService,
        Depends(get_audit_service),
    ],
) -> LoginResponse:
    correlation_id = uuid4()

    try:
        result = authentication_service.login(
            username=payload.username,
            password=payload.password,
        )

    except InvalidCredentialsError as exc:
        try:
            audit_service.record(
                AuditEvent(
                    event_type=(
                        AuditEventType.AUTH_LOGIN_FAILED
                    ),
                    outcome=AuditOutcome.FAILED,
                    actor_user_id=None,
                    resource_type="authentication",
                    correlation_id=correlation_id,
                    error_code="INVALID_CREDENTIALS",
                    metadata={
                        "auth_method": "password",
                    },
                )
            )
        except AuditPersistenceError:
            # Authentication is already denied.
            # Audit failure must never turn the
            # denial into a successful login.
            pass

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        ) from exc

    # Successful login is security-sensitive.
    # Do not disclose the token unless its audit
    # event was persisted successfully.
    audit_service.record(
        AuditEvent(
            event_type=(
                AuditEventType.AUTH_LOGIN_SUCCEEDED
            ),
            outcome=AuditOutcome.SUCCEEDED,
            actor_user_id=result.identity.user_id,
            resource_type="authentication",
            correlation_id=correlation_id,
            metadata={
                "auth_method": "password",
            },
        )
    )

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
    identity: Annotated[
        CurrentIdentity,
        Depends(get_current_identity),
    ],
    authentication_service: Annotated[
        AuthenticationService,
        Depends(get_authentication_service),
    ],
    audit_service: Annotated[
        AuditService,
        Depends(get_audit_service),
    ],
) -> Response:
    correlation_id = uuid4()

    authentication_service.logout(
        access_token
    )

    try:
        audit_service.record(
            AuditEvent(
                event_type=(
                    AuditEventType.AUTH_LOGOUT
                ),
                outcome=AuditOutcome.SUCCEEDED,
                actor_user_id=identity.user_id,
                resource_type="authentication",
                correlation_id=correlation_id,
                metadata={
                    "auth_method": "bearer",
                },
            )
        )
    except AuditPersistenceError:
        # Logout has already invalidated the
        # session. Never restore/re-enable it
        # because audit persistence failed.
        pass

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )
