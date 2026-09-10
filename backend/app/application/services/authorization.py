from uuid import UUID

from app.application.ports.authorization import (
    AuthorizationRepository,
    AuthorizationScope,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.domain.enums.permission import PermissionAction
from app.domain.exceptions.authorization import (
    AuthorizationDeniedError,
)


class AuthorizationService:
    """GovKnowledge authorization policy boundary."""

    def __init__(
        self,
        repository: AuthorizationRepository,
    ) -> None:
        self._repository = repository

    def retrieval_scope(
        self,
        identity: CurrentIdentity,
    ) -> AuthorizationScope:
        document_ids = (
            self._repository.document_ids_for_user(
                user_id=identity.user_id,
                action=PermissionAction.READ,
                active_only=True,
            )
        )

        return AuthorizationScope(
            user_id=identity.user_id,
            action=PermissionAction.READ,
            document_ids=document_ids,
        )

    def require_document_read(
        self,
        identity: CurrentIdentity,
        document_id: UUID,
    ) -> None:
        permitted = (
            self._repository.has_document_permission(
                user_id=identity.user_id,
                document_id=document_id,
                action=PermissionAction.READ,
            )
        )

        if not permitted:
            raise AuthorizationDeniedError(
                document_id=document_id
            )
