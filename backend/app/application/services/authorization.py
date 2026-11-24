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

    @staticmethod
    def _is_admin(
        identity: CurrentIdentity,
    ) -> bool:
        return any(
            str(role).strip().upper() == "ADMIN"
            for role in identity.roles
        )

    def retrieval_scope(
        self,
        identity: CurrentIdentity,
    ) -> AuthorizationScope:
        is_admin = self._is_admin(
            identity
        )

        if is_admin:
            document_ids = (
                self._repository.active_document_ids()
            )
        else:
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
            requires_document_permission=(
                not is_admin
            ),
        )

    def require_document_read(
        self,
        identity: CurrentIdentity,
        document_id: UUID,
    ) -> None:
        scope = self.retrieval_scope(
            identity
        )

        if not scope.allows(
            document_id
        ):
            raise AuthorizationDeniedError(
                document_id=document_id
            )
