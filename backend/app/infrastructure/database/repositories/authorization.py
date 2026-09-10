from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.application.ports.authorization import (
    AuthorizationRepository,
)
from app.domain.enums.permission import PermissionAction
from app.infrastructure.database.models.documents import (
    Document,
)
from app.infrastructure.database.models.permissions import (
    DocumentPermission,
)


class PostgresAuthorizationRepository(
    AuthorizationRepository
):
    """PostgreSQL-backed P0 document ACL implementation."""

    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def document_ids_for_user(
        self,
        user_id: UUID,
        action: PermissionAction,
        *,
        active_only: bool,
    ) -> frozenset[UUID]:
        statement = (
            select(DocumentPermission.document_id)
            .join(
                Document,
                Document.id
                == DocumentPermission.document_id,
            )
            .where(
                DocumentPermission.user_id == user_id,
                DocumentPermission.action == action.value,
            )
        )

        if active_only:
            statement = statement.where(
                Document.status == "ACTIVE"
            )

        document_ids = self._db.scalars(
            statement
        ).all()

        return frozenset(document_ids)

    def has_document_permission(
        self,
        user_id: UUID,
        document_id: UUID,
        action: PermissionAction,
    ) -> bool:
        statement = select(
            DocumentPermission.id
        ).where(
            DocumentPermission.user_id == user_id,
            DocumentPermission.document_id == document_id,
            DocumentPermission.action == action.value,
        )

        return self._db.scalar(statement) is not None
