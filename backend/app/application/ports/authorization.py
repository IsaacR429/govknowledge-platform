from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.domain.enums.permission import PermissionAction


@dataclass(frozen=True)
class AuthorizationScope:
    """Immutable document scope passed downstream.

    Retrieval is permitted to operate only inside document_ids.
    """

    user_id: UUID
    action: PermissionAction
    document_ids: frozenset[UUID]

    def allows(
        self,
        document_id: UUID,
    ) -> bool:
        return document_id in self.document_ids


class AuthorizationRepository(Protocol):
    def active_document_ids(
        self,
    ) -> frozenset[UUID]:
        ...

    def document_ids_for_user(
        self,
        user_id: UUID,
        action: PermissionAction,
        *,
        active_only: bool,
    ) -> frozenset[UUID]:
        ...

    def has_document_permission(
        self,
        user_id: UUID,
        document_id: UUID,
        action: PermissionAction,
    ) -> bool:
        ...
