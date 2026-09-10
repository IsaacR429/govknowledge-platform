from uuid import uuid4

import pytest

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.domain.enums.permission import PermissionAction
from app.domain.exceptions.authorization import (
    AuthorizationDeniedError,
)


class FakeAuthorizationRepository:
    def __init__(
        self,
        allowed_document_ids: set,
    ) -> None:
        self.allowed_document_ids = frozenset(
            allowed_document_ids
        )

    def document_ids_for_user(
        self,
        user_id,
        action,
        *,
        active_only,
    ):
        assert action == PermissionAction.READ
        assert active_only is True

        return self.allowed_document_ids

    def has_document_permission(
        self,
        user_id,
        document_id,
        action,
    ):
        return (
            action == PermissionAction.READ
            and document_id
            in self.allowed_document_ids
        )


def make_identity(
    *,
    roles: tuple[str, ...] = ("EMPLOYEE",),
) -> CurrentIdentity:
    return CurrentIdentity(
        user_id=uuid4(),
        username="test-user",
        display_name="Test User",
        email=None,
        department="Test",
        language_preference="fr",
        roles=roles,
    )


def test_retrieval_scope_contains_only_authorized_documents():
    allowed_document = uuid4()
    unauthorized_document = uuid4()

    repository = FakeAuthorizationRepository(
        {allowed_document}
    )

    service = AuthorizationService(repository)
    identity = make_identity()

    scope = service.retrieval_scope(identity)

    assert isinstance(scope, AuthorizationScope)
    assert scope.allows(allowed_document)
    assert not scope.allows(
        unauthorized_document
    )


def test_empty_permissions_produce_empty_retrieval_scope():
    repository = FakeAuthorizationRepository(set())

    service = AuthorizationService(repository)
    identity = make_identity()

    scope = service.retrieval_scope(identity)

    assert scope.document_ids == frozenset()


def test_unauthorized_document_is_denied():
    allowed_document = uuid4()
    unauthorized_document = uuid4()

    repository = FakeAuthorizationRepository(
        {allowed_document}
    )

    service = AuthorizationService(repository)
    identity = make_identity()

    with pytest.raises(
        AuthorizationDeniedError
    ):
        service.require_document_read(
            identity,
            unauthorized_document,
        )


def test_admin_does_not_bypass_document_acl():
    unauthorized_document = uuid4()

    repository = FakeAuthorizationRepository(set())

    service = AuthorizationService(repository)

    admin = make_identity(
        roles=("ADMIN",)
    )

    scope = service.retrieval_scope(admin)

    assert not scope.allows(
        unauthorized_document
    )
    assert scope.document_ids == frozenset()
