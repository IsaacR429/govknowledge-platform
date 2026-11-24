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
        *,
        active_document_ids: set | None = None,
        direct_permission_document_ids: set | None = None,
    ) -> None:
        self.allowed_document_ids = frozenset(
            allowed_document_ids
        )
        self.all_active_document_ids = frozenset(
            active_document_ids
            if active_document_ids is not None
            else allowed_document_ids
        )
        self.direct_permission_document_ids = frozenset(
            direct_permission_document_ids
            if direct_permission_document_ids is not None
            else allowed_document_ids
        )

    def active_document_ids(
        self,
    ):
        return self.all_active_document_ids

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
            in self.direct_permission_document_ids
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


def test_admin_retrieval_scope_contains_all_active_documents():
    first_document = uuid4()
    second_document = uuid4()

    repository = FakeAuthorizationRepository(
        set(),
        active_document_ids={
            first_document,
            second_document,
        },
    )

    service = AuthorizationService(
        repository
    )

    admin = make_identity(
        roles=("ADMIN",)
    )

    scope = service.retrieval_scope(
        admin
    )

    assert scope.allows(
        first_document
    )
    assert scope.allows(
        second_document
    )
    assert scope.document_ids == frozenset(
        {
            first_document,
            second_document,
        }
    )


def test_admin_can_read_active_document_without_acl():
    active_document = uuid4()

    repository = FakeAuthorizationRepository(
        set(),
        active_document_ids={
            active_document
        },
        direct_permission_document_ids=set(),
    )

    service = AuthorizationService(
        repository
    )

    admin = make_identity(
        roles=("ADMIN",)
    )

    service.require_document_read(
        admin,
        active_document,
    )


def test_employee_does_not_inherit_admin_document_scope():
    active_document = uuid4()

    repository = FakeAuthorizationRepository(
        set(),
        active_document_ids={
            active_document
        },
    )

    service = AuthorizationService(
        repository
    )

    employee = make_identity(
        roles=("EMPLOYEE",)
    )

    scope = service.retrieval_scope(
        employee
    )

    assert not scope.allows(
        active_document
    )
    assert scope.document_ids == frozenset()


def test_archived_document_with_stale_read_permission_is_denied():
    archived_document = uuid4()
    repository = FakeAuthorizationRepository(
        set(),
        direct_permission_document_ids={
            archived_document
        },
    )
    service = AuthorizationService(
        repository
    )
    identity = make_identity()

    with pytest.raises(
        AuthorizationDeniedError
    ):
        service.require_document_read(
            identity,
            archived_document,
        )


def test_scope_permission_recheck_semantics():
    employee_document = uuid4()
    admin_document = uuid4()

    repository = FakeAuthorizationRepository(
        {
            employee_document,
        },
        active_document_ids={
            employee_document,
            admin_document,
        },
    )

    service = AuthorizationService(
        repository
    )

    employee_scope = (
        service.retrieval_scope(
            make_identity(
                roles=("EMPLOYEE",)
            )
        )
    )

    admin_scope = (
        service.retrieval_scope(
            make_identity(
                roles=("ADMIN",)
            )
        )
    )

    assert (
        employee_scope
        .requires_document_permission
        is True
    )

    assert (
        admin_scope
        .requires_document_permission
        is False
    )

    assert (
        employee_scope.document_ids
        == frozenset(
            {
                employee_document,
            }
        )
    )

    assert (
        admin_scope.document_ids
        == frozenset(
            {
                employee_document,
                admin_document,
            }
        )
    )


def test_manual_scope_defaults_to_permission_recheck():
    document_id = uuid4()

    scope = AuthorizationScope(
        user_id=uuid4(),
        action=PermissionAction.READ,
        document_ids=frozenset(
            {
                document_id,
            }
        ),
    )

    assert (
        scope
        .requires_document_permission
        is True
    )
