from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.dependencies.audit_read import (
    require_audit_reader_identity,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)


def identity(
    *roles,
):
    return CurrentIdentity(
        user_id=uuid4(),
        username="user",
        display_name="User",
        email=None,
        department="QA",
        language_preference="fr",
        roles=tuple(
            roles
        ),
    )


def test_auditor_can_read_audit():
    user = identity(
        "AUDITOR"
    )

    assert (
        require_audit_reader_identity(
            user
        )
        is user
    )


def test_admin_can_read_audit():
    user = identity(
        "ADMIN"
    )

    assert (
        require_audit_reader_identity(
            user
        )
        is user
    )


def test_employee_cannot_read_audit():

    with pytest.raises(
        HTTPException
    ) as exc_info:

        require_audit_reader_identity(
            identity(
                "EMPLOYEE"
            )
        )

    assert (
        exc_info.value.status_code
        == 403
    )


def test_user_without_roles_cannot_read_audit():

    with pytest.raises(
        HTTPException
    ) as exc_info:

        require_audit_reader_identity(
            identity()
        )

    assert (
        exc_info.value.status_code
        == 403
    )
