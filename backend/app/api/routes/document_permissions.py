"""Administrator-managed document READ permissions."""
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, selectinload

from app.api.dependencies.administration import require_admin_identity
from app.api.schemas.document_permissions import (
    DocumentReadPermissionListResponse,
    DocumentReadPermissionUserResponse,
)
from app.application.ports.identity_provider import CurrentIdentity
from app.domain.enums.audit import AuditEventType, AuditOutcome
from app.infrastructure.database.models.audit import AuditEventRecord
from app.infrastructure.database.models.documents import Document
from app.infrastructure.database.models.identity import User
from app.infrastructure.database.models.permissions import DocumentPermission
from app.infrastructure.database.session import get_db

router = APIRouter(prefix="/documents", tags=["document-access"])


def _targets(db: Session, document_id: UUID, user_id: UUID):
    document = db.get(Document, document_id)
    user = db.get(User, user_id)
    if document is None or user is None:
        raise HTTPException(404, "Document or user not found.")
    return document, user


def _commit_with_audit(
    db: Session, *, event_type: AuditEventType, actor_id: UUID,
    document_id: UUID, user_id: UUID, permission_id: UUID,
) -> None:
    db.add(AuditEventRecord(
        event_type=event_type.value,
        outcome=AuditOutcome.SUCCEEDED.value,
        actor_user_id=actor_id,
        resource_type="DOCUMENT_PERMISSION",
        resource_id=permission_id,
        document_id=document_id,
        metadata_json={"target_user_id": str(user_id), "action": "READ"},
    ))
    try:
        db.commit()  # Permission change and audit event: one transaction.
    except Exception:
        db.rollback()
        raise


@router.get(
    "/{document_id}/permissions/read",
    response_model=DocumentReadPermissionListResponse,
)
def list_read_permissions(
    document_id: UUID,
    identity: Annotated[CurrentIdentity, Depends(require_admin_identity)],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentReadPermissionListResponse:
    document = db.get(
        Document,
        document_id,
    )
    if document is None:
        raise HTTPException(
            404,
            "Document not found.",
        )
    users = (
        db.scalars(
            select(User)
            .options(
                selectinload(
                    User.roles,
                )
            )
            .order_by(
                User.username.asc(),
            )
        )
        .unique()
        .all()
    )
    permitted_user_ids = set(
        db.scalars(
            select(
                DocumentPermission.user_id
            ).where(
                DocumentPermission.document_id
                == document_id,
                DocumentPermission.action
                == "READ",
            )
        ).all()
    )
    return DocumentReadPermissionListResponse(
        document_id=document_id,
        items=[
            DocumentReadPermissionUserResponse(
                user_id=user.id,
                username=user.username,
                display_name=user.display_name,
                department=user.department,
                is_active=user.is_active,
                roles=sorted(
                    role.name
                    for role in user.roles
                ),
                has_read=(
                    user.id
                    in permitted_user_ids
                ),
            )
            for user in users
        ],
    )


@router.post("/{document_id}/permissions/read/{user_id}")
def grant_read(
    document_id: UUID,
    user_id: UUID,
    identity: Annotated[CurrentIdentity, Depends(require_admin_identity)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    document, user = _targets(db, document_id, user_id)
    if document.status != "ACTIVE" or not user.is_active:
        raise HTTPException(409, "Document or user is inactive.")

    permission_id = db.scalar(
        insert(DocumentPermission)
        .values(id=uuid4(), user_id=user_id, document_id=document_id, action="READ")
        .on_conflict_do_nothing(constraint="uq_user_document_action")
        .returning(DocumentPermission.id)
    )
    if permission_id is None:
        db.rollback()  # Already granted; do not create a duplicate.
    else:
        _commit_with_audit(
            db, event_type=AuditEventType.DOCUMENT_READ_GRANTED,
            actor_id=identity.user_id, document_id=document_id,
            user_id=user_id, permission_id=permission_id,
        )

    return {
        "document_id": document_id, "user_id": user_id,
        "permission": "READ", "changed": permission_id is not None,
    }


@router.delete("/{document_id}/permissions/read/{user_id}")
def revoke_read(
    document_id: UUID,
    user_id: UUID,
    identity: Annotated[CurrentIdentity, Depends(require_admin_identity)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    _targets(db, document_id, user_id)

    permission_id = db.scalar(
        delete(DocumentPermission)
        .where(
            DocumentPermission.document_id == document_id,
            DocumentPermission.user_id == user_id,
            DocumentPermission.action == "READ",
        )
        .returning(DocumentPermission.id)
    )
    if permission_id is None:
        db.rollback()  # Already revoked.
    else:
        _commit_with_audit(
            db, event_type=AuditEventType.DOCUMENT_READ_REVOKED,
            actor_id=identity.user_id, document_id=document_id,
            user_id=user_id, permission_id=permission_id,
        )

    return {
        "document_id": document_id, "user_id": user_id,
        "permission": "READ", "changed": permission_id is not None,
    }
