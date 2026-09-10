from uuid import UUID

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base
from app.infrastructure.database.models.mixins import (
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class DocumentPermission(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    __tablename__ = "document_permissions"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "document_id",
            "action",
            name="uq_user_document_action",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="READ",
    )
