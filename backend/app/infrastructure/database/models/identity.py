from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, String, Table, Column
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base
from app.infrastructure.database.models.mixins import (
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


user_roles = Table(
    "user_roles",
    Base.metadata,
    Column(
        "user_id",
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "role_id",
        PGUUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(
        String(150),
        unique=True,
        nullable=False,
        index=True,
    )
    display_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    email: Mapped[str | None] = mapped_column(
        String(320),
        unique=True,
        nullable=True,
    )
    department: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )
    language_preference: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="fr",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    roles: Mapped[list["Role"]] = relationship(
        secondary=user_roles,
        back_populates="users",
    )


class Role(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    users: Mapped[list[User]] = relationship(
        secondary=user_roles,
        back_populates="roles",
    )
