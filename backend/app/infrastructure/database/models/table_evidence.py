from __future__ import annotations

from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from app.infrastructure.database.base import Base
from app.infrastructure.database.models.mixins import (
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class TableEvidenceRecord(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    __tablename__ = "table_evidence"

    __table_args__ = (
        UniqueConstraint(
            "evidence_id",
            name="uq_table_evidence_id",
        ),
        CheckConstraint(
            "page_number >= 1",
            name="ck_table_evidence_page_positive",
        ),
        CheckConstraint(
            "row_index >= 0",
            name="ck_table_evidence_row_nonnegative",
        ),
        Index(
            "ix_table_evidence_logical_document",
            "logical_key",
            "document_id",
        ),
    )

    evidence_id: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "documents.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "document_versions.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    page_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "document_pages.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    row_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )

    logical_key: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        index=True,
    )

    section_key: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        index=True,
    )

    language: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )

    source_table_refs_json: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    column_names_json: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    cells_json: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    search_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
