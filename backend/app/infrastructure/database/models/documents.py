from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base
from app.infrastructure.database.models.mixins import (
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"

    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    document_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    issuing_organization: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
    )
    jurisdiction: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    language: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )
    publication_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )
    source_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="ACTIVE",
        index=True,
    )
    confidentiality_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PUBLIC",
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentVersion(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "version_number",
            name="uq_document_version",
        ),
        CheckConstraint(
            "version_number >= 1",
            name="ck_document_version_positive",
        ),
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    storage_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    file_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    processing_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="PENDING",
    )
    effective_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )
    superseded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    document: Mapped[Document] = relationship(
        back_populates="versions",
    )
    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document_version",
        cascade="all, delete-orphan",
    )


class DocumentPage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "page_number",
            name="uq_document_version_page",
        ),
        CheckConstraint(
            "page_number >= 1",
            name="ck_document_page_positive",
        ),
    )

    document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "document_versions.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    extracted_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    document_version: Mapped[DocumentVersion] = relationship(
        back_populates="pages",
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="page",
    )


class DocumentChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "chunk_index",
            name="uq_document_version_chunk_index",
        ),
        CheckConstraint(
            "chunk_index >= 0",
            name="ck_document_chunk_index_nonnegative",
        ),
        CheckConstraint(
            "page_start >= 1",
            name="ck_document_chunk_page_start_positive",
        ),
        CheckConstraint(
            "page_end >= page_start",
            name="ck_document_chunk_page_range",
        ),
        CheckConstraint(
            "page_start = page_end",
            name="ck_document_chunk_single_page_p0",
        ),
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"),
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

    page_start: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    page_end: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    section_title: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    language: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    page: Mapped[DocumentPage] = relationship(
        back_populates="chunks",
    )
