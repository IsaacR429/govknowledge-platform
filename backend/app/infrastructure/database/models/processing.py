from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base
from app.infrastructure.database.models.mixins import (
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


class DocumentProcessingJob(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    __tablename__ = "document_processing_jobs"

    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "attempt_number",
            name="uq_processing_job_version_attempt",
        ),
        CheckConstraint(
            "attempt_number >= 1",
            name="ck_processing_job_attempt_positive",
        ),
        CheckConstraint(
            "status IN "
            "('PENDING', 'PROCESSING', 'SUCCEEDED', 'FAILED')",
            name="ck_processing_job_status",
        ),
        CheckConstraint(
            "parsing_status IS NULL "
            "OR parsing_status IN ('SUCCESS', 'FAILED')",
            name="ck_processing_job_parsing_status",
        ),
        CheckConstraint(
            "failure_class IS NULL OR failure_class IN ("
            "'MALFORMED_PAGE_TREE', "
            "'UNRESOLVED_PAGE_GEOMETRY', "
            "'CORRUPT_XREF', "
            "'CORRUPT_STREAM', "
            "'TIMEOUT', "
            "'OOM', "
            "'NATIVE_CRASH', "
            "'NORMALIZATION_FAILED', "
            "'POST_NORMALIZATION_MISMATCH', "
            "'UNKNOWN'"
            ")",
            name="ck_processing_job_failure_class",
        ),
        CheckConstraint(
            "("
            "parsing_status IS NULL "
            "AND failure_class IS NULL"
            ") OR ("
            "parsing_status = 'SUCCESS' "
            "AND failure_class IS NULL"
            ") OR ("
            "parsing_status = 'FAILED' "
            "AND failure_class IS NOT NULL"
            ")",
            name="ck_processing_job_parser_outcome",
        ),
        CheckConstraint(
            "indexing_status IS NULL OR indexing_status IN "
            "('PENDING', 'INDEXING', 'SUCCEEDED', 'FAILED', 'NO_CHUNKS')",
            name="ck_processing_job_indexing_status",
        ),
        CheckConstraint(
            "indexing_attempt_number >= 0",
            name="ck_processing_job_indexing_attempt_nonnegative",
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

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="PENDING",
        index=True,
    )

    attempt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    parsing_status: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    failure_class: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    parser_lineage_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Separate from parsing: a parsed version may fail indexing.
    # NULL means indexing was not tracked for a historical job.
    indexing_status: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    indexing_attempt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    # Store a safe error code, not raw provider exceptions or credentials.
    indexing_error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
