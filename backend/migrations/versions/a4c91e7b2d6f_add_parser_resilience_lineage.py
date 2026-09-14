"""add parser resilience lineage

Revision ID: a4c91e7b2d6f
Revises: 1d4af14dfe04
Create Date: 2026-09-14

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "a4c91e7b2d6f"
down_revision: Union[str, Sequence[str], None] = (
    "1d4af14dfe04"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_processing_jobs",
        sa.Column(
            "parsing_status",
            sa.String(length=20),
            nullable=True,
        ),
    )

    op.add_column(
        "document_processing_jobs",
        sa.Column(
            "failure_class",
            sa.String(length=64),
            nullable=True,
        ),
    )

    op.add_column(
        "document_processing_jobs",
        sa.Column(
            "parser_lineage_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )

    op.create_check_constraint(
        "ck_processing_job_parsing_status",
        "document_processing_jobs",
        "parsing_status IS NULL "
        "OR parsing_status IN ('SUCCESS', 'FAILED')",
    )

    op.create_check_constraint(
        "ck_processing_job_failure_class",
        "document_processing_jobs",
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
    )

    op.create_check_constraint(
        "ck_processing_job_parser_outcome",
        "document_processing_jobs",
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
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_processing_job_parser_outcome",
        "document_processing_jobs",
        type_="check",
    )

    op.drop_constraint(
        "ck_processing_job_failure_class",
        "document_processing_jobs",
        type_="check",
    )

    op.drop_constraint(
        "ck_processing_job_parsing_status",
        "document_processing_jobs",
        type_="check",
    )

    op.drop_column(
        "document_processing_jobs",
        "parser_lineage_json",
    )

    op.drop_column(
        "document_processing_jobs",
        "failure_class",
    )

    op.drop_column(
        "document_processing_jobs",
        "parsing_status",
    )
