"""add indexing state to processing jobs

Revision ID: 2f5fbf543943
Revises: a4c91e7b2d6f
Create Date: 2026-09-19 06:39:17.425003

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2f5fbf543943'
down_revision: Union[str, Sequence[str], None] = 'a4c91e7b2d6f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_processing_jobs",
        sa.Column("indexing_status", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "document_processing_jobs",
        sa.Column(
            "indexing_attempt_number",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.add_column(
        "document_processing_jobs",
        sa.Column("indexing_error_code", sa.String(length=100), nullable=True),
    )

    op.create_check_constraint(
        "ck_processing_job_indexing_status",
        "document_processing_jobs",
        "indexing_status IS NULL OR indexing_status IN "
        "('PENDING', 'INDEXING', 'SUCCEEDED', 'FAILED', 'NO_CHUNKS')",
    )
    op.create_check_constraint(
        "ck_processing_job_indexing_attempt_nonnegative",
        "document_processing_jobs",
        "indexing_attempt_number >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_processing_job_indexing_attempt_nonnegative",
        "document_processing_jobs",
        type_="check",
    )
    op.drop_constraint(
        "ck_processing_job_indexing_status",
        "document_processing_jobs",
        type_="check",
    )
    op.drop_column("document_processing_jobs", "indexing_error_code")
    op.drop_column("document_processing_jobs", "indexing_attempt_number")
    op.drop_column("document_processing_jobs", "indexing_status")
