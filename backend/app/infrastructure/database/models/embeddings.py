from __future__ import annotations

from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base
from app.infrastructure.database.models.mixins import (
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)


EMBEDDING_DIMENSION = 768


class ChunkEmbedding(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    __tablename__ = "chunk_embeddings"

    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "model_name",
            "model_revision",
            name="uq_chunk_embedding_model_revision",
        ),
        CheckConstraint(
            "dimension = 768",
            name="ck_chunk_embedding_dimension_768",
        ),
        CheckConstraint(
            "normalized = true",
            name="ck_chunk_embedding_normalized",
        ),
    )

    chunk_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "document_chunks.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(300),
        nullable=False,
    )

    model_revision: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=EMBEDDING_DIMENSION,
    )

    normalized: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=False,
    )
