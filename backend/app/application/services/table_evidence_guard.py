from __future__ import annotations

from app.application.ports.document_parser import ParsedDocument


class UnapprovedTableEvidenceError(ValueError):
    """A parsed document contains table evidence requiring approval."""


def reject_unapproved_table_evidence(parsed: ParsedDocument) -> None:
    """Block table-derived chunks until an approval path exists."""

    for chunk in parsed.chunks:
        metadata = chunk.metadata

        if (
            metadata.get("validation_status") == "candidate"
            or metadata.get("evidence_type") == "vector_marker_table"
            or str(metadata.get("extraction_method", "")).startswith(
                "vector_marker_table"
            )
            or any(
                key in metadata
                for key in (
                    "table_id",
                    "marked_activities",
                    "activity_evidence",
                )
            )
        ):
            raise UnapprovedTableEvidenceError(
                "Unapproved table evidence cannot be persisted "
                "as a searchable chunk."
            )
