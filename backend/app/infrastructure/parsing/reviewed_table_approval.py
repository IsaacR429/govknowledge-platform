from __future__ import annotations

import hashlib
from pathlib import Path

from app.application.ports.document_parser import ParsedDocument
from app.infrastructure.parsing.reviewed_table_reference import (
    REVIEWED_PDF_SHA256,
    REVIEWED_PAGE_NUMBER,
    verify_reviewed_table_fixture,
)
from app.infrastructure.parsing.table_evidence import (
    build_table_evidence_candidates,
)


TABLE_METADATA_KEYS = frozenset({
    "evidence_type",
    "table_id",
    "marked_activities",
    "activity_evidence",
})


def _is_table_chunk(metadata: dict) -> bool:
    return (
        metadata.get("validation_status") == "candidate"
        or str(metadata.get("extraction_method", "")).startswith(
            "vector_marker_table"
        )
        or any(key in metadata for key in TABLE_METADATA_KEYS)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_reviewed_table_evidence(
    *,
    parsed: ParsedDocument,
    stored_pdf_path: Path,
    version_file_hash: str,
) -> tuple[int, ...]:
    """
    Return approved chunk indexes for one reviewed PDF fixture.

    No metadata label or caller-supplied approval flag can
    authorize a different document or a changed table chunk.
    """
    table_chunks = [
        chunk
        for chunk in parsed.chunks
        if _is_table_chunk(chunk.metadata)
    ]

    if not table_chunks:
        return ()

    actual_hash = _sha256_file(stored_pdf_path)

    if (
        actual_hash != version_file_hash
        or actual_hash != REVIEWED_PDF_SHA256
    ):
        raise ValueError(
            "Table evidence source does not match the reviewed PDF."
        )

    matching_pages = [
        page
        for page in parsed.pages
        if page.page_number == REVIEWED_PAGE_NUMBER
    ]

    if len(matching_pages) != 1:
        raise ValueError("Reviewed table page is missing or duplicated.")

    tables = matching_pages[0].metadata.get("table_candidates")

    if not isinstance(tables, list):
        raise ValueError("Reviewed table candidates are missing.")

    verify_reviewed_table_fixture(
        pdf_path=stored_pdf_path,
        page_number=REVIEWED_PAGE_NUMBER,
        tables=tables,
    )

    expected = build_table_evidence_candidates(
        page_number=REVIEWED_PAGE_NUMBER,
        tables=tables,
    )

    if len(expected) != 39 or len(table_chunks) != len(expected):
        raise ValueError("Table evidence batch is incomplete or oversized.")

    # Match by table AND row. Organization name alone is not unique:
    # DLL occurs in both tables.
    expected_by_key = {
        (
            item["metadata"]["table_id"],
            item["metadata"]["row_index"],
        ): item
        for item in expected
    }

    if len(expected_by_key) != len(expected):
        raise ValueError("Reviewed reference has duplicate table rows.")

    approved_indexes: list[int] = []
    seen_keys: set[tuple[str, int]] = set()

    for chunk in table_chunks:
        metadata = chunk.metadata

        if (
            chunk.page_number != REVIEWED_PAGE_NUMBER
            or metadata.get("validation_status") != "candidate"
            or metadata.get("extraction_method")
            != "vector_marker_table_candidate"
            or metadata.get("evidence_type")
            != "vector_marker_table"
        ):
            raise ValueError("Unapproved table chunk type or page.")

        key = (
            metadata.get("table_id"),
            metadata.get("row_index"),
        )

        if key in seen_keys or key not in expected_by_key:
            raise ValueError("Unexpected or duplicate table evidence row.")

        seen_keys.add(key)
        reference = expected_by_key[key]

        if chunk.content != reference["content"]:
            raise ValueError("Table evidence text differs from source.")

        for field, expected_value in reference["metadata"].items():
            if metadata.get(field) != expected_value:
                raise ValueError(
                    f"Table evidence metadata differs: {field}."
                )

        approved_indexes.append(chunk.chunk_index)

    if seen_keys != set(expected_by_key):
        raise ValueError("Table evidence is missing reviewed rows.")

    return tuple(approved_indexes)
