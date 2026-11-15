from __future__ import annotations


def build_table_evidence_candidates(
    *,
    page_number: int,
    tables: list[dict],
) -> list[dict]:
    """Prepare reviewable evidence; do not insert into searchable chunks."""

    candidates: list[dict] = []

    # Sort spatially so table identity does not depend on the
    # order in which PyMuPDF happens to return the tables.
    ordered_tables = sorted(
        tables,
        key=lambda table: (
            table["table_bbox"][0],
            table["table_bbox"][1],
        ),
    )

    for table_number, table in enumerate(ordered_tables, start=1):
        if table.get("validation_status") != "candidate":
            raise ValueError("Unexpected table validation status.")

        table_id = f"page-{page_number}-table-{table_number}"

        for record in table["records"]:
            organization = record["organization"]
            activities = record["activities"]

            if not organization or not activities:
                raise ValueError("Incomplete marked table row.")

            candidates.append(
                {
                    "content": (
                        f"PDF page {page_number}, table {table_number}. "
                        f"Organization: {organization}. "
                        f"Marked activities: {', '.join(activities)}."
                    ),
                    "metadata": {
                        "evidence_type": "vector_marker_table",
                        "validation_status": "candidate",
                        "page_number": page_number,
                        "table_id": table_id,
                        "table_bbox": table["table_bbox"],
                        "row_index": record["row_index"],
                        "organization": organization,
                        "marked_activities": activities,
                        "activity_evidence": record["activity_evidence"],
                    },
                }
            )

    return candidates
