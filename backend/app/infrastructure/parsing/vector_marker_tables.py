from __future__ import annotations

import pymupdf


ACTIVITY_HEADERS = (
    "Creation",
    "Diffusion",
    "Training",
    "Network",
)

# Marker appearance observed in the test PDF.
MARKER_COLOR = (0.0, 0.388, 0.563)


def _is_marker(drawing: dict) -> bool:
    rect = drawing["rect"]
    fill = drawing.get("fill")

    return (
        drawing.get("type") == "f"
        and fill is not None
        and len(fill) == 3
        and 4 <= rect.width <= 8
        and 4 <= rect.height <= 8
        and all(
            abs(actual - expected) < 0.02
            for actual, expected in zip(fill, MARKER_COLOR)
        )
    )


def extract_vector_marker_tables(
    page: pymupdf.Page,
    *,
    page_number: int,
) -> list[dict]:
    """Extract verified marker-to-cell candidates from this table format."""

    results = []
    drawings = page.get_drawings()

    for table in page.find_tables(strategy="lines").tables:
        rows = table.extract()

        if not rows or len(rows[0]) != 6:
            continue

        headers = tuple(
            str(value or "").strip()
            for value in rows[0][2:6]
        )

        if headers != ACTIVITY_HEADERS:
            continue

        table_bounds = pymupdf.Rect(table.bbox)
        markers_by_row: dict[int, dict[int, dict]] = {}
        marker_count = 0

        for drawing in drawings:
            if not _is_marker(drawing):
                continue

            rect = drawing["rect"]

            if not table_bounds.contains(rect):
                continue

            center = pymupdf.Point(
                (rect.x0 + rect.x1) / 2,
                (rect.y0 + rect.y1) / 2,
            )

            matches = [
                (row_index, col_index, cell)
                for row_index, row in enumerate(table.rows)
                for col_index, cell in enumerate(row.cells)
                if cell is not None
                and pymupdf.Rect(cell).contains(center)
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Marker has {len(matches)} cell matches "
                    f"on PDF page {page_number}."
                )

            row_index, col_index, cell = matches[0]

            if row_index == 0 or col_index not in range(2, 6):
                raise ValueError(
                    "Marker falls outside an activity cell."
                )

            columns = markers_by_row.setdefault(row_index, {})

            if col_index in columns:
                raise ValueError(
                    "Duplicate marker in one table cell."
                )

            columns[col_index] = {
                "marker_bbox": list(rect),
                "cell_bbox": list(cell),
            }

            marker_count += 1

        if not markers_by_row:
            continue

        records = []

        for row_index, columns in sorted(markers_by_row.items()):
            label = " / ".join(
                str(value).strip()
                for value in rows[row_index][:2]
                if value and str(value).strip()
            )

            if not label:
                raise ValueError("Marked row has no organization label.")

            records.append(
                {
                    "organization": label,
                    "activities": [
                        ACTIVITY_HEADERS[col_index - 2]
                        for col_index in sorted(columns)
                    ],
                    "row_index": row_index,
                    "activity_evidence": [
                        {
                            "activity": ACTIVITY_HEADERS[col_index - 2],
                            "column_index": col_index,
                            "marker_bbox": columns[col_index]["marker_bbox"],
                            "cell_bbox": columns[col_index]["cell_bbox"],
                        }
                        for col_index in sorted(columns)
                    ],
                }
            )

        results.append(
            {
                "pdf_page_number": page_number,
                "table_bbox": list(table.bbox),
                "marker_count": marker_count,
                "records": records,
                "extraction_method": "vector_marker_table",
                "validation_status": "candidate",
            }
        )

    return results
