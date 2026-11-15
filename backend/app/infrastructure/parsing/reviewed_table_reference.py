from __future__ import annotations

import hashlib
from pathlib import Path


REVIEWED_PDF_SHA256 = (
    "d4df4c4a4c4c6e2cd364c86adf31266ae5fe55e4a7f15d23fa7c3bf558aa3940"
)

REVIEWED_PAGE_NUMBER = 19

# Independently reviewed against physical PDF page 19.
# Keep the left and right tables separate: DLL appears in both.
REVIEWED_LEFT = """
Agences régionales|Creation,Diffusion,Network
DLL|Creation,Diffusion
DRAC|Creation,Diffusion,Training
ADELC|Diffusion
ATLAS|Network
ATLF|Network
Beaumarchais-SACD|Creation,Diffusion
BNF|Diffusion,Network
CFC|Diffusion
CIPM|Creation,Diffusion,Network
CNL|Creation,Diffusion,Network
Ent’revues|Diffusion,Training,Network
FISAC|Diffusion
Fonds Jules Verne|Creation,Diffusion
Maison des auteurs|Creation,Training,Network
MEL|Network
Printemps des poètes|Diffusion
SCELF|Diffusion
SGDL|Network
SLF|Network
SNE|Diffusion,Network
SNAC|Network
SOFIA|Creation,Training
Villa Gillet|Creation,Diffusion,Network
Villa Medicis|Creation
FAEE|Network
"""

REVIEWED_RIGHT = """
FEL|Network
FEE|Network
AILF|Diffusion,Network
Biblionef|Creation
BIEF|Diffusion,Network
CELF|Diffusion
Centrale de l’édition|Diffusion
Culturesfrance|Creation,Diffusion,Network
DLL|Diffusion,Network
IFLA|Network
Missions Stendhal|Creation,Diffusion
Ubifrance|Diffusion
UIE|Network
"""


def _normalize(value: str) -> str:
    return "".join(value.split())


def _reference(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rows = []

    for line in text.strip().splitlines():
        name, activities = line.strip().split("|", 1)
        rows.append(
            (
                _normalize(name),
                tuple(part.strip() for part in activities.split(",")),
            )
        )

    return tuple(rows)


def _extracted(table: dict) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (
            _normalize(record["organization"]),
            tuple(record["activities"]),
        )
        for record in table["records"]
    )


def verify_reviewed_table_fixture(
    *,
    pdf_path: Path,
    page_number: int,
    tables: list[dict],
) -> None:
    """
    Validate the complete reviewed table fixture.

    This verifies one known PDF. It is not approval for other
    documents or table formats.
    """
    if page_number != REVIEWED_PAGE_NUMBER:
        raise ValueError("Page is not the reviewed table fixture.")

    actual_hash = hashlib.sha256(
        pdf_path.read_bytes()
    ).hexdigest()

    if actual_hash != REVIEWED_PDF_SHA256:
        raise ValueError("PDF does not match the reviewed source.")

    if len(tables) != 2:
        raise ValueError("Reviewed page must contain exactly two tables.")

    left, right = sorted(
        tables,
        key=lambda table: table["table_bbox"][0],
    )

    for table, expected, expected_markers in (
        (left, _reference(REVIEWED_LEFT), 46),
        (right, _reference(REVIEWED_RIGHT), 19),
    ):
        if table.get("pdf_page_number") != page_number:
            raise ValueError("Table page identity mismatch.")

        if table.get("validation_status") != "candidate":
            raise ValueError("Unexpected table status.")

        if _extracted(table) != expected:
            raise ValueError(
                "Extracted table differs from the reviewed reference."
            )

        if table.get("marker_count") != expected_markers:
            raise ValueError("Table marker count mismatch.")

        marker_evidence = [
            marker
            for record in table["records"]
            for marker in record["activity_evidence"]
        ]

        if len(marker_evidence) != expected_markers:
            raise ValueError("Marker evidence count mismatch.")

        for record in table["records"]:
            if (
                tuple(marker["activity"] for marker in record["activity_evidence"])
                != tuple(record["activities"])
            ):
                raise ValueError("Activity and marker evidence disagree.")

            for marker in record["activity_evidence"]:
                x0, y0, x1, y1 = marker["marker_bbox"]
                cx0, cy0, cx1, cy1 = marker["cell_bbox"]

                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2

                if not (
                    cx0 <= center_x <= cx1
                    and cy0 <= center_y <= cy1
                ):
                    raise ValueError(
                        "Marker is outside its claimed table cell."
                    )
