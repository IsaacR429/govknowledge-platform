from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path

import pymupdf

from app.infrastructure.parsing.vector_marker_tables import (
    extract_vector_marker_tables,
)


EXPECTED_SHA256 = (
    "d4df4c4a4c4c6e2cd364c86adf31266ae5fe55e4a7f15d23fa7c3bf558aa3940"
)


class VectorMarkerTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configured = os.environ.get("GOVKNOWLEDGE_TABLE_FIXTURE_PDF")
        if not configured:
            raise unittest.SkipTest("PDF fixture path not configured.")

        cls.pdf_path = Path(configured)
        if not cls.pdf_path.is_file():
            raise FileNotFoundError(cls.pdf_path)

        actual_hash = hashlib.sha256(
            cls.pdf_path.read_bytes()
        ).hexdigest()
        if actual_hash != EXPECTED_SHA256:
            raise AssertionError("PDF fixture hash mismatch.")

    def test_page_19_marker_relationships(self) -> None:
        with pymupdf.open(self.pdf_path) as pdf:
            tables = extract_vector_marker_tables(
                pdf[18],
                page_number=19,
            )

        self.assertEqual(len(tables), 2)
        self.assertEqual(
            sum(table["marker_count"] for table in tables),
            65,
        )
        self.assertEqual(
            sum(len(table["records"]) for table in tables),
            39,
        )

        # Keep the left and right tables separate.
        left, right = sorted(
            tables,
            key=lambda table: table["table_bbox"][0],
        )

        def activities(table: dict, organization: str) -> list[str]:
            matches = [
                record["activities"]
                for record in table["records"]
                if record["organization"] == organization
            ]
            self.assertEqual(len(matches), 1)
            return matches[0]

        self.assertEqual(
            activities(left, "Agences r\u00e9gionales"),
            ["Creation", "Diffusion", "Network"],
        )
        self.assertEqual(
            activities(left, "DRAC"),
            ["Creation", "Diffusion", "Training"],
        )
        self.assertEqual(
            activities(left, "DLL"),
            ["Creation", "Diffusion"],
        )
        self.assertEqual(
            activities(right, "DLL"),
            ["Diffusion", "Network"],
        )
        self.assertEqual(
            activities(right, "FEL"),
            ["Network"],
        )

    def test_page_19_complete_reference(self) -> None:
        # Reference transcribed from the original PDF page.
        # Keep the two physical tables separate: DLL appears in both.
        expected_left = """
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

        expected_right = """
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

        def normalize(value: str) -> str:
            return " ".join(value.split())

        def reference(text: str) -> dict[str, tuple[str, ...]]:
            result = {}
            for line in text.strip().splitlines():
                name, activities = line.strip().split("|", 1)
                result[normalize(name)] = tuple(
                    item.strip() for item in activities.split(",")
                )
            return result

        def extracted(table: dict) -> dict[str, tuple[str, ...]]:
            return {
                normalize(row["organization"]): tuple(row["activities"])
                for row in table["records"]
            }

        with pymupdf.open(self.pdf_path) as pdf:
            tables = extract_vector_marker_tables(
                pdf[18],
                page_number=19,
            )

        self.assertEqual(len(tables), 2)
        left, right = sorted(
            tables,
            key=lambda table: table["table_bbox"][0],
        )

        self.assertEqual(extracted(left), reference(expected_left))
        self.assertEqual(extracted(right), reference(expected_right))

    def test_cover_does_not_produce_marker_tables(self) -> None:
        with pymupdf.open(self.pdf_path) as pdf:
            tables = extract_vector_marker_tables(
                pdf[0],
                page_number=1,
            )

        self.assertEqual(tables, [])


if __name__ == "__main__":
    unittest.main()
