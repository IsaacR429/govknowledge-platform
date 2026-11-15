import pytest

from app.infrastructure.parsing.pymupdf_table_evidence import (
    extract_physical_table_evidence,
)


class FakeHeader:
    def __init__(
        self,
        *,
        external: bool,
    ) -> None:
        self.external = external


class FakePhysicalRow:
    def __init__(
        self,
        cells,
    ) -> None:
        self.cells = cells


class FakeTable:
    def __init__(
        self,
        *,
        matrix,
        bbox=(10, 20, 300, 200),
        cell_boxes=None,
        external_header=False,
    ) -> None:
        self._matrix = matrix
        self.row_count = len(
            matrix
        )
        self.col_count = (
            len(matrix[0])
            if matrix
            else 0
        )
        self.bbox = bbox

        self.header = FakeHeader(
            external=external_header
        )

        if cell_boxes is None:
            cell_boxes = [
                [
                    (
                        10 + column * 50,
                        20 + row * 20,
                        60 + column * 50,
                        40 + row * 20,
                    )
                    for column in range(
                        self.col_count
                    )
                ]
                for row in range(
                    self.row_count
                )
            ]

        self.rows = [
            FakePhysicalRow(
                row
            )
            for row in cell_boxes
        ]

    def extract(self):
        return self._matrix


class FakeFinder:
    def __init__(
        self,
        tables,
    ) -> None:
        self.tables = tables


class FakePage:
    def __init__(
        self,
        tables,
        *,
        words=(),
    ) -> None:
        self._tables = tables
        self._words = words

    def find_tables(self):
        return FakeFinder(
            self._tables
        )

    def get_text(
        self,
        mode: str,
    ):
        assert mode == "words"

        return list(
            self._words
        )


def test_extracts_generic_physical_table() -> None:
    page = FakePage(
        [
            FakeTable(
                matrix=[
                    [
                        "Domaine",
                        "Ecosystème",
                        "Composants",
                    ],
                    [
                        "Productif",
                        "Infrastructure",
                        "Equipements réseaux",
                    ],
                ]
            )
        ]
    )

    result = (
        extract_physical_table_evidence(
            page,
            page_number=9,
        )
    )

    assert len(result) == 1

    table = result[0]

    assert (
        table["source_ref"]
        == "native:p9:t0"
    )

    assert table["layer"] == "PHYSICAL"
    assert table["column_count"] == 3
    assert len(table["rows"]) == 2

    assert (
        table["rows"][0]
        ["source_header_signal"]
        is True
    )

    assert (
        table["rows"][1]
        ["cells"][0]["text"]
        == "Productif"
    )


def test_empty_string_is_observed_blank() -> None:
    page = FakePage(
        [
            FakeTable(
                matrix=[
                    [
                        "",
                        "",
                        "Value",
                    ]
                ]
            )
        ]
    )

    result = (
        extract_physical_table_evidence(
            page,
            page_number=10,
        )
    )

    cells = (
        result[0]
        ["rows"][0]
        ["cells"]
    )

    assert (
        cells[0]["text_status"]
        == "OBSERVED"
    )

    assert cells[0]["text"] == ""

    assert (
        cells[1]["text_status"]
        == "OBSERVED"
    )


def test_none_is_unavailable_not_blank() -> None:
    page = FakePage(
        [
            FakeTable(
                matrix=[
                    [
                        None,
                        "Known",
                    ]
                ]
            )
        ]
    )

    result = (
        extract_physical_table_evidence(
            page,
            page_number=1,
        )
    )

    cell = (
        result[0]
        ["rows"][0]
        ["cells"][0]
    )

    assert (
        cell["text_status"]
        == "UNAVAILABLE"
    )

    assert cell["text"] is None


def test_external_header_does_not_mark_first_row_header() -> None:
    page = FakePage(
        [
            FakeTable(
                matrix=[
                    [
                        "01",
                        "Supplier",
                    ]
                ],
                external_header=True,
            )
        ]
    )

    result = (
        extract_physical_table_evidence(
            page,
            page_number=4,
        )
    )

    assert (
        result[0]["rows"][0]
        ["source_header_signal"]
        is False
    )


def test_rejects_row_width_mismatch() -> None:
    table = FakeTable(
        matrix=[
            ["A", "B"],
        ]
    )

    table.col_count = 3

    page = FakePage(
        [table]
    )

    with pytest.raises(
        ValueError,
        match="row width",
    ):
        extract_physical_table_evidence(
            page,
            page_number=1,
        )


def test_text_span_uses_actual_content_geometry() -> None:
    page = FakePage(
        [
            FakeTable(
                matrix=[
                    [
                        "Alpha",
                    ]
                ],
                cell_boxes=[
                    [
                        (
                            10,
                            20,
                            60,
                            40,
                        )
                    ]
                ],
            )
        ],
        words=[
            (
                14,
                24,
                34,
                32,
                "Alpha",
                0,
                0,
                0,
            )
        ],
    )

    result = (
        extract_physical_table_evidence(
            page,
            page_number=1,
        )
    )

    cell = (
        result[0]
        ["rows"][0]
        ["cells"][0]
    )

    geometry = next(
        span
        for span in cell["source_spans"]
        if span["kind"] == "GEOMETRY"
    )

    text = next(
        span
        for span in cell["source_spans"]
        if span["kind"] == "TEXT"
    )

    assert geometry["bbox"] == [
        10.0,
        20.0,
        60.0,
        40.0,
    ]

    assert text["bbox"] == [
        14.0,
        24.0,
        34.0,
        32.0,
    ]
