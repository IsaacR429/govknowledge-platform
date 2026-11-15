from app.infrastructure.parsing.pymupdf_tagged_table_evidence import (
    extract_structural_table_evidence,
)


class FakePage:
    def __init__(
        self,
        xref: int,
    ) -> None:
        self.xref = xref


class FakeDocument:
    page_count = 2

    def __init__(self) -> None:
        self._pages = [
            FakePage(100),
            FakePage(101),
        ]

        self._keys = {
            (1, "StructTreeRoot"):
                ("xref", "10 0 R"),

            (10, "K"):
                ("array", "[20 0 R]"),

            (20, "S"):
                ("name", "/Table"),
            (20, "K"):
                (
                    "array",
                    "[30 0 R 40 0 R "
                    "50 0 R 60 0 R]"
                ),

            # Page 1 header
            (30, "S"):
                ("name", "/TR"),
            (30, "Pg"):
                ("xref", "100 0 R"),
            (30, "K"):
                (
                    "array",
                    "[31 0 R 32 0 R 33 0 R]"
                ),

            (31, "S"):
                ("name", "/TH"),
            (32, "S"):
                ("name", "/TH"),
            (33, "S"):
                ("name", "/TH"),

            # Page 1 data. Cells inherit page from row.
            (40, "S"):
                ("name", "/TR"),
            (40, "Pg"):
                ("xref", "100 0 R"),
            (40, "K"):
                (
                    "array",
                    "[41 0 R 42 0 R 43 0 R]"
                ),

            (41, "S"):
                ("name", "/TD"),
            (42, "S"):
                ("name", "/TD"),
            (43, "S"):
                ("name", "/TD"),

            # Page 2 repeated header
            (50, "S"):
                ("name", "/TR"),
            (50, "Pg"):
                ("xref", "101 0 R"),
            (50, "K"):
                (
                    "array",
                    "[51 0 R 52 0 R 53 0 R]"
                ),

            (51, "S"):
                ("name", "/TH"),
            (52, "S"):
                ("name", "/TH"),
            (53, "S"):
                ("name", "/TH"),

            # Page 2 mixed TH/TD data row.
            # This must NOT become a canonical header signal.
            (60, "S"):
                ("name", "/TR"),
            (60, "Pg"):
                ("xref", "101 0 R"),
            (60, "K"):
                (
                    "array",
                    "[61 0 R 62 0 R 63 0 R]"
                ),

            (61, "S"):
                ("name", "/TH"),
            (62, "S"):
                ("name", "/TD"),
            (63, "S"):
                ("name", "/TD"),
        }

    def pdf_catalog(self) -> int:
        return 1

    def load_page(
        self,
        offset: int,
    ) -> FakePage:
        return self._pages[
            offset
        ]

    def xref_get_key(
        self,
        xref: int,
        key: str,
    ):
        return self._keys.get(
            (xref, key),
            ("null", "null"),
        )


class UntaggedDocument:
    page_count = 1

    def pdf_catalog(self) -> int:
        return 1

    def load_page(
        self,
        offset: int,
    ) -> FakePage:
        return FakePage(100)

    def xref_get_key(
        self,
        xref: int,
        key: str,
    ):
        return (
            "null",
            "null",
        )


def test_splits_multipage_table_into_page_local_evidence() -> None:
    result = (
        extract_structural_table_evidence(
            FakeDocument()
        )
    )

    assert len(result) == 2

    assert [
        table["page_number"]
        for table in result
    ] == [
        1,
        2,
    ]

    assert all(
        table["column_count"] == 3
        for table in result
    )


def test_preserves_original_table_xref_across_pages() -> None:
    result = (
        extract_structural_table_evidence(
            FakeDocument()
        )
    )

    assert (
        result[0]["source_ref"]
        == "tag:xref:20:p1"
    )

    assert (
        result[1]["source_ref"]
        == "tag:xref:20:p2"
    )


def test_all_th_row_is_header_signal() -> None:
    result = (
        extract_structural_table_evidence(
            FakeDocument()
        )
    )

    page_one = result[0]

    assert (
        page_one["rows"][0]
        ["source_header_signal"]
        is True
    )


def test_mixed_th_td_row_is_not_header_signal() -> None:
    result = (
        extract_structural_table_evidence(
            FakeDocument()
        )
    )

    page_two = result[1]

    assert (
        page_two["rows"][1]
        ["source_header_signal"]
        is False
    )


def test_structural_cells_do_not_invent_text() -> None:
    result = (
        extract_structural_table_evidence(
            FakeDocument()
        )
    )

    cell = (
        result[0]["rows"][0]
        ["cells"][0]
    )

    assert (
        cell["text_status"]
        == "UNAVAILABLE"
    )

    assert cell["text"] is None

    assert (
        cell["source_role"]
        == "TH"
    )


def test_untagged_document_returns_no_structural_evidence() -> None:
    result = (
        extract_structural_table_evidence(
            UntaggedDocument()
        )
    )

    assert result == []
