import pytest

from app.domain.enums.table import (
    TableCellTextStatus,
    TableCellValueSource,
    TableEvidenceLayer,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    TableBBox,
    TableSourceSpan,
)
from app.domain.tables.canonicalizer import (
    TableCanonicalizationError,
    TableCanonicalizer,
)
from app.domain.tables.evidence import (
    NormalizedTableCellEvidence,
    NormalizedTableEvidence,
    NormalizedTableRowEvidence,
)


def source_span(
    *,
    kind: TableSpanKind,
    ref: str,
    text=None,
):
    return TableSourceSpan(
        page_number=1,
        kind=kind,
        source_ref=ref,
        bbox=TableBBox(
            x0=0,
            y0=0,
            x1=10,
            y1=10,
        ),
        text=text,
    )


def physical_cell(
    column: int,
    *,
    text=None,
    blank=False,
    unavailable=False,
):
    if unavailable:
        return NormalizedTableCellEvidence(
            column_start=column,
            column_end=column + 1,
            text_status=(
                TableCellTextStatus
                .UNAVAILABLE
            ),
            text=None,
            source_spans=(
                source_span(
                    kind=(
                        TableSpanKind.GEOMETRY
                    ),
                    ref=f"g:{column}",
                ),
            ),
        )

    value = (
        ""
        if blank
        else text
    )

    spans = [
        source_span(
            kind=(
                TableSpanKind.GEOMETRY
            ),
            ref=f"g:{column}",
        )
    ]

    if value:
        spans.append(
            source_span(
                kind=TableSpanKind.TEXT,
                ref=f"t:{column}",
                text=value,
            )
        )

    return NormalizedTableCellEvidence(
        column_start=column,
        column_end=column + 1,
        text_status=(
            TableCellTextStatus.OBSERVED
        ),
        text=value,
        source_spans=tuple(
            spans
        ),
    )


def structural_cell(
    column: int,
):
    return NormalizedTableCellEvidence(
        column_start=column,
        column_end=column + 1,
        text_status=(
            TableCellTextStatus
            .UNAVAILABLE
        ),
        text=None,
        source_spans=(
            source_span(
                kind=(
                    TableSpanKind.STRUCTURE
                ),
                ref=f"s:{column}",
            ),
        ),
        source_role="TD",
    )


def row(
    index: int,
    cells,
    *,
    header=False,
):
    return NormalizedTableRowEvidence(
        row_index=index,
        cells=tuple(
            cells
        ),
        source_header_signal=header,
    )


def table(
    *,
    ref: str,
    layer: TableEvidenceLayer,
    columns: int,
    rows,
):
    return NormalizedTableEvidence(
        source_ref=ref,
        page_number=1,
        layer=layer,
        column_count=columns,
        rows=tuple(
            rows
        ),
    )


def test_semantic_text_becomes_source_text() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=1,
        rows=(
            row(
                0,
                (
                    physical_cell(
                        0,
                        text="Agriculture",
                    ),
                ),
            ),
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=1,
        rows=(
            row(
                0,
                (
                    structural_cell(0),
                ),
            ),
        ),
    )

    result = (
        TableCanonicalizer()
        .canonicalize(
            aligned_physical=physical,
            structural=structural,
        )
    )

    cell = (
        result.table.rows[0]
        .cells[0]
    )

    assert cell.text == "Agriculture"

    assert (
        cell.value_source
        == TableCellValueSource
        .SOURCE_TEXT
    )

    assert len(
        cell.origin_spans
    ) == 1

    assert (
        cell.origin_spans[0].kind
        == TableSpanKind.TEXT
    )


def test_observed_blank_remains_source_blank() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=1,
        rows=(
            row(
                0,
                (
                    physical_cell(
                        0,
                        blank=True,
                    ),
                ),
            ),
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=1,
        rows=(
            row(
                0,
                (
                    structural_cell(0),
                ),
            ),
        ),
    )

    result = (
        TableCanonicalizer()
        .canonicalize(
            aligned_physical=physical,
            structural=structural,
        )
    )

    cell = (
        result.table.rows[0]
        .cells[0]
    )

    assert cell.text == ""

    assert (
        cell.value_source
        == TableCellValueSource
        .SOURCE_TEXT
    )

    assert (
        result.observed_blank_count
        == 1
    )


def test_unavailable_logical_cell_becomes_derived_blank() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=1,
        rows=(
            row(
                0,
                (
                    physical_cell(
                        0,
                        unavailable=True,
                    ),
                ),
            ),
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=1,
        rows=(
            row(
                0,
                (
                    structural_cell(0),
                ),
            ),
        ),
    )

    result = (
        TableCanonicalizer()
        .canonicalize(
            aligned_physical=physical,
            structural=structural,
        )
    )

    cell = (
        result.table.rows[0]
        .cells[0]
    )

    assert cell.text == ""

    assert (
        cell.value_source
        == TableCellValueSource
        .DERIVED
    )

    assert cell.origin_spans == ()

    assert (
        result.derived_blank_count
        == 1
    )


def test_full_row_preserves_logical_columns() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=3,
        rows=(
            row(
                0,
                (
                    physical_cell(
                        0,
                        unavailable=True,
                    ),
                    physical_cell(
                        1,
                        unavailable=True,
                    ),
                    physical_cell(
                        2,
                        text="Professionnels",
                    ),
                ),
            ),
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=3,
        rows=(
            row(
                0,
                (
                    structural_cell(0),
                    structural_cell(1),
                    structural_cell(2),
                ),
            ),
        ),
    )

    result = (
        TableCanonicalizer()
        .canonicalize(
            aligned_physical=physical,
            structural=structural,
        )
    )

    cells = (
        result.table.rows[0]
        .cells
    )

    assert [
        cell.column_start
        for cell in cells
    ] == [
        0,
        1,
        2,
    ]

    assert [
        cell.text
        for cell in cells
    ] == [
        "",
        "",
        "Professionnels",
    ]


def test_sparse_row_maps_by_occupied_physical_columns() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=(
            row(
                0,
                (
                    physical_cell(
                        0,
                        text="ATTRIBUTAIRES",
                    ),
                    physical_cell(
                        1,
                        unavailable=True,
                    ),
                    physical_cell(
                        2,
                        text="Lot 1 ...",
                    ),
                    physical_cell(
                        3,
                        unavailable=True,
                    ),
                    physical_cell(
                        4,
                        unavailable=True,
                    ),
                ),
            ),
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=5,
        rows=(
            row(
                0,
                (
                    structural_cell(0),
                    structural_cell(1),
                ),
            ),
        ),
    )

    result = (
        TableCanonicalizer()
        .canonicalize(
            aligned_physical=physical,
            structural=structural,
        )
    )

    cells = (
        result.table.rows[0]
        .cells
    )

    assert [
        cell.column_start
        for cell in cells
    ] == [
        0,
        2,
    ]

    assert [
        cell.text
        for cell in cells
    ] == [
        "ATTRIBUTAIRES",
        "Lot 1 ...",
    ]


def test_explicit_header_creates_column_names() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=3,
        rows=(
            row(
                0,
                (
                    physical_cell(
                        0,
                        text="Domaine",
                    ),
                    physical_cell(
                        1,
                        text="Ecosystème",
                    ),
                    physical_cell(
                        2,
                        text="Composants",
                    ),
                ),
                header=True,
            ),
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=3,
        rows=(
            row(
                0,
                (
                    structural_cell(0),
                    structural_cell(1),
                    structural_cell(2),
                ),
                header=True,
            ),
        ),
    )

    result = (
        TableCanonicalizer()
        .canonicalize(
            aligned_physical=physical,
            structural=structural,
        )
    )

    canonical = (
        result.table
    )

    assert (
        canonical.rows[0].role
        == TableRowRole.HEADER
    )

    assert canonical.column_names == (
        "Domaine",
        "Ecosystème",
        "Composants",
    )


def test_row_index_mismatch_fails_closed() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=1,
        rows=(
            row(
                0,
                (
                    physical_cell(
                        0,
                        text="A",
                    ),
                ),
            ),
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=1,
        rows=(
            row(
                1,
                (
                    structural_cell(0),
                ),
            ),
        ),
    )

    with pytest.raises(
        TableCanonicalizationError,
        match="row indexes",
    ):
        (
            TableCanonicalizer()
            .canonicalize(
                aligned_physical=physical,
                structural=structural,
            )
        )


def test_column_names_normalize_unicode_metadata() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=3,
        rows=(
            row(
                0,
                (
                    physical_cell(
                        0,
                        text="Domaine",
                    ),
                    physical_cell(
                        1,
                        text="Ecosystéme",
                    ),
                    physical_cell(
                        2,
                        text="Composants",
                    ),
                ),
                header=True,
            ),
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=3,
        rows=(
            row(
                0,
                (
                    structural_cell(0),
                    structural_cell(1),
                    structural_cell(2),
                ),
                header=True,
            ),
        ),
    )

    result = (
        TableCanonicalizer()
        .canonicalize(
            aligned_physical=physical,
            structural=structural,
        )
    )

    assert (
        result.table.column_names
        == (
            "Domaine",
            "Ecosystéme",
            "Composants",
        )
    )

    # Source-backed cell text itself is not rewritten.
    assert (
        result.table.rows[0]
        .cells[1]
        .text
        != "Ecosystéme"
    )
