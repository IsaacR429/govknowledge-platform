import pytest

from app.domain.enums.table import (
    TableCellTextStatus,
    TableEvidenceLayer,
    TableRowAlignmentKind,
    TableSpanKind,
)
from app.domain.models.table import (
    TableBBox,
    TableSourceSpan,
)
from app.domain.tables.alignment import (
    TableRowAligner,
    TableRowAlignmentError,
)
from app.domain.tables.evidence import (
    NormalizedTableCellEvidence,
    NormalizedTableEvidence,
    NormalizedTableRowEvidence,
)


def span(
    *,
    page: int,
    ref: str,
    kind: TableSpanKind,
    bbox,
    text=None,
):
    return TableSourceSpan(
        page_number=page,
        kind=kind,
        source_ref=ref,
        bbox=(
            None
            if bbox is None
            else TableBBox(
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
            )
        ),
        text=text,
    )


def semantic_cell(
    column: int,
    text: str,
    *,
    page: int = 1,
    cell_bbox=(0, 0, 100, 10),
    text_bbox=(5, 1, 50, 9),
):
    ref = (
        f"p{page}:c{column}:"
        f"{cell_bbox[1]}"
    )

    return NormalizedTableCellEvidence(
        column_start=column,
        column_end=column + 1,
        text_status=(
            TableCellTextStatus.OBSERVED
        ),
        text=text,
        source_spans=(
            span(
                page=page,
                ref=ref + ":geometry",
                kind=TableSpanKind.GEOMETRY,
                bbox=cell_bbox,
            ),
            span(
                page=page,
                ref=ref + ":text",
                kind=TableSpanKind.TEXT,
                bbox=text_bbox,
                text=text,
            ),
        ),
    )


def unavailable_cell(
    column: int,
):
    return NormalizedTableCellEvidence(
        column_start=column,
        column_end=column + 1,
        text_status=(
            TableCellTextStatus.UNAVAILABLE
        ),
        text=None,
    )


def physical_row(
    index: int,
    *,
    columns: int,
    semantic_columns,
    label: str,
    y0: float,
    y1: float,
    header: bool = False,
):
    cells = []

    for column in range(
        columns
    ):
        if column in semantic_columns:
            cells.append(
                semantic_cell(
                    column,
                    f"{label}-{column}",
                    cell_bbox=(
                        column * 100,
                        y0,
                        (column + 1) * 100,
                        y1,
                    ),
                    text_bbox=(
                        column * 100 + 5,
                        y0 + 1,
                        column * 100 + 60,
                        y1 - 1,
                    ),
                )
            )
        else:
            cells.append(
                unavailable_cell(
                    column
                )
            )

    return NormalizedTableRowEvidence(
        row_index=index,
        cells=tuple(
            cells
        ),
        source_spans=(
            span(
                page=1,
                ref=f"row:{index}",
                kind=(
                    TableSpanKind.GEOMETRY
                ),
                bbox=(
                    0,
                    y0,
                    columns * 100,
                    y1,
                ),
            ),
        ),
        source_header_signal=header,
    )


def structural_row(
    index: int,
    *,
    width: int,
    header: bool = False,
):
    return NormalizedTableRowEvidence(
        row_index=index,
        cells=tuple(
            unavailable_cell(
                column
            )
            for column in range(
                width
            )
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


def test_wrapped_cell_continuation_is_collapsed() -> None:
    first = NormalizedTableRowEvidence(
        row_index=0,
        cells=(
            semantic_cell(
                0,
                "Industrie",
                cell_bbox=(
                    0,
                    10,
                    100,
                    30,
                ),
                text_bbox=(
                    5,
                    11,
                    80,
                    19,
                ),
            ),
        ),
        source_header_signal=False,
    )

    second = NormalizedTableRowEvidence(
        row_index=1,
        cells=(
            semantic_cell(
                0,
                "productif",
                cell_bbox=(
                    5,
                    20,
                    95,
                    30,
                ),
                text_bbox=(
                    5,
                    21,
                    45,
                    29,
                ),
            ),
        ),
        source_header_signal=False,
    )

    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=1,
        rows=(
            first,
            second,
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=1,
        rows=(
            structural_row(
                0,
                width=1,
            ),
        ),
    )

    result = TableRowAligner().align(
        projected_physical=physical,
        structural=structural,
    )

    assert (
        result.aligned_physical
        .row_count
        == 1
    )

    assert (
        result.aligned_physical
        .rows[0]
        .cells[0]
        .text
        == "Industrie productif"
    )

    assert (
        result.groups[0].kind
        == TableRowAlignmentKind
        .WRAPPED_CONTINUATION
    )

    assert (
        result.groups[0]
        .physical_row_indexes
        == (
            0,
            1,
        )
    )


def test_adjacent_distinct_cells_are_not_wrapped() -> None:
    first = physical_row(
        0,
        columns=1,
        semantic_columns={0},
        label="Grandpublic",
        y0=10,
        y1=20,
    )

    second = physical_row(
        1,
        columns=1,
        semantic_columns={0},
        label="Professionnels",
        y0=20,
        y1=30,
    )

    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=1,
        rows=(
            first,
            second,
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=1,
        rows=(
            structural_row(
                0,
                width=1,
            ),
            structural_row(
                1,
                width=1,
            ),
        ),
    )

    result = TableRowAligner().align(
        projected_physical=physical,
        structural=structural,
    )

    assert (
        result.aligned_physical
        .row_count
        == 2
    )

    assert all(
        group.kind
        == TableRowAlignmentKind
        .ONE_TO_ONE
        for group in result.groups
    )


def test_sparse_structural_row_groups_two_physical_rows() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=(
            physical_row(
                0,
                columns=5,
                semantic_columns={0},
                label="Lot",
                y0=10,
                y1=20,
            ),
            physical_row(
                1,
                columns=5,
                semantic_columns={0},
                label="Envelope",
                y0=20,
                y1=30,
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
            structural_row(
                0,
                width=1,
            ),
        ),
    )

    result = TableRowAligner().align(
        projected_physical=physical,
        structural=structural,
    )

    assert (
        result.groups[0].kind
        == TableRowAlignmentKind
        .STRUCTURAL_GROUP
    )

    assert (
        result.groups[0]
        .physical_row_indexes
        == (
            0,
            1,
        )
    )

    assert (
        result.aligned_physical
        .rows[0]
        .cells[0]
        .text
        == "Lot-0 Envelope-0"
    )


def test_procurement_sequence_selects_correct_sparse_pair() -> None:
    masks = (
        {0},
        {0},
        {0, 1, 2, 3, 4},
        {0},
        {0},
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4},
        {0, 1, 2, 3, 4},
    )

    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=tuple(
            physical_row(
                index,
                columns=5,
                semantic_columns=mask,
                label=f"P{index}",
                y0=index * 10,
                y1=(index + 1) * 10,
            )
            for index, mask
            in enumerate(masks)
        ),
    )

    widths = (
        1,
        1,
        5,
        1,
        5,
        5,
        5,
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=5,
        rows=tuple(
            structural_row(
                index,
                width=width,
            )
            for index, width
            in enumerate(widths)
        ),
    )

    result = TableRowAligner().align(
        projected_physical=physical,
        structural=structural,
    )

    grouped = [
        group
        for group in result.groups
        if (
            group.kind
            == TableRowAlignmentKind
            .STRUCTURAL_GROUP
        )
    ]

    assert len(grouped) == 1

    assert (
        grouped[0]
        .physical_row_indexes
        == (
            3,
            4,
        )
    )


def test_sparse_rows_with_different_masks_do_not_group() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=(
            physical_row(
                0,
                columns=5,
                semantic_columns={0},
                label="A",
                y0=0,
                y1=10,
            ),
            physical_row(
                1,
                columns=5,
                semantic_columns={1},
                label="B",
                y0=10,
                y1=20,
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
            structural_row(
                0,
                width=1,
            ),
        ),
    )

    with pytest.raises(
        TableRowAlignmentError,
        match="No valid",
    ):
        TableRowAligner().align(
            projected_physical=physical,
            structural=structural,
        )


def test_ambiguous_sparse_group_fails_closed() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=tuple(
            physical_row(
                index,
                columns=5,
                semantic_columns={0},
                label=f"P{index}",
                y0=index * 10,
                y1=(index + 1) * 10,
            )
            for index in range(3)
        ),
    )

    structural = table(
        ref="structural",
        layer=(
            TableEvidenceLayer.STRUCTURAL
        ),
        columns=5,
        rows=(
            structural_row(
                0,
                width=1,
            ),
            structural_row(
                1,
                width=1,
            ),
        ),
    )

    with pytest.raises(
        TableRowAlignmentError,
        match="Ambiguous",
    ):
        TableRowAligner().align(
            projected_physical=physical,
            structural=structural,
        )


def test_structural_header_requires_physical_header() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=3,
        rows=(
            physical_row(
                0,
                columns=3,
                semantic_columns={
                    0,
                    1,
                    2,
                },
                label="row",
                y0=0,
                y1=10,
                header=False,
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
            structural_row(
                0,
                width=3,
                header=True,
            ),
        ),
    )

    with pytest.raises(
        TableRowAlignmentError,
        match="No valid",
    ):
        TableRowAligner().align(
            projected_physical=physical,
            structural=structural,
        )


def test_one_to_one_alignment_preserves_rows() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=3,
        rows=(
            physical_row(
                0,
                columns=3,
                semantic_columns={
                    0,
                    1,
                    2,
                },
                label="header",
                y0=0,
                y1=10,
                header=True,
            ),
            physical_row(
                1,
                columns=3,
                semantic_columns={
                    2,
                },
                label="data",
                y0=10,
                y1=20,
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
            structural_row(
                0,
                width=3,
                header=True,
            ),
            structural_row(
                1,
                width=3,
            ),
        ),
    )

    result = TableRowAligner().align(
        projected_physical=physical,
        structural=structural,
    )

    assert (
        result.aligned_physical
        .row_count
        == 2
    )

    assert (
        result.groups[0].kind
        == TableRowAlignmentKind
        .ONE_TO_ONE
    )

    assert (
        result.groups[1].kind
        == TableRowAlignmentKind
        .ONE_TO_ONE
    )


def test_merged_row_preserves_source_provenance() -> None:
    physical = table(
        ref="physical",
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=(
            physical_row(
                0,
                columns=5,
                semantic_columns={0},
                label="Lot",
                y0=10,
                y1=20,
            ),
            physical_row(
                1,
                columns=5,
                semantic_columns={0},
                label="Envelope",
                y0=20,
                y1=30,
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
            structural_row(
                0,
                width=1,
            ),
        ),
    )

    result = TableRowAligner().align(
        projected_physical=physical,
        structural=structural,
    )

    merged = (
        result.aligned_physical
        .rows[0]
    )

    assert len(
        merged.source_spans
    ) == 2

    assert len(
        merged.cells[0]
        .source_spans
    ) == 4
