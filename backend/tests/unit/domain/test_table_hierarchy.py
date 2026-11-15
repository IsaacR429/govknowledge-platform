import pytest

from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    CanonicalTable,
    CanonicalTableCell,
    CanonicalTableRow,
    TableSourceSpan,
)
from app.domain.tables.hierarchy import (
    TableHierarchyResolutionError,
    TableHierarchyResolver,
    TableHierarchyState,
)


def span(
    *,
    page: int,
    ref: str,
    text: str | None = None,
    kind: TableSpanKind = TableSpanKind.TEXT,
) -> TableSourceSpan:
    return TableSourceSpan(
        page_number=page,
        kind=kind,
        source_ref=ref,
        text=text,
    )


def source_cell(
    *,
    column: int,
    text: str,
    page: int,
    ref: str,
) -> CanonicalTableCell:
    return CanonicalTableCell(
        column_start=column,
        column_end=column + 1,
        text=text,
        value_source=(
            TableCellValueSource.SOURCE_TEXT
        ),
        origin_spans=(
            span(
                page=page,
                ref=ref,
                text=text,
            ),
        ),
    )


def blank_cell(
    *,
    column: int,
    page: int,
    ref: str,
) -> CanonicalTableCell:
    return CanonicalTableCell(
        column_start=column,
        column_end=column + 1,
        text="",
        value_source=(
            TableCellValueSource.SOURCE_TEXT
        ),
        structure_spans=(
            span(
                page=page,
                ref=ref,
                kind=TableSpanKind.STRUCTURE,
            ),
        ),
    )


def data_row(
    *,
    index: int,
    page: int,
    cells: tuple[
        CanonicalTableCell,
        ...
    ],
) -> CanonicalTableRow:
    return CanonicalTableRow(
        row_index=index,
        role=TableRowRole.DATA,
        cells=cells,
        source_spans=(
            span(
                page=page,
                ref=f"row:{index}",
                kind=TableSpanKind.STRUCTURE,
            ),
        ),
    )


def test_inherits_outer_parent_within_table() -> None:
    table = CanonicalTable(
        column_count=3,
        hierarchy_columns=(0, 1),
        rows=(
            data_row(
                index=0,
                page=1,
                cells=(
                    source_cell(
                        column=0,
                        text="Domain A",
                        page=1,
                        ref="d-a",
                    ),
                    source_cell(
                        column=1,
                        text="Group A",
                        page=1,
                        ref="g-a",
                    ),
                    source_cell(
                        column=2,
                        text="Value 1",
                        page=1,
                        ref="v-1",
                    ),
                ),
            ),
            data_row(
                index=1,
                page=1,
                cells=(
                    blank_cell(
                        column=0,
                        page=1,
                        ref="blank-d",
                    ),
                    source_cell(
                        column=1,
                        text="Group B",
                        page=1,
                        ref="g-b",
                    ),
                    source_cell(
                        column=2,
                        text="Value 2",
                        page=1,
                        ref="v-2",
                    ),
                ),
            ),
        ),
    )

    result = (
        TableHierarchyResolver()
        .resolve_table(table)
    )

    row = result.table.rows[1]

    assert row.cells[0].text == "Domain A"
    assert row.cells[0].inherited is True

    assert (
        row.cells[0]
        .origin_spans[0]
        .source_ref
        == "d-a"
    )

    assert row.cells[1].text == "Group B"

    assert result.inherited_cell_count == 1


def test_inherits_multiple_hierarchy_levels() -> None:
    table = CanonicalTable(
        column_count=3,
        hierarchy_columns=(0, 1),
        rows=(
            data_row(
                index=0,
                page=1,
                cells=(
                    source_cell(
                        column=0,
                        text="Domain A",
                        page=1,
                        ref="d-a",
                    ),
                    source_cell(
                        column=1,
                        text="Group A",
                        page=1,
                        ref="g-a",
                    ),
                    source_cell(
                        column=2,
                        text="Value 1",
                        page=1,
                        ref="v-1",
                    ),
                ),
            ),
            data_row(
                index=1,
                page=1,
                cells=(
                    blank_cell(
                        column=0,
                        page=1,
                        ref="blank-d",
                    ),
                    blank_cell(
                        column=1,
                        page=1,
                        ref="blank-g",
                    ),
                    source_cell(
                        column=2,
                        text="Value 2",
                        page=1,
                        ref="v-2",
                    ),
                ),
            ),
        ),
    )

    result = (
        TableHierarchyResolver()
        .resolve_table(table)
    )

    row = result.table.rows[1]

    assert row.cells[0].text == "Domain A"
    assert row.cells[1].text == "Group A"

    assert row.cells[0].inherited is True
    assert row.cells[1].inherited is True

    assert result.inherited_cell_count == 2


def test_new_parent_clears_deeper_state() -> None:
    table = CanonicalTable(
        column_count=3,
        hierarchy_columns=(0, 1),
        rows=(
            data_row(
                index=0,
                page=1,
                cells=(
                    source_cell(
                        column=0,
                        text="Domain A",
                        page=1,
                        ref="d-a",
                    ),
                    source_cell(
                        column=1,
                        text="Group A",
                        page=1,
                        ref="g-a",
                    ),
                    source_cell(
                        column=2,
                        text="Value 1",
                        page=1,
                        ref="v-1",
                    ),
                ),
            ),
            data_row(
                index=1,
                page=1,
                cells=(
                    source_cell(
                        column=0,
                        text="Domain B",
                        page=1,
                        ref="d-b",
                    ),
                    blank_cell(
                        column=1,
                        page=1,
                        ref="blank-g",
                    ),
                    source_cell(
                        column=2,
                        text="Value 2",
                        page=1,
                        ref="v-2",
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        TableHierarchyResolutionError,
        match="no active parent",
    ):
        (
            TableHierarchyResolver()
            .resolve_table(table)
        )


def test_cross_page_inheritance_uses_initial_state() -> None:
    domain = source_cell(
        column=0,
        text="Domain A",
        page=9,
        ref="page9-domain",
    )

    group = source_cell(
        column=1,
        text="Group A",
        page=9,
        ref="page9-group",
    )

    initial_state = TableHierarchyState(
        active_cells=(
            (0, domain),
            (1, group),
        )
    )

    table = CanonicalTable(
        column_count=3,
        hierarchy_columns=(0, 1),
        rows=(
            data_row(
                index=0,
                page=10,
                cells=(
                    blank_cell(
                        column=0,
                        page=10,
                        ref="page10-blank-domain",
                    ),
                    blank_cell(
                        column=1,
                        page=10,
                        ref="page10-blank-group",
                    ),
                    source_cell(
                        column=2,
                        text="Value 2",
                        page=10,
                        ref="page10-value",
                    ),
                ),
            ),
        ),
    )

    result = (
        TableHierarchyResolver()
        .resolve_table(
            table,
            initial_state=initial_state,
        )
    )

    row = result.table.rows[0]

    assert row.page_numbers == (10,)

    assert row.cells[0].text == "Domain A"
    assert row.cells[1].text == "Group A"

    assert (
        row.cells[0]
        .origin_spans[0]
        .page_number
        == 9
    )

    assert (
        row.cells[1]
        .origin_spans[0]
        .page_number
        == 9
    )

    assert (
        row.cells[0]
        .structure_spans[0]
        .page_number
        == 10
    )

    assert (
        row.cells[1]
        .structure_spans[0]
        .page_number
        == 10
    )


def test_header_does_not_destroy_active_state() -> None:
    domain = source_cell(
        column=0,
        text="Domain A",
        page=1,
        ref="domain-a",
    )

    group = source_cell(
        column=1,
        text="Group A",
        page=1,
        ref="group-a",
    )

    header = CanonicalTableRow(
        row_index=0,
        role=TableRowRole.HEADER,
        cells=(
            source_cell(
                column=0,
                text="Domain",
                page=2,
                ref="h0",
            ),
            source_cell(
                column=1,
                text="Group",
                page=2,
                ref="h1",
            ),
            source_cell(
                column=2,
                text="Value",
                page=2,
                ref="h2",
            ),
        ),
    )

    continuation = data_row(
        index=1,
        page=2,
        cells=(
            blank_cell(
                column=0,
                page=2,
                ref="blank-domain",
            ),
            blank_cell(
                column=1,
                page=2,
                ref="blank-group",
            ),
            source_cell(
                column=2,
                text="Value 2",
                page=2,
                ref="value-2",
            ),
        ),
    )

    table = CanonicalTable(
        column_count=3,
        hierarchy_columns=(0, 1),
        rows=(
            header,
            continuation,
        ),
    )

    result = (
        TableHierarchyResolver()
        .resolve_table(
            table,
            initial_state=(
                TableHierarchyState(
                    active_cells=(
                        (0, domain),
                        (1, group),
                    )
                )
            ),
        )
    )

    assert (
        result.table.rows[1]
        .cells[0]
        .text
        == "Domain A"
    )

    assert (
        result.table.rows[1]
        .cells[1]
        .text
        == "Group A"
    )


def test_missing_initial_parent_fails_closed() -> None:
    table = CanonicalTable(
        column_count=3,
        hierarchy_columns=(0, 1),
        rows=(
            data_row(
                index=0,
                page=1,
                cells=(
                    blank_cell(
                        column=0,
                        page=1,
                        ref="blank-domain",
                    ),
                    blank_cell(
                        column=1,
                        page=1,
                        ref="blank-group",
                    ),
                    source_cell(
                        column=2,
                        text="Value",
                        page=1,
                        ref="value",
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        TableHierarchyResolutionError,
        match="no active parent",
    ):
        (
            TableHierarchyResolver()
            .resolve_table(table)
        )


def test_non_hierarchy_column_is_not_filled() -> None:
    table = CanonicalTable(
        column_count=3,
        hierarchy_columns=(0, 1),
        rows=(
            data_row(
                index=0,
                page=1,
                cells=(
                    source_cell(
                        column=0,
                        text="Domain A",
                        page=1,
                        ref="domain",
                    ),
                    source_cell(
                        column=1,
                        text="Group A",
                        page=1,
                        ref="group",
                    ),
                    blank_cell(
                        column=2,
                        page=1,
                        ref="blank-value",
                    ),
                ),
            ),
        ),
    )

    result = (
        TableHierarchyResolver()
        .resolve_table(table)
    )

    assert (
        result.table.rows[0]
        .cells[2]
        .text
        == ""
    )


def test_original_table_remains_unchanged() -> None:
    first = data_row(
        index=0,
        page=1,
        cells=(
            source_cell(
                column=0,
                text="Domain A",
                page=1,
                ref="domain",
            ),
            source_cell(
                column=1,
                text="Group A",
                page=1,
                ref="group",
            ),
            source_cell(
                column=2,
                text="Value 1",
                page=1,
                ref="value-1",
            ),
        ),
    )

    second = data_row(
        index=1,
        page=1,
        cells=(
            blank_cell(
                column=0,
                page=1,
                ref="blank-domain",
            ),
            blank_cell(
                column=1,
                page=1,
                ref="blank-group",
            ),
            source_cell(
                column=2,
                text="Value 2",
                page=1,
                ref="value-2",
            ),
        ),
    )

    table = CanonicalTable(
        column_count=3,
        hierarchy_columns=(0, 1),
        rows=(
            first,
            second,
        ),
    )

    (
        TableHierarchyResolver()
        .resolve_table(table)
    )

    assert table.rows[1].cells[0].text == ""
    assert table.rows[1].cells[1].text == ""


def test_final_state_contains_latest_active_values() -> None:
    table = CanonicalTable(
        column_count=3,
        hierarchy_columns=(0, 1),
        rows=(
            data_row(
                index=0,
                page=1,
                cells=(
                    source_cell(
                        column=0,
                        text="Domain A",
                        page=1,
                        ref="d-a",
                    ),
                    source_cell(
                        column=1,
                        text="Group A",
                        page=1,
                        ref="g-a",
                    ),
                    source_cell(
                        column=2,
                        text="Value",
                        page=1,
                        ref="v",
                    ),
                ),
            ),
            data_row(
                index=1,
                page=1,
                cells=(
                    blank_cell(
                        column=0,
                        page=1,
                        ref="blank-d",
                    ),
                    source_cell(
                        column=1,
                        text="Group B",
                        page=1,
                        ref="g-b",
                    ),
                    source_cell(
                        column=2,
                        text="Value 2",
                        page=1,
                        ref="v2",
                    ),
                ),
            ),
        ),
    )

    result = (
        TableHierarchyResolver()
        .resolve_table(table)
    )

    assert (
        result.final_state
        .get(0)
        .text
        == "Domain A"
    )

    assert (
        result.final_state
        .get(1)
        .text
        == "Group B"
    )
