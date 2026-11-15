import pytest

from app.domain.enums.table import (
    TableCellValueSource,
    TableContinuationDecision,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    CanonicalTable,
    CanonicalTableCell,
    CanonicalTableRow,
    TableBBox,
    TableContinuationEvidence,
    TableContinuationResult,
    TableSourceSpan,
)


def text_span(
    *,
    page: int,
    ref: str,
    text: str,
) -> TableSourceSpan:
    return TableSourceSpan(
        page_number=page,
        kind=TableSpanKind.TEXT,
        source_ref=ref,
        text=text,
    )


def test_normal_three_column_table() -> None:
    row = CanonicalTableRow(
        row_index=0,
        role=TableRowRole.DATA,
        cells=(
            CanonicalTableCell(
                column_start=0,
                column_end=1,
                text="Productif",
                value_source=(
                    TableCellValueSource.SOURCE_TEXT
                ),
                origin_spans=(
                    text_span(
                        page=9,
                        ref="tag:row1:c0",
                        text="Productif",
                    ),
                ),
            ),
            CanonicalTableCell(
                column_start=1,
                column_end=2,
                text="Infrastructure",
                value_source=(
                    TableCellValueSource.SOURCE_TEXT
                ),
                origin_spans=(
                    text_span(
                        page=9,
                        ref="tag:row1:c1",
                        text="Infrastructure",
                    ),
                ),
            ),
            CanonicalTableCell(
                column_start=2,
                column_end=3,
                text=(
                    "Equipements réseaux "
                    "et support informatique"
                ),
                value_source=(
                    TableCellValueSource.SOURCE_TEXT
                ),
                origin_spans=(
                    text_span(
                        page=9,
                        ref="tag:row1:c2",
                        text=(
                            "Equipements réseaux "
                            "et support informatique"
                        ),
                    ),
                ),
            ),
        ),
        source_spans=(
            TableSourceSpan(
                page_number=9,
                kind=TableSpanKind.STRUCTURE,
                source_ref="tag:row:xref:1942",
            ),
        ),
    )

    table = CanonicalTable(
        column_count=3,
        column_names=(
            "Domaine",
            "Ecosystème",
            "Composants",
        ),
        hierarchy_columns=(
            0,
            1,
        ),
        rows=(
            row,
        ),
    )

    assert table.page_numbers == (9,)
    assert len(table.data_rows) == 1


def test_inherited_cell_preserves_previous_page_origin() -> None:
    inherited_domain = CanonicalTableCell(
        column_start=0,
        column_end=1,
        text="Institutionnel",
        value_source=TableCellValueSource.INHERITED,
        origin_spans=(
            text_span(
                page=9,
                ref="tag:xref:1994",
                text="Institutionnel",
            ),
        ),
    )

    inherited_ecosystem = CanonicalTableCell(
        column_start=1,
        column_end=2,
        text="Environnement",
        value_source=TableCellValueSource.INHERITED,
        origin_spans=(
            text_span(
                page=9,
                ref="tag:xref:1997",
                text="Environnement",
            ),
        ),
    )

    component = CanonicalTableCell(
        column_start=2,
        column_end=3,
        text=(
            "Cadre politique publique "
            "d’accompagnement"
        ),
        value_source=TableCellValueSource.SOURCE_TEXT,
        origin_spans=(
            text_span(
                page=10,
                ref="tag:xref:2034",
                text=(
                    "Cadre politique publique "
                    "d’accompagnement"
                ),
            ),
        ),
    )

    row = CanonicalTableRow(
        row_index=0,
        role=TableRowRole.DATA,
        cells=(
            inherited_domain,
            inherited_ecosystem,
            component,
        ),
        source_spans=(
            TableSourceSpan(
                page_number=10,
                kind=TableSpanKind.STRUCTURE,
                source_ref="tag:row:xref:2030",
            ),
        ),
    )

    assert row.page_numbers == (10,)

    assert (
        inherited_domain.origin_spans[0].page_number
        == 9
    )

    assert inherited_domain.inherited is True

    assert (
        component.origin_spans[0].page_number
        == 10
    )


def test_section_can_span_all_columns() -> None:
    section = CanonicalTableRow(
        row_index=0,
        role=TableRowRole.SECTION,
        section_key="LOT_1",
        cells=(
            CanonicalTableCell(
                column_start=0,
                column_end=5,
                text="Lot 1",
                value_source=(
                    TableCellValueSource.SOURCE_TEXT
                ),
                origin_spans=(
                    text_span(
                        page=3,
                        ref="tag:xref:729",
                        text="Lot 1",
                    ),
                ),
            ),
        ),
        source_spans=(
            TableSourceSpan(
                page_number=3,
                kind=TableSpanKind.STRUCTURE,
                source_ref="tag:row:xref:728",
            ),
        ),
    )

    table = CanonicalTable(
        column_count=5,
        rows=(
            section,
        ),
    )

    assert (
        table.rows[0]
        .cells[0]
        .column_span
        == 5
    )


def test_row_rejects_overlapping_cells() -> None:
    with pytest.raises(
        ValueError,
        match="cannot overlap",
    ):
        CanonicalTableRow(
            row_index=0,
            role=TableRowRole.DATA,
            cells=(
                CanonicalTableCell(
                    column_start=0,
                    column_end=2,
                    text="A",
                    value_source=(
                        TableCellValueSource.DERIVED
                    ),
                ),
                CanonicalTableCell(
                    column_start=1,
                    column_end=3,
                    text="B",
                    value_source=(
                        TableCellValueSource.DERIVED
                    ),
                ),
            ),
        )


def test_source_text_requires_provenance() -> None:
    with pytest.raises(
        ValueError,
        match="SOURCE_TEXT",
    ):
        CanonicalTableCell(
            column_start=0,
            column_end=1,
            text="Untraceable",
            value_source=(
                TableCellValueSource.SOURCE_TEXT
            ),
        )


def test_inherited_value_requires_origin() -> None:
    with pytest.raises(
        ValueError,
        match="Inherited",
    ):
        CanonicalTableCell(
            column_start=0,
            column_end=1,
            text="Institutionnel",
            value_source=(
                TableCellValueSource.INHERITED
            ),
        )


def test_table_rejects_cell_outside_schema() -> None:
    row = CanonicalTableRow(
        row_index=0,
        role=TableRowRole.SECTION,
        cells=(
            CanonicalTableCell(
                column_start=0,
                column_end=4,
                text="section",
                value_source=(
                    TableCellValueSource.DERIVED
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        CanonicalTable(
            column_count=3,
            rows=(
                row,
            ),
        )


def test_bbox_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        TableBBox(
            x0=100.0,
            y0=20.0,
            x1=50.0,
            y1=30.0,
        )


def test_continuation_result_preserves_reason() -> None:
    evidence = TableContinuationEvidence(
        left_source_ref="table:p9",
        right_source_ref="table:p10",
        left_page=9,
        right_page=10,
        same_column_count=True,
        compatible_column_geometry=True,
        repeated_header=True,
        headerless_continuation=False,
        row_key_progression=False,
        left_has_active_hierarchy=True,
        right_starts_with_inherited_cells=True,
    )

    result = TableContinuationResult(
        decision=(
            TableContinuationDecision.CONTINUE
        ),
        evidence=evidence,
        reasons=(
            "Adjacent pages with compatible schema "
            "and repeated header.",
            "Right table continues active hierarchy.",
        ),
    )

    assert (
        result.decision
        == TableContinuationDecision.CONTINUE
    )

    assert len(result.reasons) == 2
