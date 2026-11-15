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
from app.domain.tables.structure import (
    TableStructureResolutionError,
    TableStructureResolver,
)


def span(
    *,
    page: int,
    ref: str,
    text: str,
):
    return TableSourceSpan(
        page_number=page,
        kind=TableSpanKind.TEXT,
        source_ref=ref,
        text=text,
    )


def cell(
    *,
    page: int,
    row: int,
    column: int,
    text: str,
):
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
                ref=(
                    f"p{page}:r{row}:"
                    f"c{column}"
                ),
                text=text,
            ),
        )
        if text
        else (),
        structure_spans=(),
    )


def row(
    *,
    page: int,
    index: int,
    values,
    role=TableRowRole.DATA,
):
    return CanonicalTableRow(
        row_index=index,
        role=role,
        cells=tuple(
            cell(
                page=page,
                row=index,
                column=column,
                text=value,
            )
            for column, value
            in enumerate(values)
        ),
        source_spans=(),
    )


def table(
    *,
    page: int,
    rows,
):
    return CanonicalTable(
        column_count=5,
        rows=tuple(rows),
        column_names=(),
        hierarchy_columns=(),
        source_table_refs=(
            f"table:p{page}",
        ),
    )


def test_procurement_header_is_promoted():
    source = table(
        page=3,
        rows=(
            row(
                page=3,
                index=0,
                values=(
                    "N°",
                    "Soumissionnaires",
                    "Montants lus en FCFA",
                    "Montants corrigés en FCFA",
                    "Observations",
                ),
            ),
        ),
    )

    result = (
        TableStructureResolver()
        .resolve_table(
            source
        )
    )

    assert (
        result.table.rows[0].role
        == TableRowRole.HEADER
    )

    assert (
        result.promoted_header_count
        == 1
    )


def test_lot_section_and_rows_receive_keys():
    source = table(
        page=3,
        rows=(
            row(
                page=3,
                index=0,
                values=(
                    "Lot 1 : prestation de pauses café",
                    "",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=3,
                index=1,
                values=(
                    "01",
                    "Entreprise A",
                    "10",
                    "10",
                    "Conforme",
                ),
            ),
            row(
                page=3,
                index=2,
                values=(
                    "02",
                    "Entreprise B",
                    "11",
                    "11",
                    "Conforme",
                ),
            ),
            row(
                page=3,
                index=3,
                values=(
                    "03",
                    "Entreprise C",
                    "12",
                    "12",
                    "Conforme",
                ),
            ),
        ),
    )

    result = (
        TableStructureResolver()
        .resolve_table(
            source
        )
    )

    rows = (
        result.table.rows
    )

    assert (
        rows[0].role
        == TableRowRole.SECTION
    )

    assert (
        rows[0].section_key
        == "LOT_1"
    )

    assert (
        rows[1].logical_key
        == "LOT_1/01"
    )

    assert (
        rows[2].logical_key
        == "LOT_1/02"
    )

    assert (
        rows[3].logical_key
        == "LOT_1/03"
    )

    assert (
        result.final_state.section_key
        == "LOT_1"
    )

    assert (
        result.final_state.last_ordinal
        == 3
    )


def test_cross_page_state_continues_04_to_06():
    resolver = (
        TableStructureResolver()
    )

    page3 = table(
        page=3,
        rows=(
            row(
                page=3,
                index=0,
                values=(
                    "Lot 1 : prestations",
                    "",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=3,
                index=1,
                values=(
                    "01",
                    "A",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=3,
                index=2,
                values=(
                    "02",
                    "B",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=3,
                index=3,
                values=(
                    "03",
                    "C",
                    "",
                    "",
                    "",
                ),
            ),
        ),
    )

    first = (
        resolver.resolve_table(
            page3
        )
    )

    page4 = table(
        page=4,
        rows=(
            row(
                page=4,
                index=0,
                values=(
                    "04",
                    "D",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=4,
                index=1,
                values=(
                    "05",
                    "E",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=4,
                index=2,
                values=(
                    "06",
                    "F",
                    "",
                    "",
                    "",
                ),
            ),
        ),
    )

    second = (
        resolver.resolve_table(
            page4,
            initial_state=(
                first.final_state
            ),
        )
    )

    assert tuple(
        row.logical_key
        for row in second.table.rows
    ) == (
        "LOT_1/04",
        "LOT_1/05",
        "LOT_1/06",
    )


def test_new_lot_resets_numbering():
    resolver = (
        TableStructureResolver()
    )

    first = (
        resolver.resolve_table(
            table(
                page=3,
                rows=(
                    row(
                        page=3,
                        index=0,
                        values=(
                            "Lot 1 : test",
                            "",
                            "",
                            "",
                            "",
                        ),
                    ),
                    row(
                        page=3,
                        index=1,
                        values=(
                            "01",
                            "A",
                            "",
                            "",
                            "",
                        ),
                    ),
                ),
            )
        )
    )

    second = (
        resolver.resolve_table(
            table(
                page=4,
                rows=(
                    row(
                        page=4,
                        index=0,
                        values=(
                            "02",
                            "B",
                            "",
                            "",
                            "",
                        ),
                    ),
                    row(
                        page=4,
                        index=1,
                        values=(
                            "Lot 2 : autre prestation",
                            "",
                            "",
                            "",
                            "",
                        ),
                    ),
                    row(
                        page=4,
                        index=2,
                        values=(
                            "01",
                            "C",
                            "",
                            "",
                            "",
                        ),
                    ),
                    row(
                        page=4,
                        index=3,
                        values=(
                            "02",
                            "D",
                            "",
                            "",
                            "",
                        ),
                    ),
                ),
            ),
            initial_state=(
                first.final_state
            ),
        )
    )

    rows = (
        second.table.rows
    )

    assert (
        rows[0].logical_key
        == "LOT_1/02"
    )

    assert (
        rows[1].role
        == TableRowRole.SECTION
    )

    assert (
        rows[1].section_key
        == "LOT_2"
    )

    assert (
        rows[2].logical_key
        == "LOT_2/01"
    )

    assert (
        rows[3].logical_key
        == "LOT_2/02"
    )


def test_number_gap_fails_closed():
    source = table(
        page=3,
        rows=(
            row(
                page=3,
                index=0,
                values=(
                    "Lot 1 : test",
                    "",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=3,
                index=1,
                values=(
                    "01",
                    "A",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=3,
                index=2,
                values=(
                    "03",
                    "C",
                    "",
                    "",
                    "",
                ),
            ),
        ),
    )

    with pytest.raises(
        TableStructureResolutionError,
        match="expected 02",
    ):
        (
            TableStructureResolver()
            .resolve_table(
                source
            )
        )


def test_numbered_row_without_required_carried_section_fails():
    source = table(
        page=4,
        rows=(
            row(
                page=4,
                index=0,
                values=(
                    "04",
                    "Continuation",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=4,
                index=1,
                values=(
                    "Lot 2 : test",
                    "",
                    "",
                    "",
                    "",
                ),
            ),
        ),
    )

    with pytest.raises(
        TableStructureResolutionError,
        match="has no active section",
    ):
        (
            TableStructureResolver()
            .resolve_table(
                source
            )
        )


def test_summary_closes_sequence():
    source = table(
        page=4,
        rows=(
            row(
                page=4,
                index=0,
                values=(
                    "Lot 2 : test",
                    "",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=4,
                index=1,
                values=(
                    "01",
                    "A",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=4,
                index=2,
                values=(
                    "ATTRIBUTAIRES PROVISOIRES",
                    "Lot 1 : A ; Lot 2 : B",
                    "",
                    "",
                    "",
                ),
            ),
        ),
    )

    result = (
        TableStructureResolver()
        .resolve_table(
            source
        )
    )

    summary = (
        result.table.rows[2]
    )

    assert (
        summary.role
        == TableRowRole.SUMMARY
    )

    assert (
        result.summary_count
        == 1
    )

    assert (
        result.final_state.section_key
        is None
    )

    assert (
        result.final_state.last_ordinal
        is None
    )

    assert (
        result.final_state.terminated
        is True
    )


def test_generic_numeric_table_is_not_overclassified():
    source = table(
        page=1,
        rows=(
            row(
                page=1,
                index=0,
                values=(
                    "01",
                    "Generic row",
                    "",
                    "",
                    "",
                ),
            ),
        ),
    )

    result = (
        TableStructureResolver()
        .resolve_table(
            source
        )
    )

    output = (
        result.table.rows[0]
    )

    assert (
        output.logical_key
        is None
    )

    assert (
        output.section_key
        is None
    )

    assert (
        output.role
        == TableRowRole.DATA
    )

def test_incidental_attributaire_provisoire_phrase_is_not_summary():
    source = table(
        page=3,
        rows=(
            row(
                page=3,
                index=0,
                values=(
                    (
                        "RECTIFICATIF PORTANT SUR LE NOM "
                        "DE L’ATTRIBUTAIRE PROVISOIRE "
                        "AU LOT 2"
                    ),
                    "",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=3,
                index=1,
                values=(
                    "Lot 1 : prestation",
                    "",
                    "",
                    "",
                    "",
                ),
            ),
            row(
                page=3,
                index=2,
                values=(
                    "01",
                    "Entreprise A",
                    "",
                    "",
                    "",
                ),
            ),
        ),
    )

    result = (
        TableStructureResolver()
        .resolve_table(
            source
        )
    )

    preamble = (
        result.table.rows[0]
    )

    assert (
        preamble.role
        == TableRowRole.DATA
    )

    assert (
        result.summary_count
        == 0
    )

    assert (
        result.table.rows[1].role
        == TableRowRole.SECTION
    )

    assert (
        result.table.rows[2].logical_key
        == "LOT_1/01"
    )
