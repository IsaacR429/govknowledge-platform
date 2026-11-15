from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    CanonicalTable,
    CanonicalTableCell,
    CanonicalTableRow,
    TableBBox,
    TableSourceSpan,
)
from app.domain.tables.continuation_evidence import (
    TableContinuationEvidenceBuildError,
    TableContinuationEvidenceBuilder,
)
from app.domain.tables.hierarchy import (
    TableHierarchyState,
)

import pytest


def source_span(
    *,
    page: int,
    ref: str,
    text: str,
    x0: float,
    x1: float,
):
    return TableSourceSpan(
        page_number=page,
        kind=TableSpanKind.TEXT,
        source_ref=ref,
        bbox=TableBBox(
            x0=x0,
            y0=0.0,
            x1=x1,
            y1=10.0,
        ),
        text=text,
    )


def geometry_span(
    *,
    page: int,
    ref: str,
    x0: float,
    x1: float,
):
    return TableSourceSpan(
        page_number=page,
        kind=TableSpanKind.GEOMETRY,
        source_ref=ref,
        bbox=TableBBox(
            x0=x0,
            y0=0.0,
            x1=x1,
            y1=10.0,
        ),
    )


def structure_span(
    *,
    page: int,
    ref: str,
):
    return TableSourceSpan(
        page_number=page,
        kind=TableSpanKind.STRUCTURE,
        source_ref=ref,
    )


def text_cell(
    *,
    page: int,
    ref: str,
    column: int,
    text: str,
    x0: float,
    x1: float,
):
    return CanonicalTableCell(
        column_start=column,
        column_end=column + 1,
        text=text,
        value_source=(
            TableCellValueSource.SOURCE_TEXT
        ),
        origin_spans=(
            source_span(
                page=page,
                ref=f"{ref}:text",
                text=text,
                x0=x0,
                x1=x1,
            ),
        ),
        structure_spans=(
            geometry_span(
                page=page,
                ref=f"{ref}:geometry",
                x0=x0,
                x1=x1,
            ),
        ),
    )


def blank_cell(
    *,
    page: int,
    ref: str,
    column: int,
    x0: float,
    x1: float,
):
    return CanonicalTableCell(
        column_start=column,
        column_end=column + 1,
        text="",
        value_source=(
            TableCellValueSource.DERIVED
        ),
        origin_spans=(),
        structure_spans=(
            geometry_span(
                page=page,
                ref=f"{ref}:geometry",
                x0=x0,
                x1=x1,
            ),
        ),
    )


def make_table(
    *,
    page: int,
    ref: str,
    names=(
        "Domaine",
        "Ecosysteme",
        "Composants",
    ),
    data=(
        "Institutionnel",
        "Environnement",
        "Cadre",
    ),
    hierarchy_columns=(0, 1),
    logical_key=None,
    geometry=(
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 100.0),
    ),
):
    rows = []

    row_index = 0

    if names:
        rows.append(
            CanonicalTableRow(
                row_index=row_index,
                role=TableRowRole.HEADER,
                cells=tuple(
                    text_cell(
                        page=page,
                        ref=f"{ref}:h:{column}",
                        column=column,
                        text=name,
                        x0=geometry[column][0],
                        x1=geometry[column][1],
                    )
                    for column, name
                    in enumerate(names)
                ),
                source_spans=(
                    structure_span(
                        page=page,
                        ref=f"{ref}:header",
                    ),
                ),
            )
        )

        row_index += 1

    cells = []

    for column, value in enumerate(
        data
    ):
        if value:
            cells.append(
                text_cell(
                    page=page,
                    ref=f"{ref}:d:{column}",
                    column=column,
                    text=value,
                    x0=geometry[column][0],
                    x1=geometry[column][1],
                )
            )
        else:
            cells.append(
                blank_cell(
                    page=page,
                    ref=f"{ref}:d:{column}",
                    column=column,
                    x0=geometry[column][0],
                    x1=geometry[column][1],
                )
            )

    rows.append(
        CanonicalTableRow(
            row_index=row_index,
            role=TableRowRole.DATA,
            cells=tuple(cells),
            source_spans=(
                structure_span(
                    page=page,
                    ref=f"{ref}:data",
                ),
            ),
            logical_key=logical_key,
        )
    )

    return CanonicalTable(
        column_count=3,
        rows=tuple(rows),
        column_names=tuple(names),
        hierarchy_columns=(
            tuple(
                hierarchy_columns
            )
        ),
        source_table_refs=(
            ref,
        ),
    )


def test_repeated_header_and_geometry_are_derived():
    left = make_table(
        page=9,
        ref="left",
    )

    right = make_table(
        page=10,
        ref="right",
        data=(
            "",
            "",
            "Continuation",
        ),
    )

    state = TableHierarchyState.from_dict(
        {
            0: left.rows[1].cells[0],
            1: left.rows[1].cells[1],
        }
    )

    evidence = (
        TableContinuationEvidenceBuilder()
        .build(
            left=left,
            right=right,
            left_state=state,
        )
    )

    assert evidence.left_page == 9
    assert evidence.right_page == 10
    assert evidence.same_column_count
    assert (
        evidence.compatible_column_geometry
    )
    assert evidence.repeated_header
    assert not (
        evidence.headerless_continuation
    )
    assert not evidence.row_key_progression
    assert (
        evidence.left_has_active_hierarchy
    )
    assert (
        evidence.right_starts_with_inherited_cells
    )
    assert not (
        evidence.contradictory_boundary
    )


def test_different_headers_are_contradictory():
    left = make_table(
        page=1,
        ref="left",
    )

    right = make_table(
        page=2,
        ref="right",
        names=(
            "Other",
            "Schema",
            "Here",
        ),
    )

    evidence = (
        TableContinuationEvidenceBuilder()
        .build(
            left=left,
            right=right,
        )
    )

    assert not evidence.repeated_header
    assert (
        evidence.contradictory_boundary
    )


def test_incompatible_geometry_fails_closed():
    left = make_table(
        page=1,
        ref="left",
    )

    right = make_table(
        page=2,
        ref="right",
        geometry=(
            (0.0, 10.0),
            (10.0, 20.0),
            (20.0, 100.0),
        ),
    )

    evidence = (
        TableContinuationEvidenceBuilder()
        .build(
            left=left,
            right=right,
        )
    )

    assert not (
        evidence.compatible_column_geometry
    )


def test_headerless_row_key_progression_is_derived():
    left = make_table(
        page=1,
        ref="left",
        names=(),
        hierarchy_columns=(),
        logical_key="T-001",
    )

    right = make_table(
        page=2,
        ref="right",
        names=(),
        hierarchy_columns=(),
        logical_key="T-002",
    )

    evidence = (
        TableContinuationEvidenceBuilder()
        .build(
            left=left,
            right=right,
        )
    )

    assert (
        evidence.headerless_continuation
    )
    assert evidence.row_key_progression
    assert not evidence.repeated_header


def test_headerless_without_progression_is_ambiguous_signal():
    left = make_table(
        page=1,
        ref="left",
        names=(),
        hierarchy_columns=(),
        logical_key="T-001",
    )

    right = make_table(
        page=2,
        ref="right",
        names=(),
        hierarchy_columns=(),
        logical_key="T-009",
    )

    evidence = (
        TableContinuationEvidenceBuilder()
        .build(
            left=left,
            right=right,
        )
    )

    assert (
        evidence.headerless_continuation
    )
    assert not (
        evidence.row_key_progression
    )


def test_explicit_contradictory_boundary_is_preserved():
    left = make_table(
        page=1,
        ref="left",
    )

    right = make_table(
        page=2,
        ref="right",
    )

    evidence = (
        TableContinuationEvidenceBuilder()
        .build(
            left=left,
            right=right,
            explicit_contradictory_boundary=True,
        )
    )

    assert (
        evidence.contradictory_boundary
    )


def test_missing_source_ref_fails_closed():
    original = make_table(
        page=1,
        ref="left",
    )

    left = CanonicalTable(
        column_count=(
            original.column_count
        ),
        rows=original.rows,
        column_names=(
            original.column_names
        ),
        hierarchy_columns=(
            original.hierarchy_columns
        ),
        source_table_refs=(),
    )

    right = make_table(
        page=2,
        ref="right",
    )

    with pytest.raises(
        TableContinuationEvidenceBuildError,
        match="source_table_refs",
    ):
        (
            TableContinuationEvidenceBuilder()
            .build(
                left=left,
                right=right,
            )
        )

def test_headerless_boundary_ordinal_progression_without_logical_keys():
    left = make_table(
        page=1,
        ref="left",
        names=(),
        hierarchy_columns=(),
        data=(
            "03",
            "Supplier C",
            "Conforme",
        ),
        logical_key=None,
    )

    right = make_table(
        page=2,
        ref="right",
        names=(),
        hierarchy_columns=(),
        data=(
            "04",
            "Supplier D",
            "Conforme",
        ),
        logical_key=None,
    )

    evidence = (
        TableContinuationEvidenceBuilder()
        .build(
            left=left,
            right=right,
        )
    )

    assert (
        evidence.headerless_continuation
    )

    assert (
        evidence.row_key_progression
    )


def test_headerless_boundary_ordinal_gap_fails_closed():
    left = make_table(
        page=1,
        ref="left",
        names=(),
        hierarchy_columns=(),
        data=(
            "03",
            "Supplier C",
            "Conforme",
        ),
        logical_key=None,
    )

    right = make_table(
        page=2,
        ref="right",
        names=(),
        hierarchy_columns=(),
        data=(
            "05",
            "Supplier E",
            "Conforme",
        ),
        logical_key=None,
    )

    evidence = (
        TableContinuationEvidenceBuilder()
        .build(
            left=left,
            right=right,
        )
    )

    assert (
        evidence.headerless_continuation
    )

    assert not (
        evidence.row_key_progression
    )



def test_shared_structural_identity_is_derived_from_canonical_tables():
    from dataclasses import replace

    left = replace(
        make_table(
            page=1,
            ref="left",
        ),
        structural_identity_refs=(
            "structural:table-74",
        ),
    )

    right = replace(
        make_table(
            page=2,
            ref="right",
            names=(),
        ),
        structural_identity_refs=(
            "structural:table-74",
        ),
    )

    evidence = (
        TableContinuationEvidenceBuilder()
        .build(
            left=left,
            right=right,
        )
    )

    assert evidence.shared_structural_identity


def test_different_structural_identities_do_not_match():
    from dataclasses import replace

    left = replace(
        make_table(
            page=1,
            ref="left",
        ),
        structural_identity_refs=(
            "structural:table-A",
        ),
    )

    right = replace(
        make_table(
            page=2,
            ref="right",
            names=(),
        ),
        structural_identity_refs=(
            "structural:table-B",
        ),
    )

    evidence = (
        TableContinuationEvidenceBuilder()
        .build(
            left=left,
            right=right,
        )
    )

    assert not evidence.shared_structural_identity
