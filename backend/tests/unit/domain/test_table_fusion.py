from app.domain.enums.table import (
    TableCellTextStatus,
    TableColumnMappingKind,
    TableEvidenceLayer,
)
from app.domain.tables.evidence import (
    NormalizedTableCellEvidence,
    NormalizedTableEvidence,
    NormalizedTableRowEvidence,
)
from app.domain.tables.fusion import (
    TableEvidenceMatcher,
    TablePhysicalProjector,
)


def cell(
    column: int,
    text: str | None,
    *,
    unavailable: bool = False,
):
    return NormalizedTableCellEvidence(
        column_start=column,
        column_end=column + 1,
        text_status=(
            TableCellTextStatus.UNAVAILABLE
            if unavailable
            else TableCellTextStatus.OBSERVED
        ),
        text=(
            None
            if unavailable
            else text
        ),
    )


def row(
    index: int,
    cells,
    *,
    header: bool = False,
):
    return NormalizedTableRowEvidence(
        row_index=index,
        cells=tuple(cells),
        source_header_signal=header,
    )


def table(
    *,
    ref: str,
    page: int,
    layer: TableEvidenceLayer,
    columns: int,
    rows,
):
    return NormalizedTableEvidence(
        source_ref=ref,
        page_number=page,
        layer=layer,
        column_count=columns,
        rows=tuple(rows),
    )


def test_exact_schema_match() -> None:
    physical = table(
        ref="native:p3:t1",
        page=3,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=[
            row(
                0,
                [
                    cell(0, "N°"),
                    cell(1, "Soumissionnaires"),
                    cell(2, "Montants lus"),
                    cell(3, "Montants corrigés"),
                    cell(4, "Observations"),
                ],
                header=True,
            )
        ],
    )

    structural = table(
        ref="tag:xref:702:p3",
        page=3,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=5,
        rows=[
            row(
                0,
                [
                    cell(0, None, unavailable=True),
                    cell(1, None, unavailable=True),
                    cell(2, None, unavailable=True),
                    cell(3, None, unavailable=True),
                    cell(4, None, unavailable=True),
                ]
            )
        ],
    )

    result = TableEvidenceMatcher().match_page(
        physical_tables=(physical,),
        structural_tables=(structural,),
    )

    assert len(result.matches) == 1

    match = result.matches[0]

    assert (
        match.mapping_kind
        == TableColumnMappingKind.EXACT
    )

    assert match.physical_column_ranges == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    )


def test_annuaire_nine_to_three_group_match() -> None:
    physical = table(
        ref="native:p9:t0",
        page=9,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=9,
        rows=[
            row(
                0,
                [
                    cell(0, ""),
                    cell(1, "Domaine"),
                    cell(2, ""),
                    cell(3, ""),
                    cell(4, "Ecosystème"),
                    cell(5, ""),
                    cell(6, ""),
                    cell(7, "Composants"),
                    cell(8, ""),
                ],
                header=True,
            )
        ],
    )

    structural = table(
        ref="tag:xref:1930:p9",
        page=9,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=3,
        rows=[
            row(
                0,
                [
                    cell(0, None, unavailable=True),
                    cell(1, None, unavailable=True),
                    cell(2, None, unavailable=True),
                ],
                header=True,
            )
        ],
    )

    result = TableEvidenceMatcher().match_page(
        physical_tables=(physical,),
        structural_tables=(structural,),
    )

    assert len(result.matches) == 1

    match = result.matches[0]

    assert (
        match.mapping_kind
        == TableColumnMappingKind.EQUAL_GROUPS
    )

    assert match.physical_column_ranges == (
        (0, 3),
        (3, 6),
        (6, 9),
    )


def test_group_mapping_requires_clean_header_signal() -> None:
    physical = table(
        ref="native:p4:t1",
        page=4,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=10,
        rows=[
            row(
                0,
                [
                    cell(0, "OFFICE"),
                    *[
                        cell(
                            index,
                            None,
                            unavailable=True,
                        )
                        for index
                        in range(1, 10)
                    ],
                ],
                header=True,
            )
        ],
    )

    structural = table(
        ref="tag:xref:988:p4",
        page=4,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, None, unavailable=True),
                    cell(1, None, unavailable=True),
                ],
                header=False,
            )
        ],
    )

    result = TableEvidenceMatcher().match_page(
        physical_tables=(physical,),
        structural_tables=(structural,),
    )

    assert result.matches == ()


def test_exact_matching_is_selected_before_group_matching() -> None:
    physical_five = table(
        ref="native:p4:t0",
        page=4,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=[],
    )

    physical_ten = table(
        ref="native:p4:t1",
        page=4,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=10,
        rows=[],
    )

    structural_five = table(
        ref="tag:xref:781:p4",
        page=4,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=5,
        rows=[],
    )

    structural_seven = table(
        ref="tag:xref:1007:p4",
        page=4,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=7,
        rows=[],
    )

    result = TableEvidenceMatcher().match_page(
        physical_tables=(
            physical_five,
            physical_ten,
        ),
        structural_tables=(
            structural_five,
            structural_seven,
        ),
    )

    assert len(result.matches) == 1

    assert (
        result.matches[0].physical.source_ref
        == "native:p4:t0"
    )

    assert (
        result.matches[0].structural.source_ref
        == "tag:xref:781:p4"
    )


def test_ambiguous_equal_schema_fails_closed() -> None:
    physical_a = table(
        ref="physical-a",
        page=1,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=[],
    )

    physical_b = table(
        ref="physical-b",
        page=1,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=[],
    )

    structural_a = table(
        ref="structural-a",
        page=1,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=5,
        rows=[],
    )

    structural_b = table(
        ref="structural-b",
        page=1,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=5,
        rows=[],
    )

    result = TableEvidenceMatcher().match_page(
        physical_tables=(
            physical_a,
            physical_b,
        ),
        structural_tables=(
            structural_a,
            structural_b,
        ),
    )

    assert result.matches == ()

    assert set(
        result.ambiguous_physical_refs
    ) == {
        "physical-a",
        "physical-b",
    }


def test_projector_preserves_exact_five_columns() -> None:
    physical = table(
        ref="native",
        page=3,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=5,
        rows=[
            row(
                0,
                [
                    cell(0, "01"),
                    cell(1, "Supplier"),
                    cell(2, "100"),
                    cell(3, "100"),
                    cell(4, "Conforme"),
                ]
            )
        ],
    )

    structural = table(
        ref="tagged",
        page=3,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=5,
        rows=[],
    )

    match = TableEvidenceMatcher().match_page(
        physical_tables=(physical,),
        structural_tables=(structural,),
    ).matches[0]

    projected = (
        TablePhysicalProjector()
        .project(match)
    )

    assert projected.column_count == 5

    assert [
        c.text
        for c in projected.rows[0].cells
    ] == [
        "01",
        "Supplier",
        "100",
        "100",
        "Conforme",
    ]


def test_projector_collapses_annuaire_header() -> None:
    physical = table(
        ref="native",
        page=9,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=9,
        rows=[
            row(
                0,
                [
                    cell(0, ""),
                    cell(1, "Domaine"),
                    cell(2, ""),
                    cell(3, ""),
                    cell(4, "Ecosystème"),
                    cell(5, ""),
                    cell(6, ""),
                    cell(7, "Composants"),
                    cell(8, ""),
                ],
                header=True,
            )
        ],
    )

    structural = table(
        ref="tagged",
        page=9,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=3,
        rows=[
            row(
                0,
                [
                    cell(0, None, unavailable=True),
                    cell(1, None, unavailable=True),
                    cell(2, None, unavailable=True),
                ],
                header=True,
            )
        ],
    )

    match = TableEvidenceMatcher().match_page(
        physical_tables=(physical,),
        structural_tables=(structural,),
    ).matches[0]

    projected = (
        TablePhysicalProjector()
        .project(match)
    )

    assert [
        c.text
        for c in projected.rows[0].cells
    ] == [
        "Domaine",
        "Ecosystème",
        "Composants",
    ]


def test_projected_group_distinguishes_blank_from_unavailable() -> None:
    physical = table(
        ref="native",
        page=10,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=9,
        rows=[
            row(
                0,
                [
                    cell(0, ""),
                    cell(1, None, unavailable=True),
                    cell(2, None, unavailable=True),
                    cell(3, ""),
                    cell(4, None, unavailable=True),
                    cell(5, None, unavailable=True),
                    cell(6, ""),
                    cell(7, "Cadre politique"),
                    cell(8, ""),
                ]
            )
        ],
    )

    structural = table(
        ref="tagged",
        page=10,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=3,
        rows=[],
    )

    # Build the already-established 3-column grouping
    # directly because this isolated test has no header row.
    from app.domain.tables.fusion import (
        TableEvidenceMatch,
    )

    match = TableEvidenceMatch(
        physical=physical,
        structural=structural,
        mapping_kind=(
            TableColumnMappingKind.EQUAL_GROUPS
        ),
        physical_column_ranges=(
            (0, 3),
            (3, 6),
            (6, 9),
        ),
    )

    projected = (
        TablePhysicalProjector()
        .project(match)
    )

    cells = projected.rows[0].cells

    assert cells[0].observed_blank is True
    assert cells[1].observed_blank is True

    assert cells[2].text == "Cadre politique"


def test_projected_all_unavailable_group_stays_unavailable() -> None:
    physical = table(
        ref="native",
        page=1,
        layer=TableEvidenceLayer.PHYSICAL,
        columns=3,
        rows=[
            row(
                0,
                [
                    cell(0, None, unavailable=True),
                    cell(1, None, unavailable=True),
                    cell(2, None, unavailable=True),
                ]
            )
        ],
    )

    structural = table(
        ref="tagged",
        page=1,
        layer=TableEvidenceLayer.STRUCTURAL,
        columns=1,
        rows=[],
    )

    from app.domain.tables.fusion import (
        TableEvidenceMatch,
    )

    match = TableEvidenceMatch(
        physical=physical,
        structural=structural,
        mapping_kind=(
            TableColumnMappingKind.EQUAL_GROUPS
        ),
        physical_column_ranges=(
            (0, 3),
        ),
    )

    projected = (
        TablePhysicalProjector()
        .project(match)
    )

    result = projected.rows[0].cells[0]

    assert (
        result.text_status
        == TableCellTextStatus.UNAVAILABLE
    )

    assert result.text is None
