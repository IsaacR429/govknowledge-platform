from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.application.services.table_evidence_compilation import (
    TableEvidenceCompilationError,
    TableEvidenceCompilationResult,
    TableEvidenceCompilationService,
    TableEvidenceCompiler,
)
from app.domain.enums.table import (
    TableCellTextStatus,
    TableCellValueSource,
    TableColumnMappingKind,
    TableEvidenceLayer,
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
from app.domain.tables.evidence import (
    NormalizedTableCellEvidence,
    NormalizedTableEvidence,
    NormalizedTableRowEvidence,
)
from app.domain.tables.fusion import (
    TableEvidenceMatch,
    TableEvidenceMatchingResult,
)


def source_span(
    *,
    page: int,
    ref: str,
    kind: TableSpanKind = TableSpanKind.TEXT,
    text: str | None = None,
) -> TableSourceSpan:
    return TableSourceSpan(
        page_number=page,
        kind=kind,
        source_ref=ref,
        bbox=TableBBox(
            x0=0.0,
            y0=0.0,
            x1=100.0,
            y1=20.0,
        ),
        text=text,
    )


def cell(
    column: int,
    text: str | None = None,
    *,
    page: int = 1,
    unavailable: bool = False,
) -> NormalizedTableCellEvidence:
    spans = [
        source_span(
            page=page,
            ref=f"p{page}:c{column}:geom",
            kind=TableSpanKind.GEOMETRY,
        )
    ]
    if text is not None and not unavailable:
        spans.append(
            source_span(
                page=page,
                ref=f"p{page}:c{column}:text",
                kind=TableSpanKind.TEXT,
                text=text,
            )
        )
    return NormalizedTableCellEvidence(
        column_start=column,
        column_end=column + 1,
        text_status=(
            TableCellTextStatus.UNAVAILABLE
            if unavailable
            else TableCellTextStatus.OBSERVED
        ),
        text=None if unavailable else text,
        source_spans=tuple(spans),
    )


def row(
    index: int,
    cells: list[NormalizedTableCellEvidence],
    *,
    header: bool = False,
) -> NormalizedTableRowEvidence:
    page = (
        cells[0].source_spans[0].page_number
        if cells and cells[0].source_spans
        else 1
    )
    return NormalizedTableRowEvidence(
        row_index=index,
        cells=tuple(cells),
        source_spans=(
            source_span(
                page=page,
                ref=f"p{page}:r{index}:geom",
                kind=TableSpanKind.GEOMETRY,
            ),
        ),
        source_header_signal=header,
    )


def physical_table(
    *,
    ref: str,
    page: int = 1,
    columns: int = 2,
    rows: list[NormalizedTableRowEvidence] | None = None,
) -> NormalizedTableEvidence:
    return NormalizedTableEvidence(
        source_ref=ref,
        page_number=page,
        layer=TableEvidenceLayer.PHYSICAL,
        column_count=columns,
        rows=tuple(rows or []),
    )


def structural_table(
    *,
    ref: str,
    page: int = 1,
    columns: int = 2,
    rows: list[NormalizedTableRowEvidence] | None = None,
) -> NormalizedTableEvidence:
    return NormalizedTableEvidence(
        source_ref=ref,
        page_number=page,
        layer=TableEvidenceLayer.STRUCTURAL,
        column_count=columns,
        rows=tuple(rows or []),
    )


def test_compiler_alias() -> None:
    assert TableEvidenceCompiler is TableEvidenceCompilationService


def test_empty_input_compilation() -> None:
    service = TableEvidenceCompilationService()
    result = service.compile()

    assert isinstance(result, TableEvidenceCompilationResult)
    assert result.tables == ()
    assert result.matched_table_count == 0
    assert result.unmatched_physical_refs == ()
    assert result.unmatched_structural_refs == ()
    assert result.ambiguous_physical_refs == ()
    assert result.ambiguous_structural_refs == ()


def test_invalid_layer_validation() -> None:
    service = TableEvidenceCompilationService()

    invalid_physical = structural_table(ref="s1", page=1)
    with pytest.raises(
        TableEvidenceCompilationError,
        match="physical_tables contains non-PHYSICAL evidence",
    ):
        service.compile(
            physical_tables=[invalid_physical],
            structural_tables=[],
        )

    invalid_structural = physical_table(ref="p1", page=1)
    with pytest.raises(
        TableEvidenceCompilationError,
        match="structural_tables contains non-STRUCTURAL evidence",
    ):
        service.compile(
            physical_tables=[],
            structural_tables=[invalid_structural],
        )


def test_single_table_end_to_end_compilation() -> None:
    p_tab = physical_table(
        ref="phys:p1:t1",
        page=1,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, "Code", page=1),
                    cell(1, "Libellé", page=1),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, "001", page=1),
                    cell(1, "Projet A", page=1),
                ],
            ),
        ],
    )

    s_tab = structural_table(
        ref="struct:p1:t1",
        page=1,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, None, page=1, unavailable=True),
                    cell(1, None, page=1, unavailable=True),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, None, page=1, unavailable=True),
                    cell(1, None, page=1, unavailable=True),
                ],
            ),
        ],
    )

    service = TableEvidenceCompilationService()
    result = service.compile(
        physical_tables=[p_tab],
        structural_tables=[s_tab],
    )

    assert result.matched_table_count == 1
    assert len(result.tables) == 1
    assert result.unmatched_physical_refs == ()
    assert result.unmatched_structural_refs == ()
    assert result.ambiguous_physical_refs == ()
    assert result.ambiguous_structural_refs == ()

    table = result.tables[0]
    assert isinstance(table, CanonicalTable)
    assert table.column_count == 2
    assert len(table.rows) == 2
    assert table.rows[0].role == TableRowRole.HEADER
    assert table.rows[1].role == TableRowRole.DATA
    assert [c.text for c in table.rows[1].cells] == ["001", "Projet A"]


def test_cross_page_repeated_header_is_resolved_as_one_sequence() -> None:
    p_tab1 = physical_table(
        ref="p1:t1",
        page=1,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, "Supplier", page=1),
                    cell(1, "Amount", page=1),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, "Acme", page=1),
                    cell(1, "100", page=1),
                ],
            ),
        ],
    )
    s_tab1 = structural_table(
        ref="s1:t1",
        page=1,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, None, page=1, unavailable=True),
                    cell(1, None, page=1, unavailable=True),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, None, page=1, unavailable=True),
                    cell(1, None, page=1, unavailable=True),
                ],
            ),
        ],
    )

    p_tab2 = physical_table(
        ref="p2:t1",
        page=2,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, "Supplier", page=2),
                    cell(1, "Amount", page=2),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, "Beta", page=2),
                    cell(1, "200", page=2),
                ],
            ),
        ],
    )
    s_tab2 = structural_table(
        ref="s2:t1",
        page=2,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, None, page=2, unavailable=True),
                    cell(1, None, page=2, unavailable=True),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, None, page=2, unavailable=True),
                    cell(1, None, page=2, unavailable=True),
                ],
            ),
        ],
    )

    recorded_calls: list[tuple] = []

    class MockStructureSequenceResolver:
        def resolve(
            self,
            tables,
            *,
            boundaries=None,
        ):
            recorded_calls.append(
                (
                    tuple(tables),
                    boundaries,
                )
            )
            return SimpleNamespace(
                tables=tuple(tables)
            )

    service = TableEvidenceCompilationService(
        structure_resolver=(
            MockStructureSequenceResolver()
        ),
    )

    result = service.compile(
        physical_tables=[
            p_tab1,
            p_tab2,
        ],
        structural_tables=[
            s_tab1,
            s_tab2,
        ],
    )

    assert result.matched_table_count == 2
    assert len(result.tables) == 2
    assert len(result.sequences) == 1
    assert len(result.sequences[0]) == 2

    # The two page-local fragments belong to the same
    # qualified logical sequence.
    assert len(recorded_calls) == 1
    assert len(recorded_calls[0][0]) == 2
    assert recorded_calls[0][1] is None

    left, right = recorded_calls[0][0]
    assert left.page_numbers == (1,)
    assert right.page_numbers == (2,)
    assert left.column_names == (
        "Supplier",
        "Amount",
    )
    assert right.column_names == (
        "Supplier",
        "Amount",
    )


def test_cross_page_different_headers_start_new_sequences() -> None:
    p_tab1 = physical_table(
        ref="p1:t1",
        page=1,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, "Supplier", page=1),
                    cell(1, "Amount", page=1),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, "Acme", page=1),
                    cell(1, "100", page=1),
                ],
            ),
        ],
    )
    s_tab1 = structural_table(
        ref="s1:t1",
        page=1,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(
                        0,
                        None,
                        page=1,
                        unavailable=True,
                    ),
                    cell(
                        1,
                        None,
                        page=1,
                        unavailable=True,
                    ),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(
                        0,
                        None,
                        page=1,
                        unavailable=True,
                    ),
                    cell(
                        1,
                        None,
                        page=1,
                        unavailable=True,
                    ),
                ],
            ),
        ],
    )

    p_tab2 = physical_table(
        ref="p2:t1",
        page=2,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, "Department", page=2),
                    cell(1, "Budget", page=2),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, "Finance", page=2),
                    cell(1, "200", page=2),
                ],
            ),
        ],
    )
    s_tab2 = structural_table(
        ref="s2:t1",
        page=2,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(
                        0,
                        None,
                        page=2,
                        unavailable=True,
                    ),
                    cell(
                        1,
                        None,
                        page=2,
                        unavailable=True,
                    ),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(
                        0,
                        None,
                        page=2,
                        unavailable=True,
                    ),
                    cell(
                        1,
                        None,
                        page=2,
                        unavailable=True,
                    ),
                ],
            ),
        ],
    )

    recorded_calls: list[
        tuple
    ] = []

    class MockStructureSequenceResolver:
        def resolve(
            self,
            tables,
            *,
            boundaries=None,
        ):
            recorded_calls.append(
                (
                    tuple(tables),
                    boundaries,
                )
            )
            return SimpleNamespace(
                tables=tuple(tables)
            )

    service = TableEvidenceCompilationService(
        structure_resolver=(
            MockStructureSequenceResolver()
        ),
    )

    result = service.compile(
        physical_tables=[
            p_tab1,
            p_tab2,
        ],
        structural_tables=[
            s_tab1,
            s_tab2,
        ],
    )

    assert result.matched_table_count == 2
    assert len(result.tables) == 2
    assert len(result.sequences) == 2
    assert all(
        len(sequence) == 1
        for sequence in result.sequences
    )

    # Different headers are contradictory evidence.
    # The page boundary must therefore remain closed.
    assert len(recorded_calls) == 2
    assert len(recorded_calls[0][0]) == 1
    assert len(recorded_calls[1][0]) == 1


def test_cross_page_ambiguous_boundary_starts_new_sequence_without_failing() -> None:
    p_tab1 = physical_table(
        ref="p1:t1",
        page=1,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, "Alpha", page=1),
                    cell(1, "100", page=1),
                ],
            ),
        ],
    )
    s_tab1 = structural_table(
        ref="s1:t1",
        page=1,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, None, page=1, unavailable=True),
                    cell(1, None, page=1, unavailable=True),
                ],
            ),
        ],
    )

    p_tab2 = physical_table(
        ref="p2:t1",
        page=2,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, "Beta", page=2),
                    cell(1, "200", page=2),
                ],
            ),
        ],
    )
    s_tab2 = structural_table(
        ref="s2:t1",
        page=2,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, None, page=2, unavailable=True),
                    cell(1, None, page=2, unavailable=True),
                ],
            ),
        ],
    )

    recorded_calls: list[tuple] = []

    class MockStructureSequenceResolver:
        def resolve(
            self,
            tables,
            *,
            boundaries=None,
        ):
            recorded_calls.append(
                (
                    tuple(tables),
                    boundaries,
                )
            )
            return SimpleNamespace(
                tables=tuple(tables)
            )

    service = TableEvidenceCompilationService(
        structure_resolver=(
            MockStructureSequenceResolver()
        ),
    )

    result = service.compile(
        physical_tables=[
            p_tab1,
            p_tab2,
        ],
        structural_tables=[
            s_tab1,
            s_tab2,
        ],
    )

    assert result.matched_table_count == 2
    assert len(result.tables) == 2
    assert len(recorded_calls) == 2
    assert len(recorded_calls[0][0]) == 1
    assert len(recorded_calls[1][0]) == 1


def test_cross_page_procurement_state_transfer_preserves_page_local_provenance() -> None:
    p_tab1 = physical_table(
        ref="p1:t1",
        page=1,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, "N°", page=1),
                    cell(1, "Soumissionnaires", page=1),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, "Lot 1 : Travaux", page=1),
                    cell(1, "", page=1),
                ],
            ),
            row(
                2,
                [
                    cell(0, "01", page=1),
                    cell(1, "Supplier A", page=1),
                ],
            ),
        ],
    )
    s_tab1 = structural_table(
        ref="s1:t1",
        page=1,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, None, page=1, unavailable=True),
                    cell(1, None, page=1, unavailable=True),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, None, page=1, unavailable=True),
                    cell(1, None, page=1, unavailable=True),
                ],
            ),
            row(
                2,
                [
                    cell(0, None, page=1, unavailable=True),
                    cell(1, None, page=1, unavailable=True),
                ],
            ),
        ],
    )

    p_tab2 = physical_table(
        ref="p2:t1",
        page=2,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, "N°", page=2),
                    cell(1, "Soumissionnaires", page=2),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, "02", page=2),
                    cell(1, "Supplier B", page=2),
                ],
            ),
        ],
    )
    s_tab2 = structural_table(
        ref="s2:t1",
        page=2,
        columns=2,
        rows=[
            row(
                0,
                [
                    cell(0, None, page=2, unavailable=True),
                    cell(1, None, page=2, unavailable=True),
                ],
                header=True,
            ),
            row(
                1,
                [
                    cell(0, None, page=2, unavailable=True),
                    cell(1, None, page=2, unavailable=True),
                ],
            ),
        ],
    )

    service = TableEvidenceCompilationService()
    result = service.compile(
        physical_tables=[
            p_tab1,
            p_tab2,
        ],
        structural_tables=[
            s_tab1,
            s_tab2,
        ],
    )

    assert result.matched_table_count == 2
    assert len(result.tables) == 2

    page_1_table, page_2_table = result.tables

    assert page_1_table.page_numbers == (1,)
    assert page_2_table.page_numbers == (2,)

    assert page_1_table.rows[1].role == TableRowRole.SECTION
    assert page_1_table.rows[1].section_key == "LOT_1"
    assert page_1_table.rows[2].role == TableRowRole.DATA
    assert page_1_table.rows[2].section_key == "LOT_1"
    assert page_1_table.rows[2].logical_key == "LOT_1/01"

    assert page_2_table.rows[1].role == TableRowRole.DATA
    assert page_2_table.rows[1].section_key == "LOT_1"
    assert page_2_table.rows[1].logical_key == "LOT_1/02"
    assert page_2_table.rows[1].page_numbers == (2,)
    assert [
        span.page_number
        for span in page_2_table.rows[1].cells[1].origin_spans
    ] == [2]


def test_unmatched_and_ambiguous_tracking() -> None:
    # Page 1: 2 physical tables with 2 columns, only 1 structural with 2 columns -> ambiguous
    p1 = physical_table(
        ref="p1:ambig1",
        page=1,
        columns=2,
        rows=[row(0, [cell(0, "A"), cell(1, "B")])],
    )
    p2 = physical_table(
        ref="p1:ambig2",
        page=1,
        columns=2,
        rows=[row(0, [cell(0, "C"), cell(1, "D")])],
    )
    s1 = structural_table(
        ref="s1:ambig",
        page=1,
        columns=2,
        rows=[row(0, [cell(0, None, unavailable=True), cell(1, None, unavailable=True)])],
    )

    # Page 2: 1 physical table with 3 columns, no structural tables -> unmatched physical
    p3 = physical_table(
        ref="p2:unmatched",
        page=2,
        columns=3,
        rows=[],
    )

    # Page 3: 1 structural table with 2 columns, no physical tables -> unmatched structural
    s2 = structural_table(
        ref="p3:unmatched",
        page=3,
        columns=2,
        rows=[],
    )

    service = TableEvidenceCompilationService()
    result = service.compile(
        physical_tables=[p1, p2, p3],
        structural_tables=[s1, s2],
    )

    assert result.matched_table_count == 0
    assert result.tables == ()
    assert "p2:unmatched" in result.unmatched_physical_refs
    assert "p3:unmatched" in result.unmatched_structural_refs
    assert "p1:ambig1" in result.ambiguous_physical_refs
    assert "p1:ambig2" in result.ambiguous_physical_refs
    assert "s1:ambig" in result.ambiguous_structural_refs
