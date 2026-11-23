from dataclasses import replace
from uuid import uuid4

import pytest

from app.application.services.table_evidence_atoms import (
    TableCellEvidenceAtom,
    TableDocumentContext,
    TableEvidenceAtom,
    TableEvidenceAtomBuilder,
    TableEvidenceBuildError,
    TableEvidenceLookupError,
    LabeledRowComparisonRequest,
    LabeledRowLookupRequest,
    TableEvidenceLookupService,
    TableLookupRequest,
)
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


def span(
    *,
    page: int,
    ref: str,
    text: str | None = None,
    kind: TableSpanKind | None = None,
):
    return TableSourceSpan(
        page_number=page,
        kind=(
            kind
            if kind is not None
            else (
                TableSpanKind.TEXT
                if text is not None
                else TableSpanKind.STRUCTURE
            )
        ),
        source_ref=ref,
        text=text,
    )


def cell(
    *,
    page: int,
    row: int,
    column: int,
    text: str,
    value_source=(
        TableCellValueSource.SOURCE_TEXT
    ),
    origin_page: int | None = None,
):
    origin = (
        page
        if origin_page is None
        else origin_page
    )

    return CanonicalTableCell(
        column_start=column,
        column_end=column + 1,
        text=text,
        value_source=value_source,
        origin_spans=(
            (
                span(
                    page=origin,
                    ref=(
                        f"origin:p{origin}:"
                        f"r{row}:c{column}"
                    ),
                    text=text,
                ),
            )
            if text
            else ()
        ),
        structure_spans=(
            span(
                page=page,
                ref=(
                    f"structure:p{page}:"
                    f"r{row}:c{column}"
                ),
            ),
        ),
    )


def row(
    *,
    page: int,
    index: int,
    values,
    role=TableRowRole.DATA,
    logical_key=None,
    section_key=None,
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
        source_spans=(
            span(
                page=page,
                ref=(
                    f"row:p{page}:r{index}"
                ),
            ),
        ),
        logical_key=logical_key,
        section_key=section_key,
    )


def table(
    *,
    page: int,
    ref: str,
    rows,
    names=(),
):
    return CanonicalTable(
        column_count=5,
        rows=tuple(rows),
        column_names=tuple(names),
        hierarchy_columns=(),
        source_table_refs=(
            ref,
        ),
    )


def context():
    return TableDocumentContext(
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_ids=(
            (
                3,
                uuid4(),
            ),
            (
                4,
                uuid4(),
            ),
            (
                9,
                uuid4(),
            ),
            (
                10,
                uuid4(),
            ),
        ),
        language="fr",
    )


def procurement_tables():
    header = row(
        page=3,
        index=0,
        role=TableRowRole.HEADER,
        values=(
            "N°",
            "Soumissionnaires",
            "Montants lus en FCFA",
            "Montants corrigés en FCFA",
            "Observations",
        ),
    )

    section1 = row(
        page=3,
        index=1,
        role=TableRowRole.SECTION,
        section_key="LOT_1",
        values=(
            "Lot 1 : test",
            "",
            "",
            "",
            "",
        ),
    )

    lot1_05 = row(
        page=3,
        index=2,
        logical_key="LOT_1/05",
        section_key="LOT_1",
        values=(
            "05",
            "WOURE SERVICES",
            "100",
            "100",
            "Ecartée",
        ),
    )

    p3 = table(
        page=3,
        ref="table:p3",
        rows=(
            header,
            section1,
            lot1_05,
        ),
    )

    lot2_section = row(
        page=4,
        index=0,
        role=TableRowRole.SECTION,
        section_key="LOT_2",
        values=(
            "Lot 2 : test",
            "",
            "",
            "",
            "",
        ),
    )

    lot2_05 = row(
        page=4,
        index=1,
        logical_key="LOT_2/05",
        section_key="LOT_2",
        values=(
            "05",
            (
                "ENTREPRISE ZERBO AWA "
                "ET FRERES (EZAF)"
            ),
            "3 972 300 HTVA",
            "4 417 530 TTC",
            "Conforme 1er",
        ),
    )

    summary = row(
        page=4,
        index=2,
        role=TableRowRole.SUMMARY,
        values=(
            "ATTRIBUTAIRES PROVISOIRES",
            "Lot 2 : EZAF",
            "",
            "",
            "",
        ),
    )

    p4 = table(
        page=4,
        ref="table:p4",
        rows=(
            lot2_section,
            lot2_05,
            summary,
        ),
    )

    return (
        p3,
        p4,
    )


def test_builder_propagates_header_schema_across_pages():
    p3, p4 = (
        procurement_tables()
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                p3,
                p4,
            ),
            context=context(),
        )
    )

    lot2 = next(
        atom
        for atom in atoms
        if (
            atom.logical_key
            == "LOT_2/05"
        )
    )

    assert (
        lot2.cells[1].column_name
        == "Soumissionnaires"
    )


def test_builder_excludes_header_and_unkeyed_preamble():
    preamble = row(
        page=3,
        index=0,
        values=(
            "MINISTERE",
            "",
            "",
            "",
            "",
        ),
    )

    header = row(
        page=3,
        index=1,
        role=TableRowRole.HEADER,
        values=(
            "N°",
            "Soumissionnaires",
            "Montants lus",
            "Montants corrigés",
            "Observations",
        ),
    )

    data = row(
        page=3,
        index=2,
        logical_key="LOT_1/01",
        section_key="LOT_1",
        values=(
            "01",
            "A",
            "1",
            "1",
            "Conforme",
        ),
    )

    source = table(
        page=3,
        ref="table:p3",
        rows=(
            preamble,
            header,
            data,
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                source,
            ),
            context=context(),
        )
    )

    assert len(atoms) == 1

    assert (
        atoms[0].logical_key
        == "LOT_1/01"
    )


def test_lookup_distinguishes_same_ordinal_between_lots():
    p3, p4 = (
        procurement_tables()
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                p3,
                p4,
            ),
            context=context(),
        )
    )

    service = (
        TableEvidenceLookupService()
    )

    lot1 = service.lookup(
        request=TableLookupRequest(
            section_key="LOT_1",
            ordinal=5,
            column_name="Soumissionnaires",
        ),
        evidence=atoms,
    )

    lot2 = service.lookup(
        request=TableLookupRequest(
            section_key="LOT_2",
            ordinal=5,
            column_name="Soumissionnaires",
        ),
        evidence=atoms,
    )

    assert lot1 is not None
    assert lot2 is not None

    assert (
        lot1.cell.text
        == "WOURE SERVICES"
    )

    assert (
        "EZAF"
        in lot2.cell.text
    )


def test_lookup_returns_none_for_missing_logical_row():
    p3, p4 = (
        procurement_tables()
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                p3,
                p4,
            ),
            context=context(),
        )
    )

    result = (
        TableEvidenceLookupService()
        .lookup(
            request=TableLookupRequest(
                section_key="LOT_2",
                ordinal=9,
                column_name=(
                    "Soumissionnaires"
                ),
            ),
            evidence=atoms,
        )
    )

    assert result is None


def test_lookup_returns_none_for_unknown_column():
    p3, p4 = (
        procurement_tables()
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                p3,
                p4,
            ),
            context=context(),
        )
    )

    result = (
        TableEvidenceLookupService()
        .lookup(
            request=TableLookupRequest(
                section_key="LOT_2",
                ordinal=5,
                column_name="Unknown",
            ),
            evidence=atoms,
        )
    )

    assert result is None


def test_lookup_fails_closed_on_duplicate_logical_key():
    p3, p4 = (
        procurement_tables()
    )

    atoms = list(
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                p3,
                p4,
            ),
            context=context(),
        )
    )

    target = next(
        atom
        for atom in atoms
        if (
            atom.logical_key
            == "LOT_2/05"
        )
    )

    atoms.append(
        TableEvidenceAtom(
            evidence_id=(
                target.evidence_id
                + ":duplicate"
            ),
            document_id=(
                target.document_id
            ),
            document_version_id=(
                target.document_version_id
            ),
            page_id=(
                target.page_id
            ),
            page_number=(
                target.page_number
            ),
            source_table_refs=(
                target.source_table_refs
            ),
            row_index=999,
            role=target.role,
            logical_key=(
                target.logical_key
            ),
            section_key=(
                target.section_key
            ),
            column_names=(
                target.column_names
            ),
            cells=target.cells,
            language=(
                target.language
            ),
        )
    )

    with pytest.raises(
        TableEvidenceLookupError,
        match="ambiguous",
    ):
        (
            TableEvidenceLookupService()
            .lookup(
                request=(
                    TableLookupRequest(
                        section_key="LOT_2",
                        ordinal=5,
                        column_name=(
                            "Soumissionnaires"
                        ),
                    )
                ),
                evidence=tuple(
                    atoms
                ),
            )
        )


def test_search_text_contains_key_and_named_cells():
    p3, p4 = (
        procurement_tables()
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                p3,
                p4,
            ),
            context=context(),
        )
    )

    target = next(
        atom
        for atom in atoms
        if (
            atom.logical_key
            == "LOT_2/05"
        )
    )

    assert (
        "LOT_2/05"
        in target.search_text
    )

    assert (
        "Soumissionnaires:"
        in target.search_text
    )

    assert (
        "EZAF"
        in target.search_text
    )


def test_inherited_cell_preserves_value_and_structure_pages():
    header = row(
        page=10,
        index=0,
        role=TableRowRole.HEADER,
        values=(
            "Domaine",
            "Ecosystème",
            "Composants",
            "X",
            "Y",
        ),
    )

    inherited = (
        CanonicalTableCell(
            column_start=0,
            column_end=1,
            text="Institutionnel",
            value_source=(
                TableCellValueSource.INHERITED
            ),
            origin_spans=(
                span(
                    page=9,
                    ref="origin:p9",
                    text="Institutionnel",
                ),
            ),
            structure_spans=(
                span(
                    page=10,
                    ref="structure:p10",
                ),
            ),
        )
    )

    data = CanonicalTableRow(
        row_index=1,
        role=TableRowRole.DATA,
        cells=(
            inherited,
            cell(
                page=10,
                row=1,
                column=1,
                text="Environnement",
            ),
            cell(
                page=10,
                row=1,
                column=2,
                text="Cadre",
            ),
            cell(
                page=10,
                row=1,
                column=3,
                text="x",
            ),
            cell(
                page=10,
                row=1,
                column=4,
                text="y",
            ),
        ),
        source_spans=(
            span(
                page=10,
                ref="row:p10:r1",
            ),
        ),
        logical_key="ROW/01",
        section_key="ROW",
    )

    source = table(
        page=10,
        ref="hierarchy:p10",
        rows=(
            header,
            data,
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                source,
            ),
            context=context(),
        )
    )

    first = atoms[0].cells[0]

    assert (
        first.value_source
        == TableCellValueSource.INHERITED
    )

    assert (
        first.origin_pages
        == (9,)
    )

    assert (
        first.structure_pages
        == (10,)
    )

def generic_three_column_table(
    *,
    rows,
    ref="native:p1:t0",
):
    return CanonicalTable(
        column_count=3,
        rows=tuple(rows),
        column_names=(
            "Feature",
            "Credit Score System",
            "Indian EMI Calculator System",
        ),
        hierarchy_columns=(),
        source_table_refs=(
            ref,
        ),
    )


def generic_context():
    return TableDocumentContext(
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_ids=(
            (
                3,
                uuid4(),
            ),
        ),
        language="en",
    )


def test_generic_unique_first_column_labels_emit_data_atoms():
    source = generic_three_column_table(
        rows=(
            row(
                page=3,
                index=1,
                values=(
                    "What it is",
                    "A creditworthiness score",
                    "A payment calculator",
                ),
            ),
            row(
                page=3,
                index=2,
                values=(
                    "Output",
                    (
                        "A 3-digit score "
                        "(typically 300 to 900)"
                    ),
                    "Monthly EMI",
                ),
            ),
            row(
                page=3,
                index=3,
                values=(
                    "Primary Purpose",
                    "Assess credit risk",
                    "Estimate repayments",
                ),
            ),
        ),
    )

    build_context = generic_context()

    first = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                source,
            ),
            context=build_context,
        )
    )

    second = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                source,
            ),
            context=build_context,
        )
    )

    assert len(first) == 3

    assert tuple(
        atom.logical_key
        for atom in first
    ) == tuple(
        atom.logical_key
        for atom in second
    )

    assert all(
        atom.logical_key is not None
        and atom.logical_key.startswith(
            "GENERIC/"
        )
        for atom in first
    )

    assert len(
        {
            atom.logical_key
            for atom in first
        }
    ) == 3

    output = next(
        atom
        for atom in first
        if atom.cells[0].text
        == "Output"
    )

    assert output.section_key is None

    assert output.column_names == (
        "Feature",
        "Credit Score System",
        "Indian EMI Calculator System",
    )

    assert (
        output.cells[1].column_name
        == "Credit Score System"
    )

    assert (
        output.cells[1].text
        == (
            "A 3-digit score "
            "(typically 300 to 900)"
        )
    )

    assert output.cells[1].origin_spans


def test_generic_duplicate_normalized_labels_fail_closed():
    source = generic_three_column_table(
        rows=(
            row(
                page=3,
                index=1,
                values=(
                    "Output",
                    "First",
                    "A",
                ),
            ),
            row(
                page=3,
                index=2,
                values=(
                    "OUT PUT",
                    "Second",
                    "B",
                ),
            ),
            row(
                page=3,
                index=3,
                values=(
                    "Purpose",
                    "Third",
                    "C",
                ),
            ),
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                source,
            ),
            context=generic_context(),
        )
    )

    assert len(atoms) == 1

    assert (
        atoms[0].cells[0].text
        == "Purpose"
    )


def test_generic_inherited_first_column_label_is_not_addressable():
    inherited_label = CanonicalTableCell(
        column_start=0,
        column_end=1,
        text="Output",
        value_source=(
            TableCellValueSource.INHERITED
        ),
        origin_spans=(
            span(
                page=3,
                ref="origin:p3:inherited",
                text="Output",
            ),
        ),
        structure_spans=(
            span(
                page=3,
                ref="structure:p3:inherited",
            ),
        ),
    )

    inherited_row = CanonicalTableRow(
        row_index=1,
        role=TableRowRole.DATA,
        cells=(
            inherited_label,
            cell(
                page=3,
                row=1,
                column=1,
                text="Unsafe target",
            ),
            cell(
                page=3,
                row=1,
                column=2,
                text="Other",
            ),
        ),
        source_spans=(
            span(
                page=3,
                ref="row:p3:r1",
            ),
        ),
        logical_key=None,
        section_key=None,
    )

    direct_row = row(
        page=3,
        index=2,
        values=(
            "Purpose",
            "Safe target",
            "Other",
        ),
    )

    source = generic_three_column_table(
        rows=(
            inherited_row,
            direct_row,
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                source,
            ),
            context=generic_context(),
        )
    )

    assert len(atoms) == 1
    assert (
        atoms[0].cells[0].text
        == "Purpose"
    )


def test_existing_procurement_logical_keys_are_unchanged():
    source_tables = procurement_tables()
    build_context = context()

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=source_tables,
            context=build_context,
        )
    )

    keys = tuple(
        atom.logical_key
        for atom in atoms
        if atom.role
        == TableRowRole.DATA
    )

    assert keys
    assert all(
        key is not None
        and not key.startswith(
            "GENERIC/"
        )
        for key in keys
    )

def test_labeled_row_lookup_selects_exact_generic_cell():
    source = generic_three_column_table(
        rows=(
            row(
                page=3,
                index=1,
                values=(
                    "Output",
                    "A 3-digit score",
                    "Monthly EMI",
                ),
            ),
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                source,
            ),
            context=generic_context(),
        )
    )

    assert len(atoms) == 1

    target = atoms[0]

    assert target.logical_key is not None

    parts = (
        target.logical_key.split("/")
    )

    request = LabeledRowLookupRequest(
        table_key="/".join(
            parts[:2]
        ),
        row_label="Output",
        logical_key=(
            target.logical_key
        ),
        column_name=(
            "Credit Score System"
        ),
    )

    selection = (
        TableEvidenceLookupService()
        .lookup(
            request=request,
            evidence=atoms,
        )
    )

    assert selection is not None

    assert (
        selection.row.logical_key
        == target.logical_key
    )

    assert (
        selection.cell.text
        == "A 3-digit score"
    )


def test_labeled_row_lookup_rejects_wrong_row_label():
    source = generic_three_column_table(
        rows=(
            row(
                page=3,
                index=1,
                values=(
                    "Output",
                    "A 3-digit score",
                    "Monthly EMI",
                ),
            ),
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                source,
            ),
            context=generic_context(),
        )
    )

    target = atoms[0]

    assert target.logical_key is not None

    parts = (
        target.logical_key.split("/")
    )

    request = LabeledRowLookupRequest(
        table_key="/".join(
            parts[:2]
        ),
        row_label="Purpose",
        logical_key=(
            target.logical_key
        ),
        column_name=(
            "Credit Score System"
        ),
    )

    with pytest.raises(
        TableEvidenceLookupError,
        match="row label",
    ):
        (
            TableEvidenceLookupService()
            .lookup(
                request=request,
                evidence=atoms,
            )
        )


def test_labeled_row_request_rejects_cross_table_key():
    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        LabeledRowLookupRequest(
            table_key=(
                "GENERIC/"
                + "a" * 64
            ),
            row_label="Output",
            logical_key=(
                "GENERIC/"
                + "b" * 64
                + "/"
                + "c" * 64
            ),
            column_name="Output",
        )


def test_table_atom_reports_multi_page_physical_structure() -> None:
    first = TableCellEvidenceAtom(
        column_start=0,
        column_end=1,
        column_name="Ministry",
        text="Ministry of Health",
        value_source=TableCellValueSource.SOURCE_TEXT,
        origin_spans=(
            span(
                page=18,
                ref="p18:c0:text",
                text="Ministry of Health",
            ),
        ),
        structure_spans=(
            span(
                page=18,
                ref="p18:c0:structure",
                kind=TableSpanKind.STRUCTURE,
            ),
        ),
    )

    second = TableCellEvidenceAtom(
        column_start=1,
        column_end=2,
        column_name="Responsibility",
        text="Public health and vaccination",
        value_source=TableCellValueSource.SOURCE_TEXT,
        origin_spans=(
            span(
                page=18,
                ref="p18:c1:text",
                text="Public health",
            ),
            span(
                page=19,
                ref="p19:c1:text",
                text="and vaccination",
            ),
        ),
        structure_spans=(
            span(
                page=18,
                ref="p18:c1:structure",
                kind=TableSpanKind.STRUCTURE,
            ),
            span(
                page=19,
                ref="p19:c1:structure",
                kind=TableSpanKind.STRUCTURE,
            ),
        ),
    )

    atom = TableEvidenceAtom(
        evidence_id="table:test:split-row",
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=18,
        source_table_refs=(
            "table:p18",
            "table:p19",
        ),
        row_index=3,
        role=TableRowRole.DATA,
        logical_key="MINISTRY/03",
        section_key="MINISTRY",
        column_names=(
            "Ministry",
            "Responsibility",
        ),
        cells=(
            first,
            second,
        ),
        language="en",
    )

    assert atom.origin_pages == (
        18,
        19,
    )
    assert atom.structure_pages == (
        18,
        19,
    )
    assert atom.spans_multiple_physical_pages

def test_builder_addresses_headerless_continuation_rows_with_carried_schema():
    header = row(
        page=3,
        index=0,
        role=TableRowRole.HEADER,
        values=(
            "Crit?res",
            "Cellule animale",
            "Cellule v?g?tale",
        ),
    )

    chloroplastes = row(
        page=3,
        index=1,
        values=(
            "Chloroplastes",
            "Absents",
            "Pr?sents",
        ),
    )

    page_three = CanonicalTable(
        column_count=3,
        rows=(
            header,
            chloroplastes,
        ),
        column_names=(),
        hierarchy_columns=(),
        source_table_refs=(
            "bio:p3",
        ),
    )

    vacuoles = row(
        page=4,
        index=0,
        values=(
            "Vacuoles",
            "Petites, nombreuses et temporaires",
            "Une grande vacuole centrale permanente",
        ),
    )

    page_four = CanonicalTable(
        column_count=3,
        rows=(
            vacuoles,
        ),
        column_names=(),
        hierarchy_columns=(),
        source_table_refs=(
            "bio:p4",
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                page_three,
                page_four,
            ),
            context=context(),
        )
    )

    page_four_atoms = tuple(
        atom
        for atom in atoms
        if atom.page_number == 4
    )

    assert len(page_four_atoms) == 1

    atom = page_four_atoms[0]

    assert atom.logical_key is not None

    assert atom.column_names == (
        "Crit?res",
        "Cellule animale",
        "Cellule v?g?tale",
    )

    assert tuple(
        cell.text
        for cell in atom.cells
    ) == (
        "Vacuoles",
        "Petites, nombreuses et temporaires",
        "Une grande vacuole centrale permanente",
    )



def test_builder_uses_one_generic_table_identity_for_approved_sequence():
    header = row(
        page=3,
        index=0,
        role=TableRowRole.HEADER,
        values=(
            "Crit?res",
            "Cellule animale",
            "Cellule v?g?tale",
        ),
    )

    chloroplastes = row(
        page=3,
        index=1,
        values=(
            "Chloroplastes",
            "Absents",
            "Pr?sents",
        ),
    )

    page_three = CanonicalTable(
        column_count=3,
        rows=(
            header,
            chloroplastes,
        ),
        column_names=(),
        hierarchy_columns=(),
        source_table_refs=(
            "bio:p3",
        ),
        structural_identity_refs=(
            "structural:bio-table",
        ),
    )

    vacuoles = row(
        page=4,
        index=2,
        values=(
            "Vacuoles",
            "Petites, nombreuses et temporaires",
            "Une grande vacuole centrale permanente",
        ),
    )

    page_four = CanonicalTable(
        column_count=3,
        rows=(
            vacuoles,
        ),
        column_names=(),
        hierarchy_columns=(),
        source_table_refs=(
            "bio:p4",
        ),
        structural_identity_refs=(
            "structural:bio-table",
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                page_three,
                page_four,
            ),
            context=context(),
        )
    )

    generic_atoms = tuple(
        atom
        for atom in atoms
        if (
            atom.logical_key is not None
            and atom.logical_key.startswith(
                "GENERIC/"
            )
        )
    )

    assert len(generic_atoms) == 2

    table_keys = {
        "/".join(
            atom.logical_key.split("/")[:2]
        )
        for atom in generic_atoms
    }

    assert len(table_keys) == 1

    assert {
        atom.page_number
        for atom in generic_atoms
    } == {
        3,
        4,
    }

    # Logical identity may be shared, but physical
    # provenance must remain fragment-local.
    page_three_atom = next(
        atom
        for atom in generic_atoms
        if atom.page_number == 3
    )

    page_four_atom = next(
        atom
        for atom in generic_atoms
        if atom.page_number == 4
    )

    assert (
        page_three_atom.source_table_refs
        == ("bio:p3",)
    )

    assert (
        page_four_atom.source_table_refs
        == ("bio:p4",)
    )


def test_labeled_row_comparison_selects_multiple_cells_from_same_row():

    source = generic_three_column_table(
        rows=(
            row(
                page=3,
                index=1,
                values=(
                    "Output",
                    "A 3-digit score",
                    "Monthly EMI",
                ),
            ),
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(
                source,
            ),
            context=generic_context(),
        )
    )

    assert len(atoms) == 1

    target = atoms[0]

    assert target.logical_key is not None

    parts = (
        target.logical_key.split("/")
    )

    request = LabeledRowComparisonRequest(
        table_key="/".join(
            parts[:2]
        ),
        row_label="Output",
        logical_key=(
            target.logical_key
        ),
        column_names=(
            "Indian EMI Calculator System",
            "Credit Score System",
        ),
    )

    selection = (
        TableEvidenceLookupService()
        .lookup(
            request=request,
            evidence=atoms,
        )
    )

    assert selection is not None

    assert (
        selection.row.logical_key
        == target.logical_key
    )

    assert (
        tuple(
            cell.column_name
            for cell in selection.cells
        )
        == (
            "Indian EMI Calculator System",
            "Credit Score System",
        )
    )

    assert (
        tuple(
            cell.text
            for cell in selection.cells
        )
        == (
            "Monthly EMI",
            "A 3-digit score",
        )
    )

    assert all(
        cell in selection.row.cells
        for cell in selection.cells
    )


def test_labeled_row_comparison_returns_none_for_missing_cell():

    source = generic_three_column_table(
        rows=(
            row(
                page=3,
                index=1,
                values=(
                    "Output",
                    "A 3-digit score",
                    "Monthly EMI",
                ),
            ),
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(source,),
            context=generic_context(),
        )
    )

    target = atoms[0]

    assert target.logical_key is not None

    parts = target.logical_key.split("/")

    request = LabeledRowComparisonRequest(
        table_key="/".join(
            parts[:2]
        ),
        row_label="Output",
        logical_key=target.logical_key,
        column_names=(
            "Credit Score System",
            "Missing Column",
        ),
    )

    result = (
        TableEvidenceLookupService()
        .lookup(
            request=request,
            evidence=atoms,
        )
    )

    assert result is None


def test_labeled_row_comparison_fails_closed_on_ambiguous_cell():

    source = generic_three_column_table(
        rows=(
            row(
                page=3,
                index=1,
                values=(
                    "Output",
                    "A 3-digit score",
                    "Monthly EMI",
                ),
            ),
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(source,),
            context=generic_context(),
        )
    )

    target = atoms[0]

    assert target.logical_key is not None

    parts = target.logical_key.split("/")

    # Deliberately make the first requested value
    # ambiguous inside the same resolved row.
    ambiguous_target = replace(
        target,
        cells=(
            target.cells
            + (
                target.cells[1],
            )
        ),
    )

    request = LabeledRowComparisonRequest(
        table_key="/".join(
            parts[:2]
        ),
        row_label="Output",
        logical_key=target.logical_key,
        column_names=(
            "Credit Score System",
            "Indian EMI Calculator System",
        ),
    )

    with pytest.raises(
        TableEvidenceLookupError,
        match="ambiguous",
    ):
        (
            TableEvidenceLookupService()
            .lookup(
                request=request,
                evidence=(
                    ambiguous_target,
                ),
            )
        )


def test_labeled_row_comparison_rejects_wrong_row_label():

    source = generic_three_column_table(
        rows=(
            row(
                page=3,
                index=1,
                values=(
                    "Output",
                    "A 3-digit score",
                    "Monthly EMI",
                ),
            ),
        ),
    )

    atoms = (
        TableEvidenceAtomBuilder()
        .build_sequence(
            tables=(source,),
            context=generic_context(),
        )
    )

    target = atoms[0]

    assert target.logical_key is not None

    parts = target.logical_key.split("/")

    request = LabeledRowComparisonRequest(
        table_key="/".join(
            parts[:2]
        ),
        row_label="Purpose",
        logical_key=target.logical_key,
        column_names=(
            "Credit Score System",
            "Indian EMI Calculator System",
        ),
    )

    with pytest.raises(
        TableEvidenceLookupError,
        match="row label",
    ):
        (
            TableEvidenceLookupService()
            .lookup(
                request=request,
                evidence=atoms,
            )
        )
