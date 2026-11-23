from dataclasses import replace
from uuid import uuid4

import pytest

from app.application.services.table_claim_provenance import (
    DirectTableClaimBuilder,
    TableClaimProvenanceError,
)
from app.application.services.table_evidence_atoms import (
    LabeledRowComparisonRequest,
    TableComparisonSelection,
    TableCellEvidenceAtom,
    TableEvidenceAtom,
    TableLookupRequest,
    TableLookupSelection,
)
from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    TableSourceSpan,
)


def span(
    *,
    page=4,
    kind=TableSpanKind.TEXT,
    text="EZAF",
):
    return TableSourceSpan(
        page_number=page,
        kind=kind,
        source_ref=(
            f"fixture:p{page}:"
            f"{kind.value}"
        ),
        bbox=None,
        text=text,
    )


def selection(
    *,
    logical_key="LOT_2/05",
    row_page=4,
    origin_page=4,
    column_name="Soumissionnaires",
    text=(
        "ENTREPRISE ZERBO AWA "
        "ET FRERES (EZAF)"
    ),
    value_source=(
        TableCellValueSource
        .SOURCE_TEXT
    ),
    origin=True,
):

    cell = TableCellEvidenceAtom(
        column_start=1,
        column_end=2,
        column_name=column_name,
        text=text,
        value_source=(
            value_source
        ),
        origin_spans=(
            (
                span(
                    page=origin_page,
                    text=text,
                ),
            )
            if origin
            else ()
        ),
        structure_spans=(
            span(
                page=row_page,
                kind=(
                    TableSpanKind
                    .STRUCTURE
                ),
                text=None,
            ),
        ),
    )

    row = TableEvidenceAtom(
        evidence_id=(
            "table:test:"
            f"{logical_key}"
        ),
        document_id=uuid4(),
        document_version_id=(
            uuid4()
        ),
        page_id=uuid4(),
        page_number=row_page,
        source_table_refs=(
            "fixture-table",
        ),
        row_index=5,
        role=TableRowRole.DATA,
        logical_key=logical_key,
        section_key="LOT_2",
        column_names=(
            "N°",
            "Soumissionnaires",
            "Observations",
        ),
        cells=(
            TableCellEvidenceAtom(
                column_start=0,
                column_end=1,
                column_name="N°",
                text="05",
                value_source=(
                    TableCellValueSource
                    .SOURCE_TEXT
                ),
                origin_spans=(
                    span(
                        page=row_page,
                        text="05",
                    ),
                ),
                structure_spans=(),
            ),
            cell,
        ),
        language="fr",
    )

    request = TableLookupRequest(
        section_key="LOT_2",
        ordinal=5,
        column_name=(
            "Soumissionnaires"
        ),
    )

    return (
        request,
        TableLookupSelection(
            request=request,
            row=row,
            cell=cell,
        ),
    )


def test_builds_direct_source_backed_claim():
    request, selected = selection()

    claim = (
        DirectTableClaimBuilder()
        .build(
            request=request,
            selection=selected,
        )
    )

    assert (
        claim.text
        == (
            "ENTREPRISE ZERBO AWA "
            "ET FRERES (EZAF)"
        )
    )

    assert (
        claim.origin.logical_key
        == "LOT_2/05"
    )

    assert (
        claim.origin.column_name
        == "Soumissionnaires"
    )

    assert (
        claim.origin.page_number
        == 4
    )

    assert (
        claim.origin.origin_pages
        == (4,)
    )

    assert (
        claim.origin.structure_pages
        == (4,)
    )


def test_preserves_real_source_reference():
    request, selected = selection()

    claim = (
        DirectTableClaimBuilder()
        .build(
            request=request,
            selection=selected,
        )
    )

    assert (
        claim.origin.source_refs
        == (
            "fixture:p4:TEXT",
        )
    )


def test_wrong_logical_key_fails_closed():
    request, selected = selection(
        logical_key="LOT_1/05"
    )

    with pytest.raises(
        TableClaimProvenanceError,
        match="logical key",
    ):
        (
            DirectTableClaimBuilder()
            .build(
                request=request,
                selection=selected,
            )
        )


def test_wrong_column_fails_closed():
    request, selected = selection(
        column_name="Observations"
    )

    with pytest.raises(
        TableClaimProvenanceError,
        match="requested column",
    ):
        (
            DirectTableClaimBuilder()
            .build(
                request=request,
                selection=selected,
            )
        )


def test_inherited_value_fails_closed_for_direct_path():
    request, selected = selection(
        value_source=(
            TableCellValueSource
            .INHERITED
        )
    )

    with pytest.raises(
        TableClaimProvenanceError,
        match="direct source text",
    ):
        (
            DirectTableClaimBuilder()
            .build(
                request=request,
                selection=selected,
            )
        )


def test_cross_page_origin_fails_closed():
    request, selected = selection(
        row_page=4,
        origin_page=3,
    )

    with pytest.raises(
        TableClaimProvenanceError,
        match="Cross-page",
    ):
        (
            DirectTableClaimBuilder()
            .build(
                request=request,
                selection=selected,
            )
        )


def test_missing_origin_fails_closed():
    request, selected = selection(
        origin=False
    )

    with pytest.raises(
        TableClaimProvenanceError,
        match="source provenance",
    ):
        (
            DirectTableClaimBuilder()
            .build(
                request=request,
                selection=selected,
            )
        )


def test_comparison_builds_one_direct_claim_per_selected_cell():

    _, base_selection = selection()

    table_key = (
        "GENERIC/"
        + "a" * 64
    )

    logical_key = (
        table_key
        + "/"
        + "b" * 64
    )

    label = TableCellEvidenceAtom(
        column_start=0,
        column_end=1,
        column_name="Crit\u00e8res",
        text="Chloroplastes",
        value_source=(
            TableCellValueSource.SOURCE_TEXT
        ),
        origin_spans=(
            span(
                page=4,
                text="Chloroplastes",
            ),
        ),
        structure_spans=(),
    )

    animal = TableCellEvidenceAtom(
        column_start=1,
        column_end=2,
        column_name="Celluleanimale",
        text="Absents",
        value_source=(
            TableCellValueSource.SOURCE_TEXT
        ),
        origin_spans=(
            span(
                page=4,
                text="Absents",
            ),
        ),
        structure_spans=(
            span(
                page=4,
                kind=(
                    TableSpanKind.STRUCTURE
                ),
            ),
        ),
    )

    plant = TableCellEvidenceAtom(
        column_start=2,
        column_end=3,
        column_name=(
            "Cellulev\u00e9g\u00e9tale"
        ),
        text=(
            "Pr\u00e9sents "
            "(contiennent la chlorophylle "
            "pour la photosynth\u00e8se)"
        ),
        value_source=(
            TableCellValueSource.SOURCE_TEXT
        ),
        origin_spans=(
            span(
                page=4,
                text=(
                    "Pr\u00e9sents "
                    "(contiennent la chlorophylle "
                    "pour la photosynth\u00e8se)"
                ),
            ),
        ),
        structure_spans=(
            span(
                page=4,
                kind=(
                    TableSpanKind.STRUCTURE
                ),
            ),
        ),
    )

    evidence_row = replace(
        base_selection.row,
        logical_key=logical_key,
        section_key=None,
        column_names=(
            "Crit\u00e8res",
            "Celluleanimale",
            "Cellulev\u00e9g\u00e9tale",
        ),
        cells=(
            label,
            animal,
            plant,
        ),
    )

    request = LabeledRowComparisonRequest(
        table_key=table_key,
        row_label="Chloroplastes",
        logical_key=logical_key,
        column_names=(
            "Celluleanimale",
            "Cellulev\u00e9g\u00e9tale",
        ),
    )

    selected = TableComparisonSelection(
        request=request,
        row=evidence_row,
        cells=(
            animal,
            plant,
        ),
    )

    claims = (
        DirectTableClaimBuilder()
        .build_comparison(
            request=request,
            selection=selected,
        )
    )

    assert len(claims) == 2

    assert (
        tuple(
            claim.claim_id
            for claim in claims
        )
        == (
            "table-claim-1",
            "table-claim-2",
        )
    )

    assert (
        tuple(
            claim.text
            for claim in claims
        )
        == (
            "Absents",
            (
                "Pr\u00e9sents "
                "(contiennent la chlorophylle "
                "pour la photosynth\u00e8se)"
            ),
        )
    )

    assert (
        tuple(
            claim.origin.column_name
            for claim in claims
        )
        == (
            "Celluleanimale",
            "Cellulev\u00e9g\u00e9tale",
        )
    )

    assert all(
        claim.origin.logical_key
        == logical_key
        for claim in claims
    )

    assert all(
        claim.origin.page_number == 4
        for claim in claims
    )
