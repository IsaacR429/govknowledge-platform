import pytest

from app.domain.enums.table import (
    TableCellTextStatus,
    TableEvidenceLayer,
    TableSpanKind,
)
from app.domain.models.table import (
    TableSourceSpan,
)
from app.domain.tables.evidence import (
    NormalizedTableCellEvidence,
    NormalizedTableEvidence,
    NormalizedTableEvidenceBundle,
    NormalizedTableRowEvidence,
)


def span(
    *,
    page: int,
    ref: str,
) -> TableSourceSpan:
    return TableSourceSpan(
        page_number=page,
        kind=TableSpanKind.STRUCTURE,
        source_ref=ref,
    )


def observed_cell(
    *,
    column: int,
    text: str,
    page: int = 1,
    role: str | None = None,
) -> NormalizedTableCellEvidence:
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
                ref=f"cell:{column}",
            ),
        ),
        source_role=role,
    )


def test_observed_blank_is_not_unavailable() -> None:
    cell = NormalizedTableCellEvidence(
        column_start=0,
        column_end=1,
        text_status=(
            TableCellTextStatus.OBSERVED
        ),
        text="",
    )

    assert cell.text_observed is True
    assert cell.observed_blank is True
    assert cell.has_semantic_text is False


def test_unavailable_text_is_not_observed_blank() -> None:
    cell = NormalizedTableCellEvidence(
        column_start=0,
        column_end=1,
        text_status=(
            TableCellTextStatus.UNAVAILABLE
        ),
        text=None,
    )

    assert cell.text_observed is False
    assert cell.observed_blank is False
    assert cell.has_semantic_text is False


def test_observed_requires_text_value() -> None:
    with pytest.raises(
        ValueError,
        match="OBSERVED",
    ):
        NormalizedTableCellEvidence(
            column_start=0,
            column_end=1,
            text_status=(
                TableCellTextStatus.OBSERVED
            ),
            text=None,
        )


def test_unavailable_rejects_fake_text() -> None:
    with pytest.raises(
        ValueError,
        match="UNAVAILABLE",
    ):
        NormalizedTableCellEvidence(
            column_start=0,
            column_end=1,
            text_status=(
                TableCellTextStatus.UNAVAILABLE
            ),
            text="",
        )


def test_source_role_is_preserved_without_canonicalizing_it() -> None:
    cell = observed_cell(
        column=0,
        text="01",
        role="TH",
    )

    assert cell.source_role == "TH"

    # The normalized evidence layer has deliberately
    # not converted this into a canonical row role.
    assert not hasattr(
        cell,
        "role",
    )


def test_physical_and_structural_evidence_may_disagree() -> None:
    physical = NormalizedTableEvidence(
        source_ref="native:p9:t1",
        page_number=9,
        layer=TableEvidenceLayer.PHYSICAL,
        column_count=9,
        rows=(),
    )

    structural = NormalizedTableEvidence(
        source_ref="tag:xref:1930",
        page_number=9,
        layer=TableEvidenceLayer.STRUCTURAL,
        column_count=3,
        rows=(),
    )

    bundle = NormalizedTableEvidenceBundle(
        document_ref="document:test",
        tables=(
            physical,
            structural,
        ),
    )

    page = bundle.for_page(9)

    assert len(page) == 2

    assert {
        table.column_count
        for table in page
    } == {
        9,
        3,
    }


def test_structural_cell_can_span_columns() -> None:
    cell = NormalizedTableCellEvidence(
        column_start=0,
        column_end=5,
        text_status=(
            TableCellTextStatus.OBSERVED
        ),
        text="Lot 1",
        source_role="TH",
    )

    assert cell.column_span == 5


def test_cell_cannot_exceed_evidence_schema() -> None:
    row = NormalizedTableRowEvidence(
        row_index=0,
        cells=(
            NormalizedTableCellEvidence(
                column_start=0,
                column_end=4,
                text_status=(
                    TableCellTextStatus.OBSERVED
                ),
                text="too wide",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        NormalizedTableEvidence(
            source_ref="tag:test",
            page_number=1,
            layer=(
                TableEvidenceLayer.STRUCTURAL
            ),
            column_count=3,
            rows=(
                row,
            ),
        )


def test_raw_evidence_cannot_borrow_cross_page_origin() -> None:
    row = NormalizedTableRowEvidence(
        row_index=0,
        cells=(
            NormalizedTableCellEvidence(
                column_start=0,
                column_end=1,
                text_status=(
                    TableCellTextStatus.OBSERVED
                ),
                text="Domain A",
                source_spans=(
                    span(
                        page=9,
                        ref="wrong-page-origin",
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="another page",
    ):
        NormalizedTableEvidence(
            source_ref="native:p10:t0",
            page_number=10,
            layer=TableEvidenceLayer.PHYSICAL,
            column_count=1,
            rows=(
                row,
            ),
        )
