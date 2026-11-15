from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.domain.enums.table import (
    TableCellValueSource,
    TableContinuationDecision,
    TableRowRole,
    TableSpanKind,
)


@dataclass(frozen=True)
class TableBBox:
    """Page-coordinate bounding box.

    The domain model deliberately does not depend on PyMuPDF,
    Docling, or any other parser geometry type.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0:
            raise ValueError(
                "TableBBox.x1 must be greater than or equal to x0."
            )

        if self.y1 < self.y0:
            raise ValueError(
                "TableBBox.y1 must be greater than or equal to y0."
            )


@dataclass(frozen=True)
class TableSourceSpan:
    """Traceable source evidence contributing to canonical structure/value.

    source_ref is parser-neutral and opaque to the domain layer.

    Examples:
        native:p4:t0:r3:c2
        tag:xref:2034
        tag:row:xref:2030
    """

    page_number: int
    kind: TableSpanKind
    source_ref: str
    bbox: TableBBox | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError(
                "TableSourceSpan.page_number must be >= 1."
            )

        if not self.source_ref.strip():
            raise ValueError(
                "TableSourceSpan.source_ref cannot be empty."
            )


@dataclass(frozen=True)
class CanonicalTableCell:
    """One logical table cell.

    column_end is exclusive.

    A normal three-column row therefore uses:
        0..1
        1..2
        2..3

    A section spanning all five logical columns may use:
        0..5
    """

    column_start: int
    column_end: int
    text: str

    value_source: TableCellValueSource

    origin_spans: tuple[TableSourceSpan, ...] = ()
    structure_spans: tuple[TableSourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if self.column_start < 0:
            raise ValueError(
                "CanonicalTableCell.column_start must be >= 0."
            )

        if self.column_end <= self.column_start:
            raise ValueError(
                "CanonicalTableCell.column_end must be greater "
                "than column_start."
            )

        if self.value_source == TableCellValueSource.INHERITED:
            if not self.origin_spans:
                raise ValueError(
                    "Inherited canonical cells must preserve "
                    "the source span from which the value "
                    "was inherited."
                )

        if self.text and not self.origin_spans:
            if self.value_source == TableCellValueSource.SOURCE_TEXT:
                raise ValueError(
                    "SOURCE_TEXT cells must preserve at least "
                    "one origin span."
                )

    @property
    def inherited(self) -> bool:
        return (
            self.value_source
            == TableCellValueSource.INHERITED
        )

    @property
    def column_span(self) -> int:
        return (
            self.column_end
            - self.column_start
        )


@dataclass(frozen=True)
class CanonicalTableRow:
    """One logical row.

    source_spans describe where the row itself exists physically.

    Individual cells preserve their own value provenance separately.
    This is important for cross-page inheritance:

        row source span -> page 10
        inherited parent cell origin -> page 9
    """

    row_index: int
    role: TableRowRole

    cells: tuple[CanonicalTableCell, ...]

    source_spans: tuple[TableSourceSpan, ...] = ()

    logical_key: str | None = None
    section_key: str | None = None

    def __post_init__(self) -> None:
        if self.row_index < 0:
            raise ValueError(
                "CanonicalTableRow.row_index must be >= 0."
            )

        if not self.cells:
            raise ValueError(
                "CanonicalTableRow must contain at least one cell."
            )

        ordered = sorted(
            self.cells,
            key=lambda cell: (
                cell.column_start,
                cell.column_end,
            ),
        )

        previous_end: int | None = None

        for cell in ordered:
            if (
                previous_end is not None
                and cell.column_start < previous_end
            ):
                raise ValueError(
                    "CanonicalTableRow cells cannot overlap."
                )

            previous_end = cell.column_end

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    span.page_number
                    for span in self.source_spans
                }
            )
        )


@dataclass(frozen=True)
class CanonicalTable:
    """Parser-neutral logical representation of a table."""

    column_count: int
    rows: tuple[CanonicalTableRow, ...]

    column_names: tuple[str, ...] = ()

    # Columns whose blank values may inherit the latest
    # active parent value after structural validation.
    hierarchy_columns: tuple[int, ...] = ()

    # Page-local / observation provenance.
    source_table_refs: tuple[str, ...] = ()

    # Stable logical identities supplied by structural evidence.
    #
    # These are deliberately separate from source_table_refs:
    # physical observations remain page-local while a logical
    # structural table may span multiple physical pages.
    structural_identity_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.column_count < 1:
            raise ValueError(
                "CanonicalTable.column_count must be >= 1."
            )

        if (
            self.column_names
            and len(self.column_names)
            != self.column_count
        ):
            raise ValueError(
                "column_names length must equal column_count."
            )

        if len(set(self.hierarchy_columns)) != len(
            self.hierarchy_columns
        ):
            raise ValueError(
                "hierarchy_columns cannot contain duplicates."
            )

        for column in self.hierarchy_columns:
            if not 0 <= column < self.column_count:
                raise ValueError(
                    "hierarchy column is outside the table schema."
                )

        seen_row_indexes: set[int] = set()

        for row in self.rows:
            if row.row_index in seen_row_indexes:
                raise ValueError(
                    "CanonicalTable row indexes must be unique."
                )

            seen_row_indexes.add(
                row.row_index
            )

            for cell in row.cells:
                if cell.column_end > self.column_count:
                    raise ValueError(
                        "Canonical cell exceeds table column count."
                    )

    @property
    def page_numbers(self) -> tuple[int, ...]:
        pages: set[int] = set()

        for row in self.rows:
            pages.update(
                row.page_numbers
            )

        return tuple(
            sorted(pages)
        )

    @property
    def data_rows(
        self,
    ) -> tuple[CanonicalTableRow, ...]:
        return tuple(
            row
            for row in self.rows
            if row.role == TableRowRole.DATA
        )


@dataclass(frozen=True)
class TableContinuationEvidence:
    """Evidence used by the continuation resolver.

    This object records the facts. It does not itself decide policy.
    """

    left_source_ref: str
    right_source_ref: str

    left_page: int
    right_page: int

    same_column_count: bool
    compatible_column_geometry: bool

    repeated_header: bool
    headerless_continuation: bool

    row_key_progression: bool

    # True only when both canonical fragments retain at least one
    # identical stable structural table identity.
    shared_structural_identity: bool = False

    left_has_active_hierarchy: bool = False
    right_starts_with_inherited_cells: bool = False

    contradictory_boundary: bool = False

    def __post_init__(self) -> None:
        if self.left_page < 1 or self.right_page < 1:
            raise ValueError(
                "Continuation pages must be >= 1."
            )

        if self.right_page <= self.left_page:
            raise ValueError(
                "right_page must come after left_page."
            )


@dataclass(frozen=True)
class TableContinuationResult:
    decision: TableContinuationDecision
    evidence: TableContinuationEvidence
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ValueError(
                "Continuation decisions must preserve "
                "at least one deterministic reason."
            )


def source_pages(
    spans: Iterable[TableSourceSpan],
) -> tuple[int, ...]:
    """Return unique pages represented by source spans."""

    return tuple(
        sorted(
            {
                span.page_number
                for span in spans
            }
        )
    )
