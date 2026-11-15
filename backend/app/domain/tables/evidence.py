from __future__ import annotations

from dataclasses import dataclass

from app.domain.enums.table import (
    TableCellTextStatus,
    TableEvidenceLayer,
)
from app.domain.models.table import (
    TableBBox,
    TableSourceSpan,
)


@dataclass(frozen=True)
class NormalizedTableCellEvidence:
    """Parser-neutral evidence for one observed table cell.

    This is NOT yet a CanonicalTableCell.

    column_start / column_end refer only to the schema
    reported by this particular evidence source.

    Examples:

        Native PyMuPDF:
            physical schema may contain 9 columns.

        Tagged PDF:
            structural schema may contain 3 columns.

    Both are valid observations. Reconciliation happens later.
    """

    column_start: int
    column_end: int

    text_status: TableCellTextStatus
    text: str | None

    source_spans: tuple[
        TableSourceSpan,
        ...
    ] = ()

    # Preserve what the source/parser declared without
    # treating it as canonical semantics.
    #
    # Examples:
    #     "TH"
    #     "TD"
    #     "native-header"
    #
    source_role: str | None = None

    def __post_init__(self) -> None:
        if self.column_start < 0:
            raise ValueError(
                "Normalized cell column_start must be >= 0."
            )

        if self.column_end <= self.column_start:
            raise ValueError(
                "Normalized cell column_end must be greater "
                "than column_start."
            )

        if (
            self.text_status
            == TableCellTextStatus.OBSERVED
            and self.text is None
        ):
            raise ValueError(
                "OBSERVED cell text cannot be None."
            )

        if (
            self.text_status
            == TableCellTextStatus.UNAVAILABLE
            and self.text is not None
        ):
            raise ValueError(
                "UNAVAILABLE cell text must be None."
            )

        if (
            self.source_role is not None
            and not self.source_role.strip()
        ):
            raise ValueError(
                "source_role cannot be blank."
            )

    @property
    def text_observed(self) -> bool:
        return (
            self.text_status
            == TableCellTextStatus.OBSERVED
        )

    @property
    def observed_blank(self) -> bool:
        return (
            self.text_observed
            and self.text is not None
            and not self.text.strip()
        )

    @property
    def has_semantic_text(self) -> bool:
        return (
            self.text_observed
            and self.text is not None
            and bool(self.text.strip())
        )

    @property
    def column_span(self) -> int:
        return (
            self.column_end
            - self.column_start
        )


@dataclass(frozen=True)
class NormalizedTableRowEvidence:
    """One row as reported by an evidence source.

    source_header_signal means the parser/source declared or
    strongly marked this row as a header.

    It is evidence only. It does not become canonical HEADER
    merely because the source says so.
    """

    row_index: int

    cells: tuple[
        NormalizedTableCellEvidence,
        ...
    ]

    source_spans: tuple[
        TableSourceSpan,
        ...
    ] = ()

    source_header_signal: bool | None = None

    source_role: str | None = None

    def __post_init__(self) -> None:
        if self.row_index < 0:
            raise ValueError(
                "Normalized row_index must be >= 0."
            )

        if (
            self.source_role is not None
            and not self.source_role.strip()
        ):
            raise ValueError(
                "row source_role cannot be blank."
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
                    "Normalized row cells cannot overlap."
                )

            previous_end = cell.column_end


@dataclass(frozen=True)
class NormalizedTableEvidence:
    """One table observation from one normalized evidence layer.

    A physical and structural observation of the same visible
    table are allowed to disagree about dimensions.

    Example already observed during qualification:

        physical:
            11 rows x 9 columns

        structural:
            10 rows x 3 columns

    Neither observation is silently rewritten here.
    """

    source_ref: str
    page_number: int

    layer: TableEvidenceLayer

    column_count: int

    rows: tuple[
        NormalizedTableRowEvidence,
        ...
    ]

    bbox: TableBBox | None = None

    source_spans: tuple[
        TableSourceSpan,
        ...
    ] = ()

    def __post_init__(self) -> None:
        if not self.source_ref.strip():
            raise ValueError(
                "Normalized table source_ref cannot be empty."
            )

        if self.page_number < 1:
            raise ValueError(
                "Normalized table page_number must be >= 1."
            )

        if self.column_count < 1:
            raise ValueError(
                "Normalized table column_count must be >= 1."
            )

        seen_indexes: set[int] = set()

        for row in self.rows:
            if row.row_index in seen_indexes:
                raise ValueError(
                    "Normalized table row indexes must be unique."
                )

            seen_indexes.add(
                row.row_index
            )

            for cell in row.cells:
                if cell.column_end > self.column_count:
                    raise ValueError(
                        "Normalized cell exceeds the evidence "
                        "table column count."
                    )

                self._validate_span_pages(
                    cell.source_spans
                )

            self._validate_span_pages(
                row.source_spans
            )

        self._validate_span_pages(
            self.source_spans
        )

    def _validate_span_pages(
        self,
        spans: tuple[
            TableSourceSpan,
            ...
        ],
    ) -> None:
        for span in spans:
            if span.page_number != self.page_number:
                raise ValueError(
                    "Raw normalized evidence cannot borrow "
                    "provenance from another page. Cross-page "
                    "inheritance belongs to canonicalization."
                )

    @property
    def row_count(self) -> int:
        return len(
            self.rows
        )


@dataclass(frozen=True)
class NormalizedTableEvidenceBundle:
    """All normalized table observations for one document.

    Multiple evidence layers may describe the same page/table.
    """

    document_ref: str

    tables: tuple[
        NormalizedTableEvidence,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.document_ref.strip():
            raise ValueError(
                "document_ref cannot be empty."
            )

    def for_page(
        self,
        page_number: int,
    ) -> tuple[
        NormalizedTableEvidence,
        ...
    ]:
        return tuple(
            table
            for table in self.tables
            if table.page_number
            == page_number
        )

    def for_layer(
        self,
        layer: TableEvidenceLayer,
    ) -> tuple[
        NormalizedTableEvidence,
        ...
    ]:
        return tuple(
            table
            for table in self.tables
            if table.layer == layer
        )
