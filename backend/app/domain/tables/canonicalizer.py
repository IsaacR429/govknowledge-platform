from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from app.domain.enums.table import (
    TableCellValueSource,
    TableEvidenceLayer,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    CanonicalTable,
    CanonicalTableCell,
    CanonicalTableRow,
    TableSourceSpan,
)
from app.domain.tables.evidence import (
    NormalizedTableCellEvidence,
    NormalizedTableEvidence,
    NormalizedTableRowEvidence,
)


class TableCanonicalizationError(
    ValueError
):
    pass


@dataclass(frozen=True)
class TableCanonicalizationResult:
    table: CanonicalTable
    source_text_cell_count: int
    observed_blank_count: int
    derived_blank_count: int


class TableCanonicalizer:
    """Combine aligned physical evidence with logical structure.

    Text and observed blank state come from PHYSICAL evidence.

    Logical cell existence / shape comes from STRUCTURAL evidence.

    An UNAVAILABLE physical value is never silently treated as
    source text. If a corresponding logical structural cell exists,
    it becomes a DERIVED empty canonical value so later structure
    resolution may reason over the logical blank explicitly.
    """

    def canonicalize(
        self,
        *,
        aligned_physical:
            NormalizedTableEvidence,
        structural:
            NormalizedTableEvidence,
        hierarchy_columns: tuple[
            int,
            ...
        ] = (),
    ) -> TableCanonicalizationResult:

        self._validate_tables(
            aligned_physical=(
                aligned_physical
            ),
            structural=structural,
        )

        canonical_rows = []

        source_text_count = 0
        observed_blank_count = 0
        derived_blank_count = 0

        for (
            physical_row,
            structural_row,
        ) in zip(
            aligned_physical.rows,
            structural.rows,
            strict=True,
        ):
            mappings = (
                self._map_row_cells(
                    physical_row=(
                        physical_row
                    ),
                    structural_row=(
                        structural_row
                    ),
                    column_count=(
                        structural.column_count
                    ),
                )
            )

            canonical_cells = []

            for (
                physical_cell,
                structural_cell,
            ) in mappings:

                (
                    canonical_cell,
                    classification,
                ) = self._canonical_cell(
                    physical_cell=(
                        physical_cell
                    ),
                    structural_cell=(
                        structural_cell
                    ),
                )

                canonical_cells.append(
                    canonical_cell
                )

                if (
                    classification
                    == "SOURCE_TEXT"
                ):
                    source_text_count += 1

                elif (
                    classification
                    == "OBSERVED_BLANK"
                ):
                    observed_blank_count += 1

                elif (
                    classification
                    == "DERIVED_BLANK"
                ):
                    derived_blank_count += 1

            role = (
                TableRowRole.HEADER
                if (
                    structural_row
                    .source_header_signal
                    is True
                )
                else TableRowRole.DATA
            )

            canonical_rows.append(
                CanonicalTableRow(
                    row_index=(
                        structural_row
                        .row_index
                    ),
                    role=role,
                    cells=tuple(
                        canonical_cells
                    ),
                    source_spans=(
                        physical_row
                        .source_spans
                        + structural_row
                        .source_spans
                    ),
                )
            )

        column_names = (
            self._derive_column_names(
                rows=tuple(
                    canonical_rows
                ),
                column_count=(
                    structural.column_count
                ),
            )
        )

        table = CanonicalTable(
            column_count=(
                structural.column_count
            ),
            rows=tuple(
                canonical_rows
            ),
            column_names=(
                column_names
            ),
            hierarchy_columns=(
                hierarchy_columns
            ),
            source_table_refs=(
                aligned_physical.source_ref,
                structural.source_ref,
            ),
            structural_identity_refs=tuple(
                dict.fromkeys(
                    span.source_ref.strip()
                    for span
                    in structural.source_spans
                    if span.source_ref.strip()
                )
            ),
        )

        return TableCanonicalizationResult(
            table=table,
            source_text_cell_count=(
                source_text_count
            ),
            observed_blank_count=(
                observed_blank_count
            ),
            derived_blank_count=(
                derived_blank_count
            ),
        )

    def _map_row_cells(
        self,
        *,
        physical_row:
            NormalizedTableRowEvidence,
        structural_row:
            NormalizedTableRowEvidence,
        column_count: int,
    ) -> tuple[
        tuple[
            NormalizedTableCellEvidence,
            NormalizedTableCellEvidence,
        ],
        ...
    ]:
        structural_cells = (
            tuple(
                structural_row.cells
            )
        )

        if not structural_cells:
            raise TableCanonicalizationError(
                "Structural row has no cells."
            )

        # A full logical row must preserve atomic column
        # positions, including unavailable parent cells.
        if self._is_full_atomic_row(
            structural_cells,
            column_count=column_count,
        ):
            physical_by_column = {
                cell.column_start:
                    cell
                for cell
                in physical_row.cells
                if (
                    cell.column_end
                    == cell.column_start + 1
                )
            }

            mappings = []

            for structural_cell in (
                structural_cells
            ):
                physical_cell = (
                    physical_by_column.get(
                        structural_cell
                        .column_start
                    )
                )

                if physical_cell is None:
                    raise TableCanonicalizationError(
                        "Aligned physical row does "
                        "not contain the required "
                        "logical column."
                    )

                mappings.append(
                    (
                        physical_cell,
                        structural_cell,
                    )
                )

            return tuple(
                mappings
            )

        # Sparse structural rows occur for merged headers,
        # section-like rows, and summaries. Their tag-cell
        # ordinal does not necessarily equal the physical
        # logical column. Match them by the ordered occupied
        # physical cells instead.
        semantic_cells = tuple(
            cell
            for cell
            in physical_row.cells
            if cell.has_semantic_text
        )

        if (
            len(semantic_cells)
            == len(structural_cells)
        ):
            selected = (
                semantic_cells
            )

        else:
            informative_cells = tuple(
                cell
                for cell
                in physical_row.cells
                if (
                    cell.has_semantic_text
                    or cell.observed_blank
                )
            )

            if (
                len(informative_cells)
                != len(structural_cells)
            ):
                raise TableCanonicalizationError(
                    "Sparse structural row cannot "
                    "be mapped uniquely to occupied "
                    "physical cells."
                )

            selected = (
                informative_cells
            )

        return tuple(
            zip(
                selected,
                structural_cells,
                strict=True,
            )
        )

    @staticmethod
    def _is_full_atomic_row(
        cells: tuple[
            NormalizedTableCellEvidence,
            ...
        ],
        *,
        column_count: int,
    ) -> bool:

        if len(cells) != column_count:
            return False

        expected = tuple(
            (
                column,
                column + 1,
            )
            for column in range(
                column_count
            )
        )

        actual = tuple(
            (
                cell.column_start,
                cell.column_end,
            )
            for cell in cells
        )

        return (
            actual == expected
        )

    def _canonical_cell(
        self,
        *,
        physical_cell:
            NormalizedTableCellEvidence,
        structural_cell:
            NormalizedTableCellEvidence,
    ) -> tuple[
        CanonicalTableCell,
        str,
    ]:

        structure_spans = (
            self._non_text_spans(
                physical_cell
                .source_spans
            )
            + structural_cell
            .source_spans
        )

        if physical_cell.has_semantic_text:

            assert (
                physical_cell.text
                is not None
            )

            origin_spans = (
                self._text_spans(
                    physical_cell
                    .source_spans
                )
            )

            if not origin_spans:
                raise (
                    TableCanonicalizationError(
                        "Semantic physical text "
                        "has no TEXT provenance."
                    )
                )

            return (
                CanonicalTableCell(
                    column_start=(
                        physical_cell
                        .column_start
                    ),
                    column_end=(
                        physical_cell
                        .column_end
                    ),
                    text=(
                        physical_cell.text
                    ),
                    value_source=(
                        TableCellValueSource
                        .SOURCE_TEXT
                    ),
                    origin_spans=(
                        origin_spans
                    ),
                    structure_spans=(
                        structure_spans
                    ),
                ),
                "SOURCE_TEXT",
            )

        if physical_cell.observed_blank:

            geometry_spans = (
                self._geometry_spans(
                    physical_cell
                    .source_spans
                )
            )

            return (
                CanonicalTableCell(
                    column_start=(
                        physical_cell
                        .column_start
                    ),
                    column_end=(
                        physical_cell
                        .column_end
                    ),
                    text="",
                    value_source=(
                        TableCellValueSource
                        .SOURCE_TEXT
                    ),
                    origin_spans=(
                        geometry_spans
                    ),
                    structure_spans=(
                        structural_cell
                        .source_spans
                    ),
                ),
                "OBSERVED_BLANK",
            )

        # Physical extraction could not independently
        # determine a value, but the tag tree confirms that
        # a logical cell exists at this position.
        #
        # This is a logical/derived blank, not source text.
        return (
            CanonicalTableCell(
                column_start=(
                    physical_cell
                    .column_start
                ),
                column_end=(
                    physical_cell
                    .column_end
                ),
                text="",
                value_source=(
                    TableCellValueSource
                    .DERIVED
                ),
                origin_spans=(),
                structure_spans=(
                    structure_spans
                ),
            ),
            "DERIVED_BLANK",
        )

    @staticmethod
    def _text_spans(
        spans: tuple[
            TableSourceSpan,
            ...
        ],
    ) -> tuple[
        TableSourceSpan,
        ...
    ]:
        return tuple(
            span
            for span in spans
            if (
                span.kind
                == TableSpanKind.TEXT
                and span.text is not None
                and span.text.strip()
            )
        )

    @staticmethod
    def _geometry_spans(
        spans: tuple[
            TableSourceSpan,
            ...
        ],
    ) -> tuple[
        TableSourceSpan,
        ...
    ]:
        return tuple(
            span
            for span in spans
            if (
                span.kind
                == TableSpanKind.GEOMETRY
            )
        )

    @staticmethod
    def _non_text_spans(
        spans: tuple[
            TableSourceSpan,
            ...
        ],
    ) -> tuple[
        TableSourceSpan,
        ...
    ]:
        return tuple(
            span
            for span in spans
            if (
                span.kind
                != TableSpanKind.TEXT
            )
        )

    @staticmethod
    def _normalize_metadata_text(
        value: str,
    ) -> str:
        return unicodedata.normalize(
            "NFC",
            " ".join(
                value.split()
            ),
        )

    @staticmethod
    def _derive_column_names(
        *,
        rows: tuple[
            CanonicalTableRow,
            ...
        ],
        column_count: int,
    ) -> tuple[
        str,
        ...
    ]:

        headers = tuple(
            row
            for row in rows
            if (
                row.role
                == TableRowRole.HEADER
            )
        )

        if len(headers) != 1:
            return ()

        header = headers[0]

        by_column = {
            cell.column_start:
                cell
            for cell in header.cells
            if (
                cell.column_end
                == cell.column_start + 1
            )
        }

        if len(by_column) != column_count:
            return ()

        names = []

        for column in range(
            column_count
        ):
            cell = (
                by_column.get(
                    column
                )
            )

            if (
                cell is None
                or not cell.text.strip()
            ):
                return ()

            names.append(
                TableCanonicalizer
                ._normalize_metadata_text(
                    cell.text
                )
            )

        return tuple(
            names
        )

    @staticmethod
    def _validate_tables(
        *,
        aligned_physical:
            NormalizedTableEvidence,
        structural:
            NormalizedTableEvidence,
    ) -> None:

        if (
            aligned_physical.layer
            != TableEvidenceLayer.PHYSICAL
        ):
            raise ValueError(
                "aligned_physical must use "
                "PHYSICAL layer."
            )

        if (
            structural.layer
            != TableEvidenceLayer.STRUCTURAL
        ):
            raise ValueError(
                "structural must use "
                "STRUCTURAL layer."
            )

        if (
            aligned_physical.page_number
            != structural.page_number
        ):
            raise ValueError(
                "Canonicalization requires "
                "same-page evidence."
            )

        if (
            aligned_physical.column_count
            != structural.column_count
        ):
            raise ValueError(
                "Physical and structural "
                "column counts must match."
            )

        if (
            len(aligned_physical.rows)
            != len(structural.rows)
        ):
            raise (
                TableCanonicalizationError(
                    "Physical and structural "
                    "row counts must match."
                )
            )

        physical_indexes = tuple(
            row.row_index
            for row
            in aligned_physical.rows
        )

        structural_indexes = tuple(
            row.row_index
            for row
            in structural.rows
        )

        if (
            physical_indexes
            != structural_indexes
        ):
            raise (
                TableCanonicalizationError(
                    "Aligned physical row indexes "
                    "must match structural row "
                    "indexes."
                )
            )
