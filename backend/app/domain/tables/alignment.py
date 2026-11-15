from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.domain.enums.table import (
    TableCellTextStatus,
    TableEvidenceLayer,
    TableRowAlignmentKind,
    TableSpanKind,
)
from app.domain.tables.evidence import (
    NormalizedTableCellEvidence,
    NormalizedTableEvidence,
    NormalizedTableRowEvidence,
)


class TableRowAlignmentError(
    ValueError
):
    pass


@dataclass(frozen=True)
class TableRowAlignmentGroup:
    structural_row_index: int
    physical_row_indexes: tuple[
        int,
        ...
    ]
    kind: TableRowAlignmentKind

    def __post_init__(self) -> None:
        if self.structural_row_index < 0:
            raise ValueError(
                "structural_row_index must "
                "be non-negative."
            )

        if not self.physical_row_indexes:
            raise ValueError(
                "physical_row_indexes must "
                "not be empty."
            )


@dataclass(frozen=True)
class TableRowAlignmentResult:
    aligned_physical: (
        NormalizedTableEvidence
    )
    groups: tuple[
        TableRowAlignmentGroup,
        ...
    ]


@dataclass(frozen=True)
class _CollapsedRow:
    row: NormalizedTableRowEvidence
    physical_row_indexes: tuple[
        int,
        ...
    ]
    kind: TableRowAlignmentKind


class TableRowAligner:
    """Align projected physical rows to structural rows.

    The aligner has two deterministic mechanisms:

    1. physical continuation fragments are collapsed when
       their rendered TEXT geometry lies inside the previous
       logical cell GEOMETRY;

    2. remaining row-count differences may be reconciled only
       against sparse structural rows, using consecutive
       physical rows with the same sparse semantic-column mask.

    Ambiguous alignments fail closed.
    """

    _GEOMETRY_TOLERANCE = 1.0

    def align(
        self,
        *,
        projected_physical:
            NormalizedTableEvidence,
        structural:
            NormalizedTableEvidence,
    ) -> TableRowAlignmentResult:

        self._validate_tables(
            projected_physical=(
                projected_physical
            ),
            structural=structural,
        )

        collapsed = (
            self._collapse_wrapped_rows(
                projected_physical.rows,
                column_count=(
                    projected_physical
                    .column_count
                ),
            )
        )

        paths = self._find_alignments(
            collapsed=collapsed,
            structural=structural,
        )

        if not paths:
            raise TableRowAlignmentError(
                "No valid row alignment exists."
            )

        if len(paths) != 1:
            raise TableRowAlignmentError(
                "Ambiguous row alignment."
            )

        path = paths[0]

        aligned_rows = []
        groups = []

        for (
            structural_position,
            (
                start,
                end,
            ),
        ) in enumerate(path):

            structural_row = (
                structural.rows[
                    structural_position
                ]
            )

            selected = (
                collapsed[
                    start:end
                ]
            )

            merged_row = (
                self._merge_rows(
                    tuple(
                        item.row
                        for item in selected
                    ),
                    row_index=(
                        structural_row.row_index
                    ),
                    column_count=(
                        structural.column_count
                    ),
                )
            )

            physical_indexes = tuple(
                physical_index
                for item in selected
                for physical_index
                in item.physical_row_indexes
            )

            if len(selected) > 1:
                alignment_kind = (
                    TableRowAlignmentKind
                    .STRUCTURAL_GROUP
                )

            elif (
                selected[0].kind
                == TableRowAlignmentKind
                .WRAPPED_CONTINUATION
            ):
                alignment_kind = (
                    TableRowAlignmentKind
                    .WRAPPED_CONTINUATION
                )

            else:
                alignment_kind = (
                    TableRowAlignmentKind
                    .ONE_TO_ONE
                )

            aligned_rows.append(
                merged_row
            )

            groups.append(
                TableRowAlignmentGroup(
                    structural_row_index=(
                        structural_row
                        .row_index
                    ),
                    physical_row_indexes=(
                        physical_indexes
                    ),
                    kind=(
                        alignment_kind
                    ),
                )
            )

        aligned = (
            NormalizedTableEvidence(
                source_ref=(
                    f"{projected_physical.source_ref}"
                    f":row-aligned:"
                    f"{structural.source_ref}"
                ),
                page_number=(
                    projected_physical
                    .page_number
                ),
                layer=(
                    TableEvidenceLayer.PHYSICAL
                ),
                column_count=(
                    structural.column_count
                ),
                rows=tuple(
                    aligned_rows
                ),
                bbox=(
                    projected_physical.bbox
                ),
                source_spans=(
                    projected_physical
                    .source_spans
                ),
            )
        )

        return TableRowAlignmentResult(
            aligned_physical=aligned,
            groups=tuple(
                groups
            ),
        )

    def _collapse_wrapped_rows(
        self,
        rows: tuple[
            NormalizedTableRowEvidence,
            ...
        ],
        *,
        column_count: int,
    ) -> tuple[
        _CollapsedRow,
        ...
    ]:

        collapsed: list[
            _CollapsedRow
        ] = []

        for row in rows:

            current = _CollapsedRow(
                row=row,
                physical_row_indexes=(
                    row.row_index,
                ),
                kind=(
                    TableRowAlignmentKind
                    .ONE_TO_ONE
                ),
            )

            if (
                collapsed
                and self
                ._is_wrapped_continuation(
                    previous=(
                        collapsed[-1].row
                    ),
                    current=row,
                )
            ):
                previous = (
                    collapsed[-1]
                )

                merged = (
                    self._merge_rows(
                        (
                            previous.row,
                            row,
                        ),
                        row_index=(
                            previous
                            .row
                            .row_index
                        ),
                        column_count=(
                            column_count
                        ),
                    )
                )

                collapsed[-1] = (
                    _CollapsedRow(
                        row=merged,
                        physical_row_indexes=(
                            previous
                            .physical_row_indexes
                            + (
                                row.row_index,
                            )
                        ),
                        kind=(
                            TableRowAlignmentKind
                            .WRAPPED_CONTINUATION
                        ),
                    )
                )

                continue

            collapsed.append(
                current
            )

        return tuple(
            collapsed
        )

    def _is_wrapped_continuation(
        self,
        *,
        previous:
            NormalizedTableRowEvidence,
        current:
            NormalizedTableRowEvidence,
    ) -> bool:

        if (
            previous.source_header_signal
            is True
            or current.source_header_signal
            is True
        ):
            return False

        current_semantic = {
            cell.column_start:
                cell
            for cell in current.cells
            if cell.has_semantic_text
        }

        if not current_semantic:
            return False

        previous_by_column = {
            cell.column_start:
                cell
            for cell in previous.cells
        }

        for (
            column,
            current_cell,
        ) in current_semantic.items():

            previous_cell = (
                previous_by_column.get(
                    column
                )
            )

            if (
                previous_cell is None
                or not (
                    previous_cell
                    .has_semantic_text
                )
            ):
                return False

            previous_geometry = (
                self._cell_bbox(
                    previous_cell,
                    kind=(
                        TableSpanKind
                        .GEOMETRY
                    ),
                )
            )

            current_text = (
                self._cell_bbox(
                    current_cell,
                    kind=(
                        TableSpanKind.TEXT
                    ),
                    require_text=True,
                )
            )

            previous_text = (
                self._cell_bbox(
                    previous_cell,
                    kind=(
                        TableSpanKind.TEXT
                    ),
                    require_text=True,
                )
            )

            if (
                previous_geometry
                is None
                or current_text
                is None
            ):
                return False

            if not self._contains(
                previous_geometry,
                current_text,
            ):
                return False

            if (
                previous_text is not None
                and current_text[1]
                < (
                    previous_text[3]
                    - self
                    ._GEOMETRY_TOLERANCE
                )
            ):
                return False

        return True

    @staticmethod
    def _cell_bbox(
        cell:
            NormalizedTableCellEvidence,
        *,
        kind: TableSpanKind,
        require_text: bool = False,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ] | None:

        spans = []

        for span in cell.source_spans:

            if (
                span.kind != kind
                or span.bbox is None
            ):
                continue

            if (
                require_text
                and (
                    span.text is None
                    or not span.text.strip()
                )
            ):
                continue

            spans.append(
                span
            )

        if not spans:
            return None

        return (
            min(
                span.bbox.x0
                for span in spans
            ),
            min(
                span.bbox.y0
                for span in spans
            ),
            max(
                span.bbox.x1
                for span in spans
            ),
            max(
                span.bbox.y1
                for span in spans
            ),
        )

    def _contains(
        self,
        outer: tuple[
            float,
            float,
            float,
            float,
        ],
        inner: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> bool:

        tolerance = (
            self._GEOMETRY_TOLERANCE
        )

        return (
            inner[0]
            >= outer[0] - tolerance
            and inner[1]
            >= outer[1] - tolerance
            and inner[2]
            <= outer[2] + tolerance
            and inner[3]
            <= outer[3] + tolerance
        )

    def _find_alignments(
        self,
        *,
        collapsed: tuple[
            _CollapsedRow,
            ...
        ],
        structural:
            NormalizedTableEvidence,
    ) -> tuple[
        tuple[
            tuple[int, int],
            ...
        ],
        ...
    ]:

        physical_count = len(
            collapsed
        )

        structural_count = len(
            structural.rows
        )

        if (
            physical_count
            < structural_count
        ):
            return ()

        @lru_cache(
            maxsize=None
        )
        def solve(
            physical_position: int,
            structural_position: int,
        ):
            if (
                structural_position
                == structural_count
            ):
                if (
                    physical_position
                    == physical_count
                ):
                    return (
                        (),
                    )

                return ()

            if (
                physical_position
                >= physical_count
            ):
                return ()

            remaining_physical = (
                physical_count
                - physical_position
            )

            remaining_structural = (
                structural_count
                - structural_position
            )

            if (
                remaining_physical
                < remaining_structural
            ):
                return ()

            structural_row = (
                structural.rows[
                    structural_position
                ]
            )

            solutions = []

            candidate = (
                collapsed[
                    physical_position
                ]
            )

            if self._single_compatible(
                physical=(
                    candidate.row
                ),
                structural=(
                    structural_row
                ),
            ):
                for suffix in solve(
                    physical_position + 1,
                    structural_position + 1,
                ):
                    solutions.append(
                        (
                            (
                                physical_position,
                                physical_position
                                + 1,
                            ),
                            *suffix,
                        )
                    )

                    if len(
                        solutions
                    ) >= 2:
                        return tuple(
                            solutions[:2]
                        )

            extra_rows = (
                remaining_physical
                - remaining_structural
            )

            max_group_size = (
                1
                + extra_rows
            )

            for group_size in range(
                2,
                max_group_size + 1,
            ):
                end = (
                    physical_position
                    + group_size
                )

                if (
                    end
                    > physical_count
                ):
                    break

                selected = (
                    collapsed[
                        physical_position:end
                    ]
                )

                if not (
                    self
                    ._sparse_group_compatible(
                        physical=selected,
                        structural=(
                            structural_row
                        ),
                        column_count=(
                            structural
                            .column_count
                        ),
                    )
                ):
                    continue

                for suffix in solve(
                    end,
                    structural_position + 1,
                ):
                    solutions.append(
                        (
                            (
                                physical_position,
                                end,
                            ),
                            *suffix,
                        )
                    )

                    if len(
                        solutions
                    ) >= 2:
                        return tuple(
                            solutions[:2]
                        )

            return tuple(
                solutions
            )

        return solve(
            0,
            0,
        )

    @staticmethod
    def _structural_capacity(
        row:
            NormalizedTableRowEvidence,
    ) -> int:
        return sum(
            cell.column_end
            - cell.column_start
            for cell in row.cells
        )

    @staticmethod
    def _semantic_mask(
        row:
            NormalizedTableRowEvidence,
    ) -> tuple[
        int,
        ...
    ]:
        return tuple(
            cell.column_start
            for cell in row.cells
            if cell.has_semantic_text
        )

    def _single_compatible(
        self,
        *,
        physical:
            NormalizedTableRowEvidence,
        structural:
            NormalizedTableRowEvidence,
    ) -> bool:

        if (
            structural.source_header_signal
            is True
            and physical.source_header_signal
            is not True
        ):
            return False

        semantic_count = len(
            self._semantic_mask(
                physical
            )
        )

        capacity = (
            self._structural_capacity(
                structural
            )
        )

        return (
            semantic_count
            <= capacity
        )

    def _sparse_group_compatible(
        self,
        *,
        physical: tuple[
            _CollapsedRow,
            ...
        ],
        structural:
            NormalizedTableRowEvidence,
        column_count: int,
    ) -> bool:

        if len(
            physical
        ) < 2:
            return False

        capacity = (
            self._structural_capacity(
                structural
            )
        )

        if (
            capacity
            >= column_count
        ):
            return False

        if (
            structural.source_header_signal
            is True
        ):
            return False

        masks = []

        for item in physical:

            if (
                item.row
                .source_header_signal
                is True
            ):
                return False

            mask = (
                self._semantic_mask(
                    item.row
                )
            )

            if not mask:
                return False

            if len(mask) > capacity:
                return False

            masks.append(
                mask
            )

        first = masks[0]

        return all(
            mask == first
            for mask in masks[1:]
        )

    @staticmethod
    def _common_role(
        values: tuple[
            str | None,
            ...
        ],
    ) -> str | None:

        present = {
            value
            for value in values
            if value is not None
        }

        if len(present) == 1:
            return next(
                iter(present)
            )

        return None

    def _merge_rows(
        self,
        rows: tuple[
            NormalizedTableRowEvidence,
            ...
        ],
        *,
        row_index: int,
        column_count: int,
    ) -> NormalizedTableRowEvidence:

        if not rows:
            raise TableRowAlignmentError(
                "Cannot merge an empty row group."
            )

        merged_cells = []

        for column in range(
            column_count
        ):
            source_cells = []

            for row in rows:

                matches = [
                    cell
                    for cell in row.cells
                    if (
                        cell.column_start
                        == column
                        and cell.column_end
                        == column + 1
                    )
                ]

                if len(matches) != 1:
                    raise (
                        TableRowAlignmentError(
                            "Projected physical rows "
                            "must contain exactly one "
                            "atomic cell per logical "
                            "column."
                        )
                    )

                source_cells.append(
                    matches[0]
                )

            semantic_parts = [
                cell.text.strip()
                for cell in source_cells
                if (
                    cell.has_semantic_text
                    and cell.text is not None
                    and cell.text.strip()
                )
            ]

            if semantic_parts:
                text_status = (
                    TableCellTextStatus
                    .OBSERVED
                )

                text = " ".join(
                    semantic_parts
                )

            elif any(
                cell.observed_blank
                for cell in source_cells
            ):
                text_status = (
                    TableCellTextStatus
                    .OBSERVED
                )

                text = ""

            else:
                text_status = (
                    TableCellTextStatus
                    .UNAVAILABLE
                )

                text = None

            merged_cells.append(
                NormalizedTableCellEvidence(
                    column_start=column,
                    column_end=column + 1,
                    text_status=(
                        text_status
                    ),
                    text=text,
                    source_spans=tuple(
                        span
                        for cell
                        in source_cells
                        for span
                        in cell.source_spans
                    ),
                    source_role=(
                        self._common_role(
                            tuple(
                                cell.source_role
                                for cell
                                in source_cells
                            )
                        )
                    ),
                )
            )

        signals = tuple(
            row.source_header_signal
            for row in rows
        )

        if any(
            signal is True
            for signal in signals
        ):
            header_signal = True

        elif all(
            signal is False
            for signal in signals
        ):
            header_signal = False

        else:
            header_signal = None

        return NormalizedTableRowEvidence(
            row_index=row_index,
            cells=tuple(
                merged_cells
            ),
            source_spans=tuple(
                span
                for row in rows
                for span in row.source_spans
            ),
            source_header_signal=(
                header_signal
            ),
            source_role=(
                self._common_role(
                    tuple(
                        row.source_role
                        for row in rows
                    )
                )
            ),
        )

    @staticmethod
    def _validate_tables(
        *,
        projected_physical:
            NormalizedTableEvidence,
        structural:
            NormalizedTableEvidence,
    ) -> None:

        if (
            projected_physical.layer
            != TableEvidenceLayer.PHYSICAL
        ):
            raise ValueError(
                "projected_physical must "
                "use PHYSICAL layer."
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
            projected_physical.page_number
            != structural.page_number
        ):
            raise ValueError(
                "Physical and structural tables "
                "must belong to the same page."
            )

        if (
            projected_physical.column_count
            != structural.column_count
        ):
            raise ValueError(
                "Physical projection and "
                "structural schema must have "
                "the same column count."
            )
