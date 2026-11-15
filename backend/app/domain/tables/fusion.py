from __future__ import annotations

from dataclasses import dataclass

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


class TableEvidenceFusionError(ValueError):
    pass


@dataclass(frozen=True)
class TableEvidenceMatch:
    physical: NormalizedTableEvidence
    structural: NormalizedTableEvidence
    mapping_kind: TableColumnMappingKind
    physical_column_ranges: tuple[
        tuple[int, int],
        ...
    ]

    def __post_init__(self) -> None:
        if (
            self.physical.layer
            != TableEvidenceLayer.PHYSICAL
        ):
            raise ValueError(
                "physical evidence must use "
                "the PHYSICAL layer."
            )

        if (
            self.structural.layer
            != TableEvidenceLayer.STRUCTURAL
        ):
            raise ValueError(
                "structural evidence must use "
                "the STRUCTURAL layer."
            )

        if (
            self.physical.page_number
            != self.structural.page_number
        ):
            raise ValueError(
                "Matched table evidence must "
                "belong to the same page."
            )

        if (
            len(self.physical_column_ranges)
            != self.structural.column_count
        ):
            raise ValueError(
                "Column mapping must contain "
                "one physical range per "
                "structural column."
            )

        expected_start = 0

        for start, end in (
            self.physical_column_ranges
        ):
            if start != expected_start:
                raise ValueError(
                    "Physical column ranges must "
                    "be contiguous."
                )

            if end <= start:
                raise ValueError(
                    "Physical column range is invalid."
                )

            expected_start = end

        if (
            expected_start
            != self.physical.column_count
        ):
            raise ValueError(
                "Physical column mapping must "
                "cover the entire physical schema."
            )


@dataclass(frozen=True)
class TableEvidenceMatchingResult:
    matches: tuple[
        TableEvidenceMatch,
        ...
    ]
    unmatched_physical_refs: tuple[
        str,
        ...
    ]
    unmatched_structural_refs: tuple[
        str,
        ...
    ]
    ambiguous_physical_refs: tuple[
        str,
        ...
    ] = ()
    ambiguous_structural_refs: tuple[
        str,
        ...
    ] = ()


class TableEvidenceMatcher:
    """Match page-local physical and structural tables.

    Matching is deterministic and fail-closed.

    Supported schema relationships:
    - identical physical/logical column count,
    - equal physical column groups validated by
      a clean header pattern.

    No fuzzy text similarity or LLM matching is used.
    """

    def match_page(
        self,
        *,
        physical_tables: tuple[
            NormalizedTableEvidence,
            ...
        ],
        structural_tables: tuple[
            NormalizedTableEvidence,
            ...
        ],
    ) -> TableEvidenceMatchingResult:

        self._validate_inputs(
            physical_tables=physical_tables,
            structural_tables=structural_tables,
        )

        remaining_physical = set(
            range(
                len(physical_tables)
            )
        )

        remaining_structural = set(
            range(
                len(structural_tables)
            )
        )

        matches: list[
            TableEvidenceMatch
        ] = []

        for mapping_kind in (
            TableColumnMappingKind.EXACT,
            TableColumnMappingKind.EQUAL_GROUPS,
        ):
            while True:
                candidates: dict[
                    tuple[int, int],
                    tuple[
                        tuple[int, int],
                        ...
                    ],
                ] = {}

                for physical_index in sorted(
                    remaining_physical
                ):
                    for structural_index in sorted(
                        remaining_structural
                    ):
                        mapping = (
                            self._mapping_for(
                                physical_tables[
                                    physical_index
                                ],
                                structural_tables[
                                    structural_index
                                ],
                            )
                        )

                        if mapping is None:
                            continue

                        kind, ranges = mapping

                        if (
                            kind
                            != mapping_kind
                        ):
                            continue

                        candidates[
                            (
                                physical_index,
                                structural_index,
                            )
                        ] = ranges

                if not candidates:
                    break

                physical_options: dict[
                    int,
                    list[int],
                ] = {}

                structural_options: dict[
                    int,
                    list[int],
                ] = {}

                for (
                    physical_index,
                    structural_index,
                ) in candidates:
                    physical_options.setdefault(
                        physical_index,
                        [],
                    ).append(
                        structural_index
                    )

                    structural_options.setdefault(
                        structural_index,
                        [],
                    ).append(
                        physical_index
                    )

                resolved = []

                for (
                    physical_index,
                    structural_index,
                ), ranges in candidates.items():

                    if (
                        len(
                            physical_options[
                                physical_index
                            ]
                        )
                        == 1
                        and len(
                            structural_options[
                                structural_index
                            ]
                        )
                        == 1
                    ):
                        resolved.append(
                            (
                                physical_index,
                                structural_index,
                                ranges,
                            )
                        )

                if not resolved:
                    break

                for (
                    physical_index,
                    structural_index,
                    ranges,
                ) in sorted(
                    resolved
                ):
                    matches.append(
                        TableEvidenceMatch(
                            physical=(
                                physical_tables[
                                    physical_index
                                ]
                            ),
                            structural=(
                                structural_tables[
                                    structural_index
                                ]
                            ),
                            mapping_kind=(
                                mapping_kind
                            ),
                            physical_column_ranges=(
                                ranges
                            ),
                        )
                    )

                    remaining_physical.discard(
                        physical_index
                    )

                    remaining_structural.discard(
                        structural_index
                    )

        ambiguous_physical: set[str] = set()
        ambiguous_structural: set[str] = set()

        for physical_index in (
            remaining_physical
        ):
            for structural_index in (
                remaining_structural
            ):
                if (
                    self._mapping_for(
                        physical_tables[
                            physical_index
                        ],
                        structural_tables[
                            structural_index
                        ],
                    )
                    is not None
                ):
                    ambiguous_physical.add(
                        physical_tables[
                            physical_index
                        ].source_ref
                    )

                    ambiguous_structural.add(
                        structural_tables[
                            structural_index
                        ].source_ref
                    )

        return TableEvidenceMatchingResult(
            matches=tuple(
                matches
            ),
            unmatched_physical_refs=tuple(
                physical_tables[index].source_ref
                for index
                in sorted(
                    remaining_physical
                )
            ),
            unmatched_structural_refs=tuple(
                structural_tables[index].source_ref
                for index
                in sorted(
                    remaining_structural
                )
            ),
            ambiguous_physical_refs=tuple(
                sorted(
                    ambiguous_physical
                )
            ),
            ambiguous_structural_refs=tuple(
                sorted(
                    ambiguous_structural
                )
            ),
        )

    def _mapping_for(
        self,
        physical: NormalizedTableEvidence,
        structural: NormalizedTableEvidence,
    ) -> tuple[
        TableColumnMappingKind,
        tuple[
            tuple[int, int],
            ...
        ],
    ] | None:

        if (
            physical.page_number
            != structural.page_number
        ):
            return None

        if (
            physical.column_count
            == structural.column_count
        ):
            return (
                TableColumnMappingKind.EXACT,
                tuple(
                    (
                        column,
                        column + 1,
                    )
                    for column in range(
                        physical.column_count
                    )
                ),
            )

        if (
            physical.column_count
            % structural.column_count
            != 0
        ):
            return None

        group_size = (
            physical.column_count
            // structural.column_count
        )

        if group_size <= 1:
            return None

        ranges = tuple(
            (
                column * group_size,
                (column + 1)
                * group_size,
            )
            for column in range(
                structural.column_count
            )
        )

        if not self._grouped_header_compatible(
            physical=physical,
            structural=structural,
            ranges=ranges,
        ):
            return None

        return (
            TableColumnMappingKind.EQUAL_GROUPS,
            ranges,
        )

    def _grouped_header_compatible(
        self,
        *,
        physical: NormalizedTableEvidence,
        structural: NormalizedTableEvidence,
        ranges: tuple[
            tuple[int, int],
            ...
        ],
    ) -> bool:

        physical_headers = [
            row
            for row in physical.rows
            if (
                row.source_header_signal
                is True
            )
        ]

        structural_headers = [
            row
            for row in structural.rows
            if (
                row.source_header_signal
                is True
            )
        ]

        if (
            len(physical_headers)
            != 1
            or len(structural_headers)
            != 1
        ):
            return False

        structural_header = (
            structural_headers[0]
        )

        if (
            len(structural_header.cells)
            != structural.column_count
        ):
            return False

        physical_header = (
            physical_headers[0]
        )

        for start, end in ranges:
            group_cells = (
                self._cells_in_range(
                    physical_header,
                    start=start,
                    end=end,
                )
            )

            semantic_count = sum(
                1
                for cell in group_cells
                if cell.has_semantic_text
            )

            if semantic_count != 1:
                return False

        return True

    @staticmethod
    def _cells_in_range(
        row: NormalizedTableRowEvidence,
        *,
        start: int,
        end: int,
    ) -> tuple[
        NormalizedTableCellEvidence,
        ...
    ]:

        selected = []

        for cell in row.cells:

            if (
                cell.column_start < end
                and cell.column_end > start
            ):
                if (
                    cell.column_start < start
                    or cell.column_end > end
                ):
                    raise TableEvidenceFusionError(
                        "Physical cell crosses a "
                        "logical column boundary."
                    )

                selected.append(
                    cell
                )

        return tuple(
            selected
        )

    @staticmethod
    def _validate_inputs(
        *,
        physical_tables: tuple[
            NormalizedTableEvidence,
            ...
        ],
        structural_tables: tuple[
            NormalizedTableEvidence,
            ...
        ],
    ) -> None:

        for table in physical_tables:
            if (
                table.layer
                != TableEvidenceLayer.PHYSICAL
            ):
                raise ValueError(
                    "physical_tables contains "
                    "non-PHYSICAL evidence."
                )

        for table in structural_tables:
            if (
                table.layer
                != TableEvidenceLayer.STRUCTURAL
            ):
                raise ValueError(
                    "structural_tables contains "
                    "non-STRUCTURAL evidence."
                )


class TablePhysicalProjector:
    """Project physical rows onto the matched logical schema.

    Row count remains physical.

    This class deliberately does NOT align or merge rows.
    """

    def project(
        self,
        match: TableEvidenceMatch,
    ) -> NormalizedTableEvidence:

        rows = tuple(
            self._project_row(
                row,
                match=match,
            )
            for row in match.physical.rows
        )

        return NormalizedTableEvidence(
            source_ref=(
                f"{match.physical.source_ref}"
                f":projected:"
                f"{match.structural.source_ref}"
            ),
            page_number=(
                match.physical.page_number
            ),
            layer=(
                TableEvidenceLayer.PHYSICAL
            ),
            column_count=(
                match.structural.column_count
            ),
            rows=rows,
            bbox=match.physical.bbox,
            source_spans=(
                match.physical.source_spans
            ),
        )

    def _project_row(
        self,
        row: NormalizedTableRowEvidence,
        *,
        match: TableEvidenceMatch,
    ) -> NormalizedTableRowEvidence:

        cells = []

        for logical_column, (
            start,
            end,
        ) in enumerate(
            match.physical_column_ranges
        ):

            physical_cells = (
                TableEvidenceMatcher
                ._cells_in_range(
                    row,
                    start=start,
                    end=end,
                )
            )

            semantic_texts = [
                cell.text
                for cell in physical_cells
                if (
                    cell.has_semantic_text
                    and cell.text
                    is not None
                )
            ]

            source_spans = tuple(
                span
                for cell in physical_cells
                for span in cell.source_spans
            )

            if semantic_texts:
                text_status = (
                    TableCellTextStatus.OBSERVED
                )

                text = " ".join(
                    semantic_texts
                )

            elif any(
                cell.observed_blank
                for cell in physical_cells
            ):
                text_status = (
                    TableCellTextStatus.OBSERVED
                )

                text = ""

            else:
                text_status = (
                    TableCellTextStatus.UNAVAILABLE
                )

                text = None

            cells.append(
                NormalizedTableCellEvidence(
                    column_start=(
                        logical_column
                    ),
                    column_end=(
                        logical_column + 1
                    ),
                    text_status=(
                        text_status
                    ),
                    text=text,
                    source_spans=(
                        source_spans
                    ),
                    source_role=None,
                )
            )

        return NormalizedTableRowEvidence(
            row_index=(
                row.row_index
            ),
            cells=tuple(
                cells
            ),
            source_spans=(
                row.source_spans
            ),
            source_header_signal=(
                row.source_header_signal
            ),
            source_role=(
                row.source_role
            ),
        )
