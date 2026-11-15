from __future__ import annotations

from dataclasses import dataclass, replace

from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
)
from app.domain.models.table import (
    CanonicalTable,
    CanonicalTableCell,
    CanonicalTableRow,
)


class TableHierarchyResolutionError(ValueError):
    """Canonical hierarchy could not be resolved deterministically."""


@dataclass(frozen=True)
class TableHierarchyState:
    """Active hierarchy parents available for inheritance.

    The state is immutable so it can safely cross physical
    table/page boundaries only after continuity has been approved.
    """

    active_cells: tuple[
        tuple[int, CanonicalTableCell],
        ...
    ] = ()

    def get(
        self,
        column: int,
    ) -> CanonicalTableCell | None:
        for active_column, cell in self.active_cells:
            if active_column == column:
                return cell

        return None

    def as_dict(
        self,
    ) -> dict[int, CanonicalTableCell]:
        return dict(self.active_cells)

    @classmethod
    def from_dict(
        cls,
        active: dict[int, CanonicalTableCell],
    ) -> "TableHierarchyState":
        return cls(
            active_cells=tuple(
                sorted(
                    active.items(),
                    key=lambda item: item[0],
                )
            )
        )


@dataclass(frozen=True)
class TableHierarchyResolutionResult:
    table: CanonicalTable
    final_state: TableHierarchyState
    inherited_cell_count: int


class TableHierarchyResolver:
    """Resolve hierarchical blank cells using active parent state.

    Rules:

    1. Only DATA rows participate.
    2. A nonblank hierarchy cell becomes the active value.
    3. Changing a parent clears all deeper active hierarchy values.
    4. A blank hierarchy cell may inherit only from an active value.
    5. Inherited cells preserve the original value provenance.
    6. Current-row structural provenance remains attached to the
       inherited canonical cell.
    7. Missing active state fails closed instead of inventing a value.
    """

    def resolve_table(
        self,
        table: CanonicalTable,
        *,
        initial_state: TableHierarchyState | None = None,
    ) -> TableHierarchyResolutionResult:

        hierarchy_columns = table.hierarchy_columns

        self._validate_hierarchy_columns(
            hierarchy_columns
        )

        active = (
            initial_state.as_dict()
            if initial_state is not None
            else {}
        )

        self._validate_initial_state(
            active=active,
            hierarchy_columns=hierarchy_columns,
        )

        resolved_rows: list[
            CanonicalTableRow
        ] = []

        inherited_count = 0

        for row in table.rows:

            if row.role != TableRowRole.DATA:
                resolved_rows.append(row)
                continue

            (
                resolved_row,
                inherited_in_row,
            ) = self._resolve_data_row(
                row=row,
                hierarchy_columns=hierarchy_columns,
                active=active,
            )

            inherited_count += (
                inherited_in_row
            )

            resolved_rows.append(
                resolved_row
            )

        resolved_table = CanonicalTable(
            column_count=table.column_count,
            rows=tuple(resolved_rows),
            column_names=table.column_names,
            hierarchy_columns=(
                table.hierarchy_columns
            ),
            source_table_refs=(
                table.source_table_refs
            ),
            structural_identity_refs=(
                table.structural_identity_refs
            ),
        )

        return TableHierarchyResolutionResult(
            table=resolved_table,
            final_state=(
                TableHierarchyState.from_dict(
                    active
                )
            ),
            inherited_cell_count=(
                inherited_count
            ),
        )

    def _resolve_data_row(
        self,
        *,
        row: CanonicalTableRow,
        hierarchy_columns: tuple[int, ...],
        active: dict[int, CanonicalTableCell],
    ) -> tuple[
        CanonicalTableRow,
        int,
    ]:

        cells = list(row.cells)

        inherited_count = 0

        for hierarchy_position, column in enumerate(
            hierarchy_columns
        ):

            cell_index = self._find_atomic_cell_index(
                row=row,
                column=column,
            )

            cell = cells[
                cell_index
            ]

            if cell.text.strip():

                # A new explicit value at this hierarchy
                # level invalidates all deeper active parents.
                for deeper_column in (
                    hierarchy_columns[
                        hierarchy_position + 1:
                    ]
                ):
                    active.pop(
                        deeper_column,
                        None,
                    )

                active[column] = cell
                continue

            active_parent = active.get(
                column
            )

            if active_parent is None:
                raise TableHierarchyResolutionError(
                    (
                        "Cannot inherit hierarchy value "
                        f"for row {row.row_index}, "
                        f"column {column}: "
                        "no active parent exists."
                    )
                )

            inherited_cell = (
                CanonicalTableCell(
                    column_start=(
                        cell.column_start
                    ),
                    column_end=(
                        cell.column_end
                    ),
                    text=active_parent.text,
                    value_source=(
                        TableCellValueSource.INHERITED
                    ),

                    # Value provenance remains the
                    # source of the active parent.
                    origin_spans=(
                        active_parent.origin_spans
                    ),

                    # Structural provenance remains
                    # attached to the blank/current cell.
                    structure_spans=(
                        cell.structure_spans
                    ),
                )
            )

            cells[
                cell_index
            ] = inherited_cell

            inherited_count += 1

        return (
            replace(
                row,
                cells=tuple(cells),
            ),
            inherited_count,
        )

    @staticmethod
    def _find_atomic_cell_index(
        *,
        row: CanonicalTableRow,
        column: int,
    ) -> int:

        matches = [
            index
            for index, cell
            in enumerate(row.cells)
            if (
                cell.column_start == column
                and cell.column_end
                == column + 1
            )
        ]

        if len(matches) != 1:
            raise TableHierarchyResolutionError(
                (
                    "Hierarchy resolution requires "
                    "exactly one atomic logical cell "
                    f"for row {row.row_index}, "
                    f"column {column}; "
                    f"found {len(matches)}."
                )
            )

        return matches[0]

    @staticmethod
    def _validate_hierarchy_columns(
        hierarchy_columns: tuple[int, ...],
    ) -> None:

        if not hierarchy_columns:
            return

        if tuple(
            sorted(hierarchy_columns)
        ) != hierarchy_columns:
            raise TableHierarchyResolutionError(
                "hierarchy_columns must be ordered "
                "from outermost to innermost."
            )

        if (
            len(set(hierarchy_columns))
            != len(hierarchy_columns)
        ):
            raise TableHierarchyResolutionError(
                "hierarchy_columns cannot contain duplicates."
            )

    @staticmethod
    def _validate_initial_state(
        *,
        active: dict[int, CanonicalTableCell],
        hierarchy_columns: tuple[int, ...],
    ) -> None:

        allowed = set(
            hierarchy_columns
        )

        invalid = (
            set(active)
            - allowed
        )

        if invalid:
            raise TableHierarchyResolutionError(
                (
                    "Initial hierarchy state contains "
                    "columns outside the table hierarchy: "
                    f"{sorted(invalid)}."
                )
            )
