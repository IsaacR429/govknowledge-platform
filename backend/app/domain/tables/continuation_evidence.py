from __future__ import annotations

import re

from app.domain.enums.table import (
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    CanonicalTable,
    CanonicalTableCell,
    CanonicalTableRow,
    TableBBox,
    TableContinuationEvidence,
)
from app.domain.tables.hierarchy import (
    TableHierarchyState,
)


class TableContinuationEvidenceBuildError(
    ValueError
):
    """Raised when continuation evidence cannot be derived safely."""


class TableContinuationEvidenceBuilder:
    """Derive parser-neutral continuation evidence from canonical tables.

    Strong signals are derived from normalized/canonical structure.

    The builder never guesses when required physical or logical
    evidence is unavailable. Missing geometry, row keys, or hierarchy
    state therefore produces conservative False values which cause the
    continuity resolver to STOP or become AMBIGUOUS rather than
    silently continuing.
    """

    _ROW_KEY_PATTERN = re.compile(
        r"^(.*?)(\d+)$"
    )

    def __init__(
        self,
        *,
        max_geometry_delta_ratio: float = 0.03,
    ) -> None:

        if not (
            0.0
            <= max_geometry_delta_ratio
            <= 1.0
        ):
            raise ValueError(
                "max_geometry_delta_ratio "
                "must be between 0 and 1."
            )

        self._max_geometry_delta_ratio = (
            max_geometry_delta_ratio
        )

    def build(
        self,
        *,
        left: CanonicalTable,
        right: CanonicalTable,
        left_state: (
            TableHierarchyState | None
        ) = None,
        explicit_contradictory_boundary: bool = False,
    ) -> TableContinuationEvidence:

        left_page = self._single_page(
            left,
            label="left",
        )

        right_page = self._single_page(
            right,
            label="right",
        )

        left_source_ref = (
            self._primary_source_ref(
                left,
                label="left",
            )
        )

        right_source_ref = (
            self._primary_source_ref(
                right,
                label="right",
            )
        )

        same_column_count = (
            left.column_count
            == right.column_count
        )

        compatible_column_geometry = (
            same_column_count
            and self._compatible_column_geometry(
                left,
                right,
            )
        )

        shared_structural_identity = bool(
            {
                value.strip()
                for value
                in left.structural_identity_refs
                if value.strip()
            }
            & {
                value.strip()
                for value
                in right.structural_identity_refs
                if value.strip()
            }
        )

        repeated_header = (
            bool(left.column_names)
            and bool(right.column_names)
            and (
                left.column_names
                == right.column_names
            )
        )

        header_conflict = (
            bool(left.column_names)
            and bool(right.column_names)
            and (
                left.column_names
                != right.column_names
            )
        )

        hierarchy_conflict = (
            (
                bool(left.hierarchy_columns)
                or bool(
                    right.hierarchy_columns
                )
            )
            and (
                left.hierarchy_columns
                != right.hierarchy_columns
            )
        )

        contradictory_boundary = (
            explicit_contradictory_boundary
            or header_conflict
            or hierarchy_conflict
        )

        headerless_continuation = (
            not repeated_header
            and not left.column_names
            and not right.column_names
            and same_column_count
            and compatible_column_geometry
        )

        row_key_progression = (
            headerless_continuation
            and self._row_key_progression(
                left,
                right,
            )
        )

        left_has_active_hierarchy = (
            self._has_complete_active_hierarchy(
                table=left,
                state=left_state,
            )
        )

        right_starts_with_inherited_cells = (
            left_has_active_hierarchy
            and self._right_starts_with_inheritable_cells(
                right=right,
                left_state=left_state,
            )
        )

        return TableContinuationEvidence(
            left_source_ref=left_source_ref,
            right_source_ref=right_source_ref,
            left_page=left_page,
            right_page=right_page,
            same_column_count=(
                same_column_count
            ),
            compatible_column_geometry=(
                compatible_column_geometry
            ),
            repeated_header=(
                repeated_header
            ),
            headerless_continuation=(
                headerless_continuation
            ),
            row_key_progression=(
                row_key_progression
            ),
            shared_structural_identity=(
                shared_structural_identity
            ),
            left_has_active_hierarchy=(
                left_has_active_hierarchy
            ),
            right_starts_with_inherited_cells=(
                right_starts_with_inherited_cells
            ),
            contradictory_boundary=(
                contradictory_boundary
            ),
        )

    @staticmethod
    def _single_page(
        table: CanonicalTable,
        *,
        label: str,
    ) -> int:

        # A canonical table is page-local at this stage.
        #
        # Derive its physical page from ROW provenance,
        # not cell origin provenance. This matters because
        # inherited cells may legitimately carry value
        # origin_spans from a previous page.
        pages = tuple(
            sorted(
                {
                    span.page_number
                    for row in table.rows
                    for span in row.source_spans
                }
            )
        )

        if len(pages) != 1:
            raise (
                TableContinuationEvidenceBuildError(
                    (
                        f"{label} canonical table "
                        "must represent exactly one "
                        "physical page from row "
                        "provenance; found "
                        f"{pages!r}."
                    )
                )
            )

        return pages[0]

    @staticmethod
    def _primary_source_ref(
        table: CanonicalTable,
        *,
        label: str,
    ) -> str:

        refs = tuple(
            table.source_table_refs
        )

        if not refs:
            raise (
                TableContinuationEvidenceBuildError(
                    (
                        f"{label} canonical table "
                        "has no source_table_refs."
                    )
                )
            )

        value = refs[0]

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            raise (
                TableContinuationEvidenceBuildError(
                    (
                        f"{label} canonical table "
                        "has an invalid primary "
                        "source reference."
                    )
                )
            )

        return value

    def _compatible_column_geometry(
        self,
        left: CanonicalTable,
        right: CanonicalTable,
    ) -> bool:

        left_signature = (
            self._geometry_signature(
                left
            )
        )

        right_signature = (
            self._geometry_signature(
                right
            )
        )

        if (
            left_signature is None
            or right_signature is None
        ):
            return False

        if (
            len(left_signature)
            != len(right_signature)
        ):
            return False

        tolerance = (
            self._max_geometry_delta_ratio
        )

        for (
            left_bounds,
            right_bounds,
        ) in zip(
            left_signature,
            right_signature,
            strict=True,
        ):

            if (
                abs(
                    left_bounds[0]
                    - right_bounds[0]
                )
                > tolerance
            ):
                return False

            if (
                abs(
                    left_bounds[1]
                    - right_bounds[1]
                )
                > tolerance
            ):
                return False

        return True

    def _geometry_signature(
        self,
        table: CanonicalTable,
    ) -> tuple[
        tuple[float, float],
        ...,
    ] | None:

        candidate_rows = tuple(
            row
            for row in table.rows
            if (
                row.role
                == TableRowRole.HEADER
            )
        ) + tuple(
            row
            for row in table.rows
            if (
                row.role
                == TableRowRole.DATA
            )
        )

        for row in candidate_rows:

            boxes: list[
                TableBBox
            ] = []

            valid = True

            for column in range(
                table.column_count
            ):

                cell = (
                    self._find_atomic_cell(
                        row=row,
                        column=column,
                    )
                )

                if cell is None:
                    valid = False
                    break

                bbox = (
                    self._cell_geometry_bbox(
                        cell
                    )
                )

                if bbox is None:
                    valid = False
                    break

                boxes.append(
                    bbox
                )

            if not valid:
                continue

            table_x0 = min(
                bbox.x0
                for bbox in boxes
            )

            table_x1 = max(
                bbox.x1
                for bbox in boxes
            )

            width = (
                table_x1
                - table_x0
            )

            if width <= 0:
                continue

            return tuple(
                (
                    (
                        bbox.x0
                        - table_x0
                    )
                    / width,
                    (
                        bbox.x1
                        - table_x0
                    )
                    / width,
                )
                for bbox in boxes
            )

        return None

    @staticmethod
    def _cell_geometry_bbox(
        cell: CanonicalTableCell,
    ) -> TableBBox | None:

        boxes = tuple(
            span.bbox
            for span
            in cell.structure_spans
            if (
                span.kind
                == TableSpanKind.GEOMETRY
                and span.bbox
                is not None
            )
        )

        if not boxes:
            return None

        return TableBBox(
            x0=min(
                bbox.x0
                for bbox in boxes
            ),
            y0=min(
                bbox.y0
                for bbox in boxes
            ),
            x1=max(
                bbox.x1
                for bbox in boxes
            ),
            y1=max(
                bbox.y1
                for bbox in boxes
            ),
        )

    @staticmethod
    def _find_atomic_cell(
        *,
        row: CanonicalTableRow,
        column: int,
    ) -> CanonicalTableCell | None:

        matches = tuple(
            cell
            for cell in row.cells
            if (
                cell.column_start
                == column
                and cell.column_end
                == column + 1
            )
        )

        if len(matches) != 1:
            return None

        return matches[0]

    @classmethod
    def _row_key_progression(
        cls,
        left: CanonicalTable,
        right: CanonicalTable,
    ) -> bool:

        # Prefer already-resolved logical keys when available.
        if cls._logical_key_progression(
            left,
            right,
        ):
            return True

        # Before TableStructureResolver runs, procurement
        # rows do not yet have LOT_1/03-style logical keys.
        #
        # In that case use only the physical boundary:
        #
        #     left last semantic DATA row  = 03
        #     right first semantic DATA row = 04
        #
        # This is intentionally strict and is only one input
        # to the wider continuation decision, which also
        # requires adjacency, equal schema and geometry.
        return (
            cls._boundary_ordinal_progression(
                left,
                right,
            )
        )

    @classmethod
    def _logical_key_progression(
        cls,
        left: CanonicalTable,
        right: CanonicalTable,
    ) -> bool:

        left_keys = tuple(
            row.logical_key
            for row in left.rows
            if (
                row.role
                == TableRowRole.DATA
                and row.logical_key
            )
        )

        right_keys = tuple(
            row.logical_key
            for row in right.rows
            if (
                row.role
                == TableRowRole.DATA
                and row.logical_key
            )
        )

        if (
            not left_keys
            or not right_keys
        ):
            return False

        left_match = (
            cls._ROW_KEY_PATTERN.fullmatch(
                left_keys[-1]
            )
        )

        right_match = (
            cls._ROW_KEY_PATTERN.fullmatch(
                right_keys[0]
            )
        )

        if (
            left_match is None
            or right_match is None
        ):
            return False

        left_prefix = (
            left_match.group(1)
        )

        right_prefix = (
            right_match.group(1)
        )

        if (
            left_prefix
            != right_prefix
        ):
            return False

        return (
            int(
                right_match.group(2)
            )
            == (
                int(
                    left_match.group(2)
                )
                + 1
            )
        )

    @classmethod
    def _boundary_ordinal_progression(
        cls,
        left: CanonicalTable,
        right: CanonicalTable,
    ) -> bool:

        if (
            left.column_count < 2
            or right.column_count < 2
        ):
            return False

        left_row = next(
            (
                row
                for row in reversed(
                    left.rows
                )
                if (
                    row.role
                    == TableRowRole.DATA
                    and cls._has_semantic_text(
                        row
                    )
                )
            ),
            None,
        )

        right_row = next(
            (
                row
                for row in right.rows
                if (
                    row.role
                    == TableRowRole.DATA
                    and cls._has_semantic_text(
                        row
                    )
                )
            ),
            None,
        )

        if (
            left_row is None
            or right_row is None
        ):
            return False

        left_ordinal = (
            cls._boundary_ordinal(
                left_row
            )
        )

        right_ordinal = (
            cls._boundary_ordinal(
                right_row
            )
        )

        if (
            left_ordinal is None
            or right_ordinal is None
        ):
            return False

        return (
            right_ordinal
            == left_ordinal + 1
        )

    @staticmethod
    def _has_semantic_text(
        row: CanonicalTableRow,
    ) -> bool:

        return any(
            cell.text.strip()
            for cell in row.cells
        )

    @classmethod
    def _boundary_ordinal(
        cls,
        row: CanonicalTableRow,
    ) -> int | None:

        first_cells = tuple(
            cell
            for cell in row.cells
            if (
                cell.column_start == 0
                and cell.column_end == 1
            )
        )

        if len(first_cells) != 1:
            return None

        # Require actual row content outside the ordinal
        # column. A bare page number or one-column numeric
        # structure must not establish continuation.
        if not any(
            cell.text.strip()
            for cell in row.cells
            if not (
                cell.column_start == 0
                and cell.column_end == 1
            )
        ):
            return None

        value = (
            first_cells[0]
            .text
            .strip()
        )

        match = re.fullmatch(
            r"0*(\d{1,3})[.)-]?",
            value,
        )

        if match is None:
            return None

        ordinal = int(
            match.group(1)
        )

        if ordinal < 1:
            return None

        return ordinal

    @staticmethod
    def _has_complete_active_hierarchy(
        *,
        table: CanonicalTable,
        state: TableHierarchyState | None,
    ) -> bool:

        hierarchy_columns = (
            table.hierarchy_columns
        )

        if (
            not hierarchy_columns
            or state is None
        ):
            return False

        return all(
            state.get(
                column
            )
            is not None
            for column
            in hierarchy_columns
        )

    @classmethod
    def _right_starts_with_inheritable_cells(
        cls,
        *,
        right: CanonicalTable,
        left_state: (
            TableHierarchyState | None
        ),
    ) -> bool:

        if left_state is None:
            return False

        hierarchy_columns = (
            right.hierarchy_columns
        )

        if not hierarchy_columns:
            return False

        first_data_row = next(
            (
                row
                for row in right.rows
                if (
                    row.role
                    == TableRowRole.DATA
                )
            ),
            None,
        )

        if first_data_row is None:
            return False

        for column in hierarchy_columns:

            if (
                left_state.get(
                    column
                )
                is None
            ):
                return False

            cell = (
                cls._find_atomic_cell(
                    row=first_data_row,
                    column=column,
                )
            )

            if cell is None:
                return False

            if cell.text.strip():
                return False

        return True
