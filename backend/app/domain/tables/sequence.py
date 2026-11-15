from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.domain.enums.table import (
    TableContinuationDecision,
)
from app.domain.models.table import (
    CanonicalTable,
    TableContinuationEvidence,
    TableContinuationResult,
)
from app.domain.tables.continuity import (
    TableContinuityResolver,
)
from app.domain.tables.continuation_evidence import (
    TableContinuationEvidenceBuilder,
)
from app.domain.tables.hierarchy import (
    TableHierarchyResolutionResult,
    TableHierarchyResolver,
    TableHierarchyState,
)


class TableSequenceResolutionError(ValueError):
    """Raised when a table sequence cannot be resolved safely."""


@dataclass(frozen=True)
class TableSequenceStep:
    """One resolved logical table within a physical table sequence."""

    table_index: int
    hierarchy: TableHierarchyResolutionResult
    incoming_continuity: (
        TableContinuationResult | None
    ) = None

    @property
    def table(
        self,
    ) -> CanonicalTable:
        return self.hierarchy.table

    @property
    def final_state(
        self,
    ) -> TableHierarchyState:
        return self.hierarchy.final_state

    @property
    def inherited_cell_count(
        self,
    ) -> int:
        return (
            self.hierarchy.inherited_cell_count
        )


@dataclass(frozen=True)
class TableSequenceResolutionResult:
    """Resolved tables plus the decisions used at each boundary."""

    steps: tuple[
        TableSequenceStep,
        ...,
    ]

    continuation_results: tuple[
        TableContinuationResult,
        ...,
    ]

    @property
    def tables(
        self,
    ) -> tuple[
        CanonicalTable,
        ...,
    ]:
        return tuple(
            step.table
            for step in self.steps
        )

    @property
    def inherited_cell_count(
        self,
    ) -> int:
        return sum(
            step.inherited_cell_count
            for step in self.steps
        )

    @property
    def final_state(
        self,
    ) -> TableHierarchyState | None:

        if not self.steps:
            return None

        return (
            self.steps[-1]
            .final_state
        )


class TableSequenceResolver:
    """Resolve hierarchy across physical table boundaries.

    State transfer is permitted only when the continuity resolver
    returns CONTINUE.

    STOP resets hierarchy state.

    AMBIGUOUS fails closed instead of guessing whether hierarchy
    state may cross the boundary.
    """

    def __init__(
        self,
        *,
        continuity_resolver: (
            TableContinuityResolver | None
        ) = None,
        hierarchy_resolver: (
            TableHierarchyResolver | None
        ) = None,
        continuation_evidence_builder: (
            TableContinuationEvidenceBuilder | None
        ) = None,
    ) -> None:

        self._continuity = (
            continuity_resolver
            or TableContinuityResolver()
        )

        self._hierarchy = (
            hierarchy_resolver
            or TableHierarchyResolver()
        )

        self._evidence_builder = (
            continuation_evidence_builder
            or TableContinuationEvidenceBuilder()
        )

    def resolve(
        self,
        tables: Sequence[
            CanonicalTable
        ],
        *,
        boundaries: Sequence[
            TableContinuationEvidence
        ] | None = None,
    ) -> TableSequenceResolutionResult:

        table_items = tuple(
            tables
        )

        boundary_items = (
            tuple(boundaries)
            if boundaries is not None
            else None
        )

        expected_boundaries = max(
            len(table_items) - 1,
            0,
        )

        if (
            boundary_items is not None
            and len(boundary_items)
            != expected_boundaries
        ):
            raise TableSequenceResolutionError(
                (
                    "Table sequence boundary count "
                    "does not match table count: "
                    f"expected {expected_boundaries}, "
                    f"received "
                    f"{len(boundary_items)}."
                )
            )

        if not table_items:
            return (
                TableSequenceResolutionResult(
                    steps=(),
                    continuation_results=(),
                )
            )

        first_hierarchy = (
            self._hierarchy.resolve_table(
                table_items[0]
            )
        )

        steps: list[
            TableSequenceStep
        ] = [
            TableSequenceStep(
                table_index=0,
                hierarchy=first_hierarchy,
                incoming_continuity=None,
            )
        ]

        continuation_results: list[
            TableContinuationResult
        ] = []

        for table_index in range(
            1,
            len(table_items),
        ):

            if boundary_items is not None:
                evidence = (
                    boundary_items[
                        table_index - 1
                    ]
                )
            else:
                evidence = (
                    self._evidence_builder.build(
                        left=table_items[
                            table_index - 1
                        ],
                        right=table_items[
                            table_index
                        ],
                        left_state=(
                            steps[-1]
                            .final_state
                        ),
                    )
                )

            continuity = (
                self._continuity.resolve(
                    evidence
                )
            )

            continuation_results.append(
                continuity
            )

            if (
                continuity.decision
                == TableContinuationDecision.AMBIGUOUS
            ):
                detail = "; ".join(
                    continuity.reasons
                )

                raise (
                    TableSequenceResolutionError(
                        (
                            "Ambiguous table boundary "
                            f"before table "
                            f"{table_index}. "
                            "Hierarchy state cannot "
                            "cross this boundary."
                            + (
                                f" {detail}"
                                if detail
                                else ""
                            )
                        )
                    )
                )

            if (
                continuity.decision
                == TableContinuationDecision.CONTINUE
            ):
                initial_state = (
                    steps[-1]
                    .final_state
                )

            elif (
                continuity.decision
                == TableContinuationDecision.STOP
            ):
                initial_state = None

            else:
                raise (
                    TableSequenceResolutionError(
                        (
                            "Unsupported table "
                            "continuation decision "
                            f"before table "
                            f"{table_index}: "
                            f"{continuity.decision!r}."
                        )
                    )
                )

            hierarchy = (
                self._hierarchy.resolve_table(
                    table_items[
                        table_index
                    ],
                    initial_state=(
                        initial_state
                    ),
                )
            )

            steps.append(
                TableSequenceStep(
                    table_index=(
                        table_index
                    ),
                    hierarchy=hierarchy,
                    incoming_continuity=(
                        continuity
                    ),
                )
            )

        return (
            TableSequenceResolutionResult(
                steps=tuple(
                    steps
                ),
                continuation_results=tuple(
                    continuation_results
                ),
            )
        )
