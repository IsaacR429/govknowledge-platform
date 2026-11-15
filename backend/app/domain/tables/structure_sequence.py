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
from app.domain.tables.structure import (
    TableStructureResolutionResult,
    TableStructureResolver,
    TableStructureState,
)


class TableStructureSequenceResolutionError(
    ValueError
):
    """Raised when structure state cannot cross a boundary safely."""


@dataclass(frozen=True)
class TableStructureSequenceStep:
    table_index: int
    structure: TableStructureResolutionResult
    incoming_continuity: (
        TableContinuationResult | None
    ) = None

    @property
    def table(
        self,
    ) -> CanonicalTable:
        return self.structure.table

    @property
    def final_state(
        self,
    ) -> TableStructureState:
        return self.structure.final_state


@dataclass(frozen=True)
class TableStructureSequenceResolutionResult:
    steps: tuple[
        TableStructureSequenceStep,
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
    def final_state(
        self,
    ) -> TableStructureState | None:

        if not self.steps:
            return None

        return (
            self.steps[-1]
            .final_state
        )


class TableStructureSequenceResolver:
    """Resolve logical table structure across approved boundaries.

    CONTINUE -> carry structure state.
    STOP     -> reset structure state.
    AMBIGUOUS -> fail closed.
    """

    def __init__(
        self,
        *,
        continuity_resolver: (
            TableContinuityResolver | None
        ) = None,
        structure_resolver: (
            TableStructureResolver | None
        ) = None,
        continuation_evidence_builder: (
            TableContinuationEvidenceBuilder | None
        ) = None,
    ) -> None:

        self._continuity = (
            continuity_resolver
            or TableContinuityResolver()
        )

        self._structure = (
            structure_resolver
            or TableStructureResolver()
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
    ) -> TableStructureSequenceResolutionResult:

        table_items = tuple(
            tables
        )

        boundary_items = (
            tuple(
                boundaries
            )
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
            raise (
                TableStructureSequenceResolutionError(
                    (
                        "Table sequence boundary count "
                        "does not match table count: "
                        f"expected {expected_boundaries}, "
                        f"received "
                        f"{len(boundary_items)}."
                    )
                )
            )

        if not table_items:
            return (
                TableStructureSequenceResolutionResult(
                    steps=(),
                    continuation_results=(),
                )
            )

        first_structure = (
            self._structure.resolve_table(
                table_items[0]
            )
        )

        steps: list[
            TableStructureSequenceStep
        ] = [
            TableStructureSequenceStep(
                table_index=0,
                structure=first_structure,
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
                    TableStructureSequenceResolutionError(
                        (
                            "Ambiguous table boundary "
                            f"before table "
                            f"{table_index}. "
                            "Structure state cannot "
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
                    TableStructureSequenceResolutionError(
                        (
                            "Unsupported continuation "
                            f"decision before table "
                            f"{table_index}: "
                            f"{continuity.decision!r}."
                        )
                    )
                )

            structure = (
                self._structure.resolve_table(
                    table_items[
                        table_index
                    ],
                    initial_state=(
                        initial_state
                    ),
                )
            )

            steps.append(
                TableStructureSequenceStep(
                    table_index=(
                        table_index
                    ),
                    structure=structure,
                    incoming_continuity=(
                        continuity
                    ),
                )
            )

        return (
            TableStructureSequenceResolutionResult(
                steps=tuple(
                    steps
                ),
                continuation_results=tuple(
                    continuation_results
                ),
            )
        )
