from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.domain.enums.table import (
    TableContinuationDecision,
    TableEvidenceLayer,
)
from app.domain.models.table import (
    CanonicalTable,
)
from app.domain.tables.alignment import (
    TableRowAligner,
)
from app.domain.tables.canonicalizer import (
    TableCanonicalizer,
)
from app.domain.tables.continuation_evidence import (
    TableContinuationEvidenceBuildError,
    TableContinuationEvidenceBuilder,
)
from app.domain.tables.continuity import (
    TableContinuityResolver,
)
from app.domain.tables.evidence import (
    NormalizedTableEvidence,
)
from app.domain.tables.fusion import (
    TableEvidenceMatcher,
    TablePhysicalProjector,
)
from app.domain.tables.structure_sequence import (
    TableStructureSequenceResolver,
)


class TableEvidenceCompilationError(
    ValueError
):
    """Raised when table evidence compilation fails."""


@dataclass(frozen=True)
class TableEvidenceCompilationResult:
    tables: tuple[
        CanonicalTable,
        ...,
    ]
    matched_table_count: int
    sequences: tuple[
        tuple[
            CanonicalTable,
            ...,
        ],
        ...,
    ] = ()
    unmatched_physical_refs: tuple[
        str,
        ...,
    ] = ()
    unmatched_structural_refs: tuple[
        str,
        ...,
    ] = ()
    ambiguous_physical_refs: tuple[
        str,
        ...,
    ] = ()
    ambiguous_structural_refs: tuple[
        str,
        ...,
    ] = ()


class TableEvidenceCompilationService:
    """Compile raw physical and structural table evidence into canonical tables.

    Matches tables page-by-page, projects physical columns onto structural schema,
    aligns rows, canonicalizes values and provenance, and resolves structure
    for each table independently.
    """

    def __init__(
        self,
        *,
        matcher: (
            TableEvidenceMatcher | None
        ) = None,
        projector: (
            TablePhysicalProjector | None
        ) = None,
        aligner: (
            TableRowAligner | None
        ) = None,
        canonicalizer: (
            TableCanonicalizer | None
        ) = None,
        structure_resolver: (
            TableStructureSequenceResolver | None
        ) = None,
        continuation_evidence_builder: (
            TableContinuationEvidenceBuilder | None
        ) = None,
        continuity_resolver: (
            TableContinuityResolver | None
        ) = None,
    ) -> None:
        self._matcher = (
            matcher
            or TableEvidenceMatcher()
        )
        self._projector = (
            projector
            or TablePhysicalProjector()
        )
        self._aligner = (
            aligner
            or TableRowAligner()
        )
        self._canonicalizer = (
            canonicalizer
            or TableCanonicalizer()
        )
        self._structure = (
            structure_resolver
            or TableStructureSequenceResolver()
        )
        self._continuation_evidence = (
            continuation_evidence_builder
            or TableContinuationEvidenceBuilder()
        )
        self._continuity = (
            continuity_resolver
            or TableContinuityResolver()
        )

    def _partition_continuation_sequences(
        self,
        tables: Sequence[
            CanonicalTable
        ],
    ) -> tuple[
        tuple[
            CanonicalTable,
            ...,
        ],
        ...,
    ]:
        items = tuple(tables)

        if not items:
            return ()

        sequences: list[
            tuple[
                CanonicalTable,
                ...,
            ]
        ] = []
        current: list[
            CanonicalTable
        ] = [items[0]]

        for right in items[1:]:
            left = current[-1]
            try:
                evidence = (
                    self._continuation_evidence
                    .build(
                        left=left,
                        right=right,
                    )
                )
                continuity = (
                    self._continuity.resolve(
                        evidence
                    )
                )
                decision = (
                    continuity.decision
                )
            except (
                TableContinuationEvidenceBuildError
            ):
                # Missing/unsafe continuation evidence
                # must never permit state transfer.
                decision = (
                    TableContinuationDecision.AMBIGUOUS
                )

            if (
                decision
                == TableContinuationDecision.CONTINUE
            ):
                current.append(
                    right
                )
                continue

            sequences.append(
                tuple(current)
            )
            current = [
                right
            ]

        sequences.append(
            tuple(current)
        )

        return tuple(
            sequences
        )

    def compile(
        self,
        physical_tables: Sequence[
            NormalizedTableEvidence
        ] = (),
        structural_tables: Sequence[
            NormalizedTableEvidence
        ] = (),
    ) -> TableEvidenceCompilationResult:
        physical_items = (
            tuple(physical_tables)
            if physical_tables is not None
            else ()
        )
        structural_items = (
            tuple(structural_tables)
            if structural_tables is not None
            else ()
        )

        for table in physical_items:
            if (
                table.layer
                != TableEvidenceLayer.PHYSICAL
            ):
                raise TableEvidenceCompilationError(
                    "physical_tables contains non-PHYSICAL evidence."
                )

        for table in structural_items:
            if (
                table.layer
                != TableEvidenceLayer.STRUCTURAL
            ):
                raise TableEvidenceCompilationError(
                    "structural_tables contains non-STRUCTURAL evidence."
                )

        pages = sorted(
            {
                table.page_number
                for table in physical_items
            }
            | {
                table.page_number
                for table in structural_items
            }
        )

        canonical_tables: list[
            CanonicalTable
        ] = []
        matched_count = 0
        unmatched_physical: list[str] = []
        unmatched_structural: list[str] = []
        ambiguous_physical: list[str] = []
        ambiguous_structural: list[str] = []

        for page in pages:
            page_physical = tuple(
                table
                for table in physical_items
                if table.page_number == page
            )
            page_structural = tuple(
                table
                for table in structural_items
                if table.page_number == page
            )

            matching = (
                self._matcher.match_page(
                    physical_tables=(
                        page_physical
                    ),
                    structural_tables=(
                        page_structural
                    ),
                )
            )

            unmatched_physical.extend(
                matching.unmatched_physical_refs
            )
            unmatched_structural.extend(
                matching.unmatched_structural_refs
            )
            ambiguous_physical.extend(
                matching.ambiguous_physical_refs
            )
            ambiguous_structural.extend(
                matching.ambiguous_structural_refs
            )

            for match in matching.matches:
                projected = (
                    self._projector.project(
                        match
                    )
                )
                aligned = (
                    self._aligner.align(
                        projected_physical=(
                            projected
                        ),
                        structural=(
                            match.structural
                        ),
                    )
                )
                canonical = (
                    self._canonicalizer.canonicalize(
                        aligned_physical=(
                            aligned.aligned_physical
                        ),
                        structural=(
                            match.structural
                        ),
                    )
                    .table
                )
                canonical_tables.append(
                    canonical
                )
                matched_count += 1

        # Cross-page production wiring:
        #
        # Canonical tables remain page-local for exact
        # provenance. Only boundaries independently
        # qualified as CONTINUE are allowed to carry
        # structural state across pages.
        #
        # STOP, AMBIGUOUS, or unavailable continuation
        # evidence starts a fresh sequence.
        resolved_tables: list[
            CanonicalTable
        ] = []
        resolved_sequences: list[
            tuple[
                CanonicalTable,
                ...,
            ]
        ] = []
        sequences = (
            self._partition_continuation_sequences(
                canonical_tables
            )
        )
        for sequence in sequences:
            resolution = (
                self._structure.resolve(
                    sequence
                )
            )
            resolved_sequence = tuple(
                resolution.tables
            )
            resolved_sequences.append(
                resolved_sequence
            )
            resolved_tables.extend(
                resolved_sequence
            )
        return (
            TableEvidenceCompilationResult(
                tables=tuple(
                    resolved_tables
                ),
                matched_table_count=(
                    matched_count
                ),
                sequences=tuple(
                    resolved_sequences
                ),
                unmatched_physical_refs=tuple(
                    unmatched_physical
                ),
                unmatched_structural_refs=tuple(
                    unmatched_structural
                ),
                ambiguous_physical_refs=tuple(
                    ambiguous_physical
                ),
                ambiguous_structural_refs=tuple(
                    ambiguous_structural
                ),
            )
        )


TableEvidenceCompiler = (
    TableEvidenceCompilationService
)
