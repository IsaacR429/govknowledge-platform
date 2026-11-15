from __future__ import annotations

from app.domain.enums.table import (
    TableContinuationDecision,
)
from app.domain.models.table import (
    TableContinuationEvidence,
    TableContinuationResult,
)


class TableContinuityResolver:
    """Deterministically decide whether two physical tables continue.

    The resolver is deliberately parser-neutral.

    It consumes already-normalized evidence and never examines:
    - PDF bytes,
    - parser objects,
    - raw text streams,
    - LLM output.

    Fail-closed policy:
    positive continuation requires explicit structural evidence.
    Otherwise the result is STOP or AMBIGUOUS.
    """

    def resolve(
        self,
        evidence: TableContinuationEvidence,
    ) -> TableContinuationResult:

        if evidence.contradictory_boundary:
            return TableContinuationResult(
                decision=TableContinuationDecision.STOP,
                evidence=evidence,
                reasons=(
                    "A contradictory table boundary was detected.",
                ),
            )

        if not evidence.same_column_count:
            return TableContinuationResult(
                decision=TableContinuationDecision.STOP,
                evidence=evidence,
                reasons=(
                    "Logical column counts are incompatible.",
                ),
            )

        if (
            evidence.right_page
            != evidence.left_page + 1
        ):
            return TableContinuationResult(
                decision=(
                    TableContinuationDecision.AMBIGUOUS
                ),
                evidence=evidence,
                reasons=(
                    "Tables are not on adjacent pages.",
                ),
            )

        if not evidence.compatible_column_geometry:
            return TableContinuationResult(
                decision=(
                    TableContinuationDecision.AMBIGUOUS
                ),
                evidence=evidence,
                reasons=(
                    "Column geometry is not sufficiently "
                    "compatible to establish continuation.",
                ),
            )

        if evidence.shared_structural_identity:
            return TableContinuationResult(
                decision=(
                    TableContinuationDecision.CONTINUE
                ),
                evidence=evidence,
                reasons=(
                    (
                        "Adjacent table fragments share an "
                        "explicit structural table identity "
                        "and compatible column geometry."
                    ),
                ),
            )

        if evidence.repeated_header:
            reasons = [
                (
                    "Adjacent tables have compatible schema "
                    "and a repeated header."
                )
            ]

            if (
                evidence.left_has_active_hierarchy
                and evidence.right_starts_with_inherited_cells
            ):
                reasons.append(
                    (
                        "The right table begins with inherited "
                        "hierarchy cells compatible with active "
                        "parent state from the left table."
                    )
                )

            return TableContinuationResult(
                decision=(
                    TableContinuationDecision.CONTINUE
                ),
                evidence=evidence,
                reasons=tuple(reasons),
            )

        if (
            evidence.headerless_continuation
            and evidence.row_key_progression
        ):
            return TableContinuationResult(
                decision=(
                    TableContinuationDecision.CONTINUE
                ),
                evidence=evidence,
                reasons=(
                    (
                        "Adjacent headerless tables have "
                        "compatible schema and deterministic "
                        "row-key progression."
                    ),
                ),
            )

        if evidence.headerless_continuation:
            return TableContinuationResult(
                decision=(
                    TableContinuationDecision.AMBIGUOUS
                ),
                evidence=evidence,
                reasons=(
                    (
                        "Headerless continuation is possible, "
                        "but deterministic row progression "
                        "was not established."
                    ),
                ),
            )

        return TableContinuationResult(
            decision=(
                TableContinuationDecision.AMBIGUOUS
            ),
            evidence=evidence,
            reasons=(
                (
                    "Compatible table shape alone is "
                    "insufficient to establish continuation."
                ),
            ),
        )
