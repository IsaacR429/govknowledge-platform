from app.domain.enums.table import (
    TableContinuationDecision,
)
from app.domain.models.table import (
    TableContinuationEvidence,
)
from app.domain.tables.continuity import (
    TableContinuityResolver,
)


def evidence(
    *,
    left_page: int = 1,
    right_page: int = 2,
    same_column_count: bool = True,
    compatible_column_geometry: bool = True,
    repeated_header: bool = False,
    headerless_continuation: bool = False,
    row_key_progression: bool = False,
    left_has_active_hierarchy: bool = False,
    right_starts_with_inherited_cells: bool = False,
    contradictory_boundary: bool = False,
) -> TableContinuationEvidence:
    return TableContinuationEvidence(
        left_source_ref="left-table",
        right_source_ref="right-table",
        left_page=left_page,
        right_page=right_page,
        same_column_count=same_column_count,
        compatible_column_geometry=(
            compatible_column_geometry
        ),
        repeated_header=repeated_header,
        headerless_continuation=(
            headerless_continuation
        ),
        row_key_progression=row_key_progression,
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


def test_repeated_header_continues() -> None:
    resolver = TableContinuityResolver()

    result = resolver.resolve(
        evidence(
            repeated_header=True,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.CONTINUE
    )


def test_repeated_header_can_preserve_hierarchy_reason() -> None:
    resolver = TableContinuityResolver()

    result = resolver.resolve(
        evidence(
            repeated_header=True,
            left_has_active_hierarchy=True,
            right_starts_with_inherited_cells=True,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.CONTINUE
    )

    assert any(
        "hierarchy" in reason.lower()
        for reason in result.reasons
    )


def test_headerless_with_row_progression_continues() -> None:
    resolver = TableContinuityResolver()

    result = resolver.resolve(
        evidence(
            headerless_continuation=True,
            row_key_progression=True,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.CONTINUE
    )


def test_headerless_without_row_progression_is_ambiguous() -> None:
    resolver = TableContinuityResolver()

    result = resolver.resolve(
        evidence(
            headerless_continuation=True,
            row_key_progression=False,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.AMBIGUOUS
    )


def test_same_schema_without_structural_signal_is_ambiguous() -> None:
    resolver = TableContinuityResolver()

    result = resolver.resolve(
        evidence()
    )

    assert (
        result.decision
        == TableContinuationDecision.AMBIGUOUS
    )


def test_contradictory_boundary_stops() -> None:
    resolver = TableContinuityResolver()

    result = resolver.resolve(
        evidence(
            repeated_header=True,
            contradictory_boundary=True,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.STOP
    )


def test_column_count_mismatch_stops() -> None:
    resolver = TableContinuityResolver()

    result = resolver.resolve(
        evidence(
            same_column_count=False,
            repeated_header=True,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.STOP
    )


def test_incompatible_geometry_does_not_continue() -> None:
    resolver = TableContinuityResolver()

    result = resolver.resolve(
        evidence(
            compatible_column_geometry=False,
            repeated_header=True,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.AMBIGUOUS
    )


def test_non_adjacent_tables_do_not_continue() -> None:
    resolver = TableContinuityResolver()

    result = resolver.resolve(
        evidence(
            left_page=1,
            right_page=3,
            repeated_header=True,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.AMBIGUOUS
    )


def test_boundary_has_priority_over_positive_evidence() -> None:
    resolver = TableContinuityResolver()

    result = resolver.resolve(
        evidence(
            repeated_header=True,
            headerless_continuation=True,
            row_key_progression=True,
            contradictory_boundary=True,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.STOP
    )



def test_shared_structural_identity_continues():
    from dataclasses import replace

    result = TableContinuityResolver().resolve(
        replace(
            evidence(),
            shared_structural_identity=True,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.CONTINUE
    )

    assert any(
        "structural" in reason.lower()
        for reason in result.reasons
    )


def test_contradictory_boundary_wins_over_shared_structural_identity():
    from dataclasses import replace

    result = TableContinuityResolver().resolve(
        replace(
            evidence(
                contradictory_boundary=True,
            ),
            shared_structural_identity=True,
        )
    )

    assert (
        result.decision
        == TableContinuationDecision.STOP
    )
