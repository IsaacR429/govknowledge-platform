from types import SimpleNamespace

import pytest

from app.domain.enums.table import (
    TableContinuationDecision,
)
from app.domain.tables.sequence import (
    TableSequenceResolutionError,
    TableSequenceResolver,
)


class FakeHierarchyResolver:
    def __init__(
        self,
    ) -> None:
        self.calls = []

    def resolve_table(
        self,
        table,
        *,
        initial_state=None,
    ):
        self.calls.append(
            (
                table,
                initial_state,
            )
        )

        return SimpleNamespace(
            table=(
                f"resolved:{table}"
            ),
            final_state=(
                f"state:{table}"
            ),
            inherited_cell_count=(
                1
                if initial_state
                is not None
                else 0
            ),
        )


class FakeContinuityResolver:
    def __init__(
        self,
    ) -> None:
        self.calls = []

    def resolve(
        self,
        evidence,
    ):
        self.calls.append(
            evidence
        )

        return evidence


def boundary(
    decision,
    *reasons,
):
    return SimpleNamespace(
        decision=decision,
        reasons=tuple(
            reasons
        ),
    )


def make_resolver():
    hierarchy = (
        FakeHierarchyResolver()
    )

    continuity = (
        FakeContinuityResolver()
    )

    resolver = (
        TableSequenceResolver(
            continuity_resolver=(
                continuity
            ),
            hierarchy_resolver=(
                hierarchy
            ),
        )
    )

    return (
        resolver,
        hierarchy,
        continuity,
    )


def test_continue_carries_previous_hierarchy_state():
    (
        resolver,
        hierarchy,
        continuity,
    ) = make_resolver()

    result = resolver.resolve(
        (
            "left",
            "right",
        ),
        boundaries=(
            boundary(
                TableContinuationDecision.CONTINUE,
                "continued",
            ),
        ),
    )

    assert hierarchy.calls == [
        (
            "left",
            None,
        ),
        (
            "right",
            "state:left",
        ),
    ]

    assert len(
        continuity.calls
    ) == 1

    assert result.tables == (
        "resolved:left",
        "resolved:right",
    )

    assert (
        result.inherited_cell_count
        == 1
    )

    assert (
        result.final_state
        == "state:right"
    )


def test_stop_resets_hierarchy_state():
    (
        resolver,
        hierarchy,
        _,
    ) = make_resolver()

    result = resolver.resolve(
        (
            "left",
            "right",
        ),
        boundaries=(
            boundary(
                TableContinuationDecision.STOP,
                "new table",
            ),
        ),
    )

    assert hierarchy.calls == [
        (
            "left",
            None,
        ),
        (
            "right",
            None,
        ),
    ]

    assert (
        result.inherited_cell_count
        == 0
    )


def test_ambiguous_boundary_fails_closed():
    (
        resolver,
        hierarchy,
        _,
    ) = make_resolver()

    with pytest.raises(
        TableSequenceResolutionError,
        match=(
            "Ambiguous table boundary"
        ),
    ):
        resolver.resolve(
            (
                "left",
                "right",
            ),
            boundaries=(
                boundary(
                    TableContinuationDecision.AMBIGUOUS,
                    (
                        "shape alone is "
                        "insufficient"
                    ),
                ),
            ),
        )

    # The right table must never be resolved
    # with unapproved hierarchy state.
    assert hierarchy.calls == [
        (
            "left",
            None,
        ),
    ]


def test_boundary_count_mismatch_fails_before_resolution():
    (
        resolver,
        hierarchy,
        _,
    ) = make_resolver()

    with pytest.raises(
        TableSequenceResolutionError,
        match=(
            "boundary count"
        ),
    ):
        resolver.resolve(
            (
                "one",
                "two",
                "three",
            ),
            boundaries=(),
        )

    assert hierarchy.calls == []


def test_empty_sequence_is_valid():
    (
        resolver,
        hierarchy,
        continuity,
    ) = make_resolver()

    result = resolver.resolve(
        (),
        boundaries=(),
    )

    assert result.steps == ()
    assert (
        result.continuation_results
        == ()
    )
    assert result.tables == ()
    assert (
        result.inherited_cell_count
        == 0
    )
    assert (
        result.final_state
        is None
    )

    assert hierarchy.calls == []
    assert continuity.calls == []

class FakeEvidenceBuilder:
    def __init__(
        self,
        decisions,
    ) -> None:
        self.decisions = list(
            decisions
        )
        self.calls = []

    def build(
        self,
        *,
        left,
        right,
        left_state=None,
        explicit_contradictory_boundary=False,
    ):
        self.calls.append(
            (
                left,
                right,
                left_state,
            )
        )

        if not self.decisions:
            raise AssertionError(
                "No fake continuation "
                "decision remains."
            )

        return self.decisions.pop(0)


def test_automatic_boundary_builder_carries_state_on_continue():
    hierarchy = (
        FakeHierarchyResolver()
    )

    continuity = (
        FakeContinuityResolver()
    )

    evidence_builder = (
        FakeEvidenceBuilder(
            (
                boundary(
                    TableContinuationDecision.CONTINUE,
                    "automatic continue",
                ),
            )
        )
    )

    resolver = TableSequenceResolver(
        continuity_resolver=(
            continuity
        ),
        hierarchy_resolver=(
            hierarchy
        ),
        continuation_evidence_builder=(
            evidence_builder
        ),
    )

    result = resolver.resolve(
        (
            "left",
            "right",
        ),
    )

    assert (
        evidence_builder.calls
        == [
            (
                "left",
                "right",
                "state:left",
            ),
        ]
    )

    assert hierarchy.calls == [
        (
            "left",
            None,
        ),
        (
            "right",
            "state:left",
        ),
    ]

    assert (
        result.continuation_results[
            0
        ].decision
        == TableContinuationDecision.CONTINUE
    )


def test_automatic_boundary_builder_fails_closed_on_ambiguous():
    hierarchy = (
        FakeHierarchyResolver()
    )

    continuity = (
        FakeContinuityResolver()
    )

    evidence_builder = (
        FakeEvidenceBuilder(
            (
                boundary(
                    TableContinuationDecision.AMBIGUOUS,
                    "automatic ambiguity",
                ),
            )
        )
    )

    resolver = TableSequenceResolver(
        continuity_resolver=(
            continuity
        ),
        hierarchy_resolver=(
            hierarchy
        ),
        continuation_evidence_builder=(
            evidence_builder
        ),
    )

    with pytest.raises(
        TableSequenceResolutionError,
        match=(
            "Ambiguous table boundary"
        ),
    ):
        resolver.resolve(
            (
                "left",
                "right",
            ),
        )

    assert (
        evidence_builder.calls
        == [
            (
                "left",
                "right",
                "state:left",
            ),
        ]
    )

    # No hierarchy processing of the right table
    # occurs after an ambiguous boundary.
    assert hierarchy.calls == [
        (
            "left",
            None,
        ),
    ]
