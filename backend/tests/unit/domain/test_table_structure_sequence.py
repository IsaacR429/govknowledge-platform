from types import SimpleNamespace

import pytest

from app.domain.enums.table import (
    TableContinuationDecision,
)
from app.domain.tables.structure_sequence import (
    TableStructureSequenceResolutionError,
    TableStructureSequenceResolver,
)


class FakeStructureResolver:
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
    ):
        self.calls.append(
            (
                left,
                right,
            )
        )

        return self.decisions.pop(
            0
        )


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


def make_resolver(
    decision,
):
    structure = (
        FakeStructureResolver()
    )

    continuity = (
        FakeContinuityResolver()
    )

    builder = (
        FakeEvidenceBuilder(
            (
                decision,
            )
        )
    )

    resolver = (
        TableStructureSequenceResolver(
            continuity_resolver=(
                continuity
            ),
            structure_resolver=(
                structure
            ),
            continuation_evidence_builder=(
                builder
            ),
        )
    )

    return (
        resolver,
        structure,
        continuity,
        builder,
    )


def test_continue_carries_structure_state():
    (
        resolver,
        structure,
        _,
        builder,
    ) = make_resolver(
        boundary(
            TableContinuationDecision.CONTINUE,
            "continued",
        )
    )

    result = resolver.resolve(
        (
            "left",
            "right",
        )
    )

    assert builder.calls == [
        (
            "left",
            "right",
        ),
    ]

    assert structure.calls == [
        (
            "left",
            None,
        ),
        (
            "right",
            "state:left",
        ),
    ]

    assert result.tables == (
        "resolved:left",
        "resolved:right",
    )


def test_stop_resets_structure_state():
    (
        resolver,
        structure,
        _,
        _,
    ) = make_resolver(
        boundary(
            TableContinuationDecision.STOP,
            "new table",
        )
    )

    resolver.resolve(
        (
            "left",
            "right",
        )
    )

    assert structure.calls == [
        (
            "left",
            None,
        ),
        (
            "right",
            None,
        ),
    ]


def test_ambiguous_boundary_fails_closed():
    (
        resolver,
        structure,
        _,
        _,
    ) = make_resolver(
        boundary(
            TableContinuationDecision.AMBIGUOUS,
            "insufficient evidence",
        )
    )

    with pytest.raises(
        TableStructureSequenceResolutionError,
        match="Ambiguous table boundary",
    ):
        resolver.resolve(
            (
                "left",
                "right",
            )
        )

    assert structure.calls == [
        (
            "left",
            None,
        ),
    ]


def test_explicit_boundary_count_mismatch_fails_first():
    structure = (
        FakeStructureResolver()
    )

    resolver = (
        TableStructureSequenceResolver(
            structure_resolver=(
                structure
            ),
        )
    )

    with pytest.raises(
        TableStructureSequenceResolutionError,
        match="boundary count",
    ):
        resolver.resolve(
            (
                "one",
                "two",
                "three",
            ),
            boundaries=(),
        )

    assert structure.calls == []


def test_empty_sequence_is_valid():
    resolver = (
        TableStructureSequenceResolver()
    )

    result = resolver.resolve(
        ()
    )

    assert result.steps == ()
    assert (
        result.continuation_results
        == ()
    )
    assert result.tables == ()
    assert (
        result.final_state
        is None
    )
