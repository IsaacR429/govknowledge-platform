from __future__ import annotations

import pytest

from app.application.ports.table_evidence_repository import (
    AuthorizedGenericTable,
    AuthorizedLabeledRow,
)
from app.application.ports.table_question_planner import (
    PlannedLabeledRowLookup,
)
from app.application.services.table_evidence_atoms import (
    LabeledRowLookupRequest,
)
from app.application.services.table_question_interpretation import (
    InvalidTableQuestionPlanError,
    ValidatedTableQuestionInterpreter,
)


class FakePlanner:

    def __init__(self, result):
        self.result = result
        self.calls = []

    def plan(self, context):
        self.calls.append(context)
        return self.result


class NeverPlanner:

    def plan(self, context):
        raise AssertionError(
            "Planner must not run without "
            "authorized vocabulary."
        )


def generic_table(
    *,
    table_char="a",
    row_hash_char="1",
    column="Credit Score System",
):
    table_key = (
        "GENERIC/"
        + table_char * 64
    )

    return AuthorizedGenericTable(
        table_key=table_key,
        rows=(
            AuthorizedLabeledRow(
                label="Output",
                logical_key=(
                    table_key
                    + "/"
                    + row_hash_char * 64
                ),
            ),
            AuthorizedLabeledRow(
                label="PrimaryPurpose",
                logical_key=(
                    table_key
                    + "/"
                    + "2" * 64
                ),
            ),
        ),
        column_names=(
            "Feature",
            column,
        ),
    )


def test_generic_plan_becomes_labeled_row_request():

    table = generic_table()

    planner = FakePlanner(
        PlannedLabeledRowLookup(
            table_key=table.table_key,
            row_label="Output",
            column_name=(
                "Credit Score System"
            ),
        )
    )

    result = (
        ValidatedTableQuestionInterpreter(
            planner=planner
        )
        .interpret(
            question=(
                "What is the typical output "
                "of a credit score system?"
            ),
            language="en",
            section_keys=(),
            column_names=(),
            generic_tables=(
                table,
            ),
        )
    )

    assert isinstance(
        result,
        LabeledRowLookupRequest,
    )

    assert (
        result.table_key
        == table.table_key
    )

    assert result.row_label == "Output"

    assert (
        result.logical_key
        == table.rows[0].logical_key
    )

    assert (
        result.column_name
        == "Credit Score System"
    )

    assert len(
        planner.calls
    ) == 1

    context = planner.calls[0]

    assert context.section_keys == ()
    assert context.column_names == ()

    assert len(
        context.generic_tables
    ) == 1

    visible = (
        context.generic_tables[0]
    )

    assert visible.row_labels == (
        "Output",
        "PrimaryPurpose",
    )

    # Planner-visible generic descriptor must
    # not contain backend logical row identities.
    assert not hasattr(
        visible,
        "logical_key",
    )


def test_generic_plan_rejects_unknown_table_key():

    table = generic_table()

    planner = FakePlanner(
        PlannedLabeledRowLookup(
            table_key=(
                "GENERIC/"
                + "b" * 64
            ),
            row_label="Output",
            column_name=(
                "Credit Score System"
            ),
        )
    )

    with pytest.raises(
        InvalidTableQuestionPlanError,
        match="table_key",
    ):
        (
            ValidatedTableQuestionInterpreter(
                planner=planner
            )
            .interpret(
                question="What is the output?",
                language="en",
                section_keys=(),
                column_names=(),
                generic_tables=(
                    table,
                ),
            )
        )


def test_same_row_label_in_two_tables_is_table_scoped():

    first = generic_table(
        table_char="a",
        row_hash_char="1",
        column="Credit Score System",
    )

    second = generic_table(
        table_char="b",
        row_hash_char="3",
        column="Revenue",
    )

    planner = FakePlanner(
        PlannedLabeledRowLookup(
            table_key=second.table_key,
            row_label="Output",
            column_name="Revenue",
        )
    )

    result = (
        ValidatedTableQuestionInterpreter(
            planner=planner
        )
        .interpret(
            question="What is the output?",
            language="en",
            section_keys=(),
            column_names=(),
            generic_tables=(
                first,
                second,
            ),
        )
    )

    assert isinstance(
        result,
        LabeledRowLookupRequest,
    )

    assert (
        result.table_key
        == second.table_key
    )

    assert (
        result.logical_key
        == second.rows[0].logical_key
    )

    assert (
        result.column_name
        == "Revenue"
    )


def test_no_table_vocabulary_never_calls_planner():

    result = (
        ValidatedTableQuestionInterpreter(
            planner=NeverPlanner()
        )
        .interpret(
            question="Hello",
            language="en",
            section_keys=(),
            column_names=(),
            generic_tables=(),
        )
    )

    assert result is None
