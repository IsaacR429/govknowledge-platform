from __future__ import annotations

import json
from types import SimpleNamespace

from app.application.ports.table_question_planner import (
    GenericTablePlanningDescriptor,
    PlannedLabeledRowLookup,
    TableQuestionPlanningContext,
)
from app.infrastructure.llm.anthropic_table_question_planner import (
    AnthropicTableQuestionPlanner,
)


class FakeMessages:

    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=json.dumps(
                        self.response
                    ),
                )
            ]
        )


class FakeClient:

    def __init__(self, response):
        self.messages = FakeMessages(
            response
        )


def test_generic_labeled_row_plan_shape_and_payload():

    table_key = (
        "GENERIC/"
        + "a" * 64
    )

    client = FakeClient(
        {
            "decision": (
                "LABELED_ROW_LOOKUP"
            ),
            "table_key": table_key,
            "row_label": "Output",
            "column_name": (
                "Credit Score System"
            ),
        }
    )

    planner = AnthropicTableQuestionPlanner(
        client=client,
        model="test-model",
    )

    result = planner.plan(
        TableQuestionPlanningContext(
            question=(
                "What is the typical output "
                "of a credit score system?"
            ),
            language="en",
            section_keys=(),
            column_names=(),
            generic_tables=(
                GenericTablePlanningDescriptor(
                    table_key=table_key,
                    row_labels=(
                        "Output",
                        "PrimaryPurpose",
                    ),
                    column_names=(
                        "Feature",
                        "Credit Score System",
                    ),
                ),
            ),
        )
    )

    assert isinstance(
        result,
        PlannedLabeledRowLookup,
    )

    assert result.table_key == table_key
    assert result.row_label == "Output"

    call = client.messages.calls[0]

    payload = json.loads(
        call["messages"][0]["content"]
    )

    assert payload[
        "allowed_section_keys"
    ] == []

    assert payload[
        "allowed_column_names"
    ] == []

    generic = payload[
        "allowed_generic_tables"
    ]

    assert len(generic) == 1

    assert generic[0] == {
        "table_key": table_key,
        "row_labels": [
            "Output",
            "PrimaryPurpose",
        ],
        "column_names": [
            "Feature",
            "Credit Score System",
        ],
    }

    serialized = json.dumps(
        payload
    )

    assert "logical_key" not in serialized
    assert "300" not in serialized
    assert "900" not in serialized


def test_generic_labeled_row_comparison_preserves_multiple_columns():

    table_key = (
        "GENERIC/"
        + "c" * 64
    )

    client = FakeClient(
        {
            "decision": (
                "LABELED_ROW_COMPARISON"
            ),
            "table_key": table_key,
            "row_label": (
                "Chloroplastes"
            ),
            "column_names": [
                "Celluleanimale",
                "Cellulev\u00e9g\u00e9tale",
            ],
        }
    )

    planner = (
        AnthropicTableQuestionPlanner(
            client=client,
            model="test-model",
        )
    )

    result = planner.plan(
        TableQuestionPlanningContext(
            question=(
                "Compare les chloroplastes "
                "chez la cellule animale "
                "et la cellule v\u00e9g\u00e9tale."
            ),
            language="fr",
            section_keys=(),
            column_names=(),
            generic_tables=(
                GenericTablePlanningDescriptor(
                    table_key=table_key,
                    row_labels=(
                        "Chloroplastes",
                        "Vacuoles",
                    ),
                    column_names=(
                        "Crit\u00e8res",
                        "Celluleanimale",
                        "Cellulev\u00e9g\u00e9tale",
                    ),
                ),
            ),
        )
    )

    assert result is not None

    assert (
        result.table_key
        == table_key
    )

    assert (
        result.row_label
        == "Chloroplastes"
    )

    assert (
        result.column_names
        == (
            "Celluleanimale",
            "Cellulev\u00e9g\u00e9tale",
        )
    )
