import pytest

from app.application.ports.table_question_planner import (
    PlannedTableLookup,
)
from app.application.services.table_question_interpretation import (
    InvalidTableQuestionPlanError,
    ValidatedTableQuestionInterpreter,
)


SECTIONS = (
    "LOT_1",
    "LOT_2",
)

COLUMNS = (
    "N°",
    "Soumissionnaires",
    "Montants lus",
    "Montants corrigés",
    "Observations",
)


class FakePlanner:
    def __init__(
        self,
        result,
    ):
        self.result = result
        self.calls = []

    def plan(
        self,
        context,
    ):
        self.calls.append(
            context
        )

        return self.result


class NeverPlanner:
    def __init__(self):
        self.calls = []

    def plan(
        self,
        context,
    ):
        self.calls.append(
            context
        )

        raise AssertionError(
            "Planner must not run "
            "without authorized vocabulary."
        )


def interpreter(
    result,
):
    planner = FakePlanner(
        result
    )

    service = (
        ValidatedTableQuestionInterpreter(
            planner=planner
        )
    )

    return (
        service,
        planner,
    )


def test_valid_plan_becomes_lookup_request():
    service, planner = (
        interpreter(
            PlannedTableLookup(
                section_key="LOT_2",
                ordinal=5,
                column_name=(
                    "Soumissionnaires"
                ),
            )
        )
    )

    result = service.interpret(
        question=(
            "Quel soumissionnaire "
            "correspond au numero "
            "05 du lot 2 ?"
        ),
        language="fr",
        section_keys=SECTIONS,
        column_names=COLUMNS,
    )

    assert result is not None

    assert (
        result.section_key
        == "LOT_2"
    )

    assert result.ordinal == 5

    assert (
        result.column_name
        == "Soumissionnaires"
    )

    assert len(
        planner.calls
    ) == 1

    context = planner.calls[0]

    assert (
        context.section_keys
        == SECTIONS
    )

    assert (
        context.column_names
        == COLUMNS
    )


def test_case_and_accents_resolve_to_canonical_values():
    service, _ = interpreter(
        PlannedTableLookup(
            section_key="lot 2",
            ordinal=5,
            column_name=(
                "montants corriges"
            ),
        )
    )

    result = service.interpret(
        question="Question",
        language="fr",
        section_keys=SECTIONS,
        column_names=COLUMNS,
    )

    assert result is not None

    assert (
        result.section_key
        == "LOT_2"
    )

    assert (
        result.column_name
        == "Montants corrigés"
    )


def test_unknown_section_fails_closed():
    service, _ = interpreter(
        PlannedTableLookup(
            section_key=(
                "SECRET_LOT"
            ),
            ordinal=5,
            column_name=(
                "Soumissionnaires"
            ),
        )
    )

    with pytest.raises(
        InvalidTableQuestionPlanError,
        match="section_key",
    ):
        service.interpret(
            question="Question",
            language="fr",
            section_keys=SECTIONS,
            column_names=COLUMNS,
        )


def test_unknown_column_fails_closed():
    service, _ = interpreter(
        PlannedTableLookup(
            section_key="LOT_2",
            ordinal=5,
            column_name=(
                "SecretField"
            ),
        )
    )

    with pytest.raises(
        InvalidTableQuestionPlanError,
        match="column_name",
    ):
        service.interpret(
            question="Question",
            language="fr",
            section_keys=SECTIONS,
            column_names=COLUMNS,
        )


@pytest.mark.parametrize(
    "ordinal",
    (
        0,
        -1,
        1000001,
    ),
)
def test_invalid_ordinal_fails_closed(
    ordinal,
):
    service, _ = interpreter(
        PlannedTableLookup(
            section_key="LOT_2",
            ordinal=ordinal,
            column_name=(
                "Soumissionnaires"
            ),
        )
    )

    with pytest.raises(
        InvalidTableQuestionPlanError,
        match="ordinal",
    ):
        service.interpret(
            question="Question",
            language="fr",
            section_keys=SECTIONS,
            column_names=COLUMNS,
        )


def test_no_authorized_vocabulary_skips_planner():
    planner = NeverPlanner()

    service = (
        ValidatedTableQuestionInterpreter(
            planner=planner
        )
    )

    result = service.interpret(
        question="Question",
        language="fr",
        section_keys=(),
        column_names=(),
    )

    assert result is None

    assert planner.calls == []


def test_planner_can_decline_table_interpretation():
    service, planner = (
        interpreter(None)
    )

    result = service.interpret(
        question=(
            "Resume ce document."
        ),
        language="fr",
        section_keys=SECTIONS,
        column_names=COLUMNS,
    )

    assert result is None

    assert len(
        planner.calls
    ) == 1


def test_language_is_normalized_for_planner():
    service, planner = (
        interpreter(None)
    )

    service.interpret(
        question="Question",
        language=" FR ",
        section_keys=SECTIONS,
        column_names=COLUMNS,
    )

    assert (
        planner.calls[0]
        .language
        == "fr"
    )
