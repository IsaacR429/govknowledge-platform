from __future__ import annotations

import unicodedata

from app.application.ports.table_evidence_repository import (
    AuthorizedGenericTable,
)
from app.application.ports.table_question_planner import (
    GenericTablePlanningDescriptor,
    PlannedLabeledRowLookup,
    PlannedTableLookup,
    TableQuestionPlanner,
    TableQuestionPlanningContext,
)
from app.application.services.table_evidence_atoms import (
    LabeledRowLookupRequest,
    TableLookupRequest,
)


class InvalidTableQuestionPlanError(
    RuntimeError
):
    pass


def _canonical_token(
    value: str,
) -> str:

    decomposed = unicodedata.normalize(
        "NFKD",
        value,
    )

    return "".join(
        character.casefold()
        for character in decomposed
        if (
            character.isalnum()
            and not unicodedata.combining(
                character
            )
        )
    )


def _resolve_unique(
    *,
    candidate: str,
    allowed: tuple[str, ...],
    field_name: str,
) -> str:

    cleaned = candidate.strip()

    if not cleaned:
        raise InvalidTableQuestionPlanError(
            f"{field_name} cannot be empty."
        )

    if cleaned in allowed:
        return cleaned

    normalized = _canonical_token(
        cleaned
    )

    matches = tuple(
        value
        for value in allowed
        if (
            _canonical_token(value)
            == normalized
        )
    )

    if len(matches) != 1:
        raise InvalidTableQuestionPlanError(
            f"Planner returned an unknown "
            f"or ambiguous {field_name}."
        )

    return matches[0]


class ValidatedTableQuestionInterpreter:
    """
    Security boundary between an untrusted planner
    and canonical typed table lookup requests.

    The planner may select only backend-provided
    structural vocabulary.

    Generic logical row keys are resolved here and
    are never supplied to the planner.
    """

    def __init__(
        self,
        *,
        planner: TableQuestionPlanner,
        max_ordinal: int = 1000000,
    ) -> None:

        if max_ordinal < 1:
            raise ValueError(
                "max_ordinal must be positive."
            )

        self._planner = planner
        self._max_ordinal = max_ordinal

    def interpret(
        self,
        *,
        question: str,
        language: str,
        section_keys: tuple[str, ...],
        column_names: tuple[str, ...],
        generic_tables: tuple[
            AuthorizedGenericTable,
            ...,
        ] = (),
    ) -> (
        TableLookupRequest
        | LabeledRowLookupRequest
        | None
    ):

        clean_question = question.strip()

        clean_language = (
            language
            .strip()
            .lower()
        )

        if not clean_question:
            raise ValueError(
                "question cannot be empty."
            )

        if not clean_language:
            raise ValueError(
                "language cannot be empty."
            )

        has_procurement = bool(
            section_keys
            and column_names
        )

        has_generic = bool(
            generic_tables
        )

        if (
            not has_procurement
            and not has_generic
        ):
            return None

        planning_generic_tables = tuple(
            GenericTablePlanningDescriptor(
                table_key=table.table_key,
                row_labels=tuple(
                    row.label
                    for row in table.rows
                ),
                column_names=(
                    table.column_names
                ),
            )
            for table in generic_tables
        )

        context = TableQuestionPlanningContext(
            question=clean_question,
            language=clean_language,
            section_keys=section_keys,
            column_names=column_names,
            generic_tables=(
                planning_generic_tables
            ),
        )

        plan = self._planner.plan(
            context
        )

        if plan is None:
            return None

        # ====================================================
        # EXISTING PROCUREMENT PLAN
        # ====================================================

        if isinstance(
            plan,
            PlannedTableLookup,
        ):

            if (
                isinstance(plan.ordinal, bool)
                or not isinstance(
                    plan.ordinal,
                    int,
                )
            ):
                raise InvalidTableQuestionPlanError(
                    "Planner ordinal "
                    "must be an integer."
                )

            if (
                plan.ordinal < 1
                or plan.ordinal
                > self._max_ordinal
            ):
                raise InvalidTableQuestionPlanError(
                    "Planner ordinal is "
                    "outside allowed range."
                )

            section_key = _resolve_unique(
                candidate=plan.section_key,
                allowed=section_keys,
                field_name="section_key",
            )

            column_name = _resolve_unique(
                candidate=plan.column_name,
                allowed=column_names,
                field_name="column_name",
            )

            return TableLookupRequest(
                section_key=section_key,
                ordinal=plan.ordinal,
                column_name=column_name,
            )

        # ====================================================
        # GENERIC LABELED-ROW PLAN
        # ====================================================

        if isinstance(
            plan,
            PlannedLabeledRowLookup,
        ):

            clean_table_key = (
                plan.table_key.strip()
            )

            matching_tables = tuple(
                table
                for table in generic_tables
                if (
                    table.table_key
                    == clean_table_key
                )
            )

            if len(matching_tables) != 1:
                raise (
                    InvalidTableQuestionPlanError(
                        "Planner returned an "
                        "unknown or ambiguous "
                        "table_key."
                    )
                )

            table = matching_tables[0]

            row_label = _resolve_unique(
                candidate=plan.row_label,
                allowed=tuple(
                    row.label
                    for row in table.rows
                ),
                field_name="row_label",
            )

            matching_rows = tuple(
                row
                for row in table.rows
                if row.label == row_label
            )

            if len(matching_rows) != 1:
                raise (
                    InvalidTableQuestionPlanError(
                        "Authorized generic row "
                        "label is ambiguous."
                    )
                )

            column_name = _resolve_unique(
                candidate=plan.column_name,
                allowed=(
                    table.column_names
                ),
                field_name="column_name",
            )

            row = matching_rows[0]

            return LabeledRowLookupRequest(
                table_key=table.table_key,
                row_label=row.label,
                logical_key=(
                    row.logical_key
                ),
                column_name=column_name,
            )

        raise InvalidTableQuestionPlanError(
            "Planner returned an unsupported "
            "table plan type."
        )
