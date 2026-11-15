from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenericTablePlanningDescriptor:
    """
    Planner-visible structure for one authorized
    generic table.

    It deliberately contains:
      - opaque table identity;
      - structural row labels;
      - column names.

    It deliberately does NOT contain:
      - logical row keys;
      - answer/data cell values.
    """

    table_key: str
    row_labels: tuple[str, ...]
    column_names: tuple[str, ...]

    def __post_init__(self) -> None:

        if (
            not isinstance(self.table_key, str)
            or not self.table_key.strip()
            or self.table_key != self.table_key.strip()
        ):
            raise ValueError(
                "table_key contains an invalid value."
            )

        if (
            len(set(self.row_labels))
            != len(self.row_labels)
        ):
            raise ValueError(
                "row_labels must be unique."
            )

        if (
            len(set(self.column_names))
            != len(self.column_names)
        ):
            raise ValueError(
                "column_names must be unique."
            )

        for value in self.row_labels:

            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(
                    "row_labels contains "
                    "an invalid value."
                )

        for value in self.column_names:

            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(
                    "column_names contains "
                    "an invalid value."
                )


@dataclass(frozen=True)
class TableQuestionPlanningContext:
    """
    Backend-approved structural vocabulary that an
    untrusted question interpreter may choose from.

    No answer/data cell values are present.
    """

    question: str
    language: str

    # Existing procurement vocabulary.
    section_keys: tuple[str, ...]
    column_names: tuple[str, ...]

    # Relationship-preserving generic tables.
    generic_tables: tuple[
        GenericTablePlanningDescriptor,
        ...,
    ] = ()

    def __post_init__(self) -> None:

        if not self.question.strip():
            raise ValueError(
                "question cannot be empty."
            )

        if not self.language.strip():
            raise ValueError(
                "language cannot be empty."
            )

        if (
            len(set(self.section_keys))
            != len(self.section_keys)
        ):
            raise ValueError(
                "section_keys must be unique."
            )

        if (
            len(set(self.column_names))
            != len(self.column_names)
        ):
            raise ValueError(
                "column_names must be unique."
            )

        generic_keys = tuple(
            table.table_key
            for table in self.generic_tables
        )

        if (
            len(set(generic_keys))
            != len(generic_keys)
        ):
            raise ValueError(
                "generic table keys must be unique."
            )


@dataclass(frozen=True)
class PlannedTableLookup:
    """
    Existing procurement-style untrusted plan.
    """

    section_key: str
    ordinal: int
    column_name: str


@dataclass(frozen=True)
class PlannedLabeledRowLookup:
    """
    Generic-table untrusted structural plan.

    logical_key is intentionally absent. The trusted
    interpreter resolves it from authorized vocabulary.
    """

    table_key: str
    row_label: str
    column_name: str


TableQuestionPlan = (
    PlannedTableLookup
    | PlannedLabeledRowLookup
)


class TableQuestionPlanner(Protocol):

    def plan(
        self,
        context: TableQuestionPlanningContext,
    ) -> TableQuestionPlan | None:
        ...
