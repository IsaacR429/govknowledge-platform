from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Protocol,
)
from uuid import UUID

from app.application.ports.authorization import (
    AuthorizationScope,
)


if TYPE_CHECKING:
    from app.application.services.table_evidence_atoms import (
        TableEvidenceAtom,
    )


@dataclass(frozen=True)
class AuthorizedLabeledRow:
    """
    One backend-qualified structural row label.

    logical_key is backend-generated and is never
    created by the question planner.
    """

    label: str
    logical_key: str

    def __post_init__(
        self,
    ) -> None:

        if (
            not isinstance(
                self.label,
                str,
            )
            or not self.label.strip()
            or self.label
            != self.label.strip()
        ):
            raise ValueError(
                "label contains an invalid value."
            )

        if (
            not isinstance(
                self.logical_key,
                str,
            )
            or not self.logical_key.strip()
            or self.logical_key
            != self.logical_key.strip()
        ):
            raise ValueError(
                "logical_key contains "
                "an invalid value."
            )

        parts = (
            self.logical_key.split("/")
        )

        if (
            len(parts) != 3
            or parts[0] != "GENERIC"
            or not parts[1]
            or not parts[2]
        ):
            raise ValueError(
                "logical_key is not a valid "
                "generic table key."
            )


@dataclass(frozen=True)
class AuthorizedGenericTable:
    """
    Structural vocabulary for one authorized
    generic table.

    Row labels are qualified structural labels.
    Answer/data cell values are deliberately absent.
    """

    table_key: str

    rows: tuple[
        AuthorizedLabeledRow,
        ...,
    ]

    column_names: tuple[
        str,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:

        if (
            not isinstance(
                self.table_key,
                str,
            )
            or not self.table_key.strip()
            or self.table_key
            != self.table_key.strip()
        ):
            raise ValueError(
                "table_key contains "
                "an invalid value."
            )

        table_parts = (
            self.table_key.split("/")
        )

        if (
            len(table_parts) != 2
            or table_parts[0] != "GENERIC"
            or not table_parts[1]
        ):
            raise ValueError(
                "table_key is not a valid "
                "generic table identity."
            )

        if not self.rows:
            raise ValueError(
                "generic table rows "
                "cannot be empty."
            )

        if not self.column_names:
            raise ValueError(
                "generic table column_names "
                "cannot be empty."
            )

        labels = tuple(
            row.label
            for row in self.rows
        )

        logical_keys = tuple(
            row.logical_key
            for row in self.rows
        )

        if (
            len(set(labels))
            != len(labels)
        ):
            raise ValueError(
                "generic row labels "
                "must be unique."
            )

        if (
            len(set(logical_keys))
            != len(logical_keys)
        ):
            raise ValueError(
                "generic row logical keys "
                "must be unique."
            )

        expected_prefix = (
            self.table_key + "/"
        )

        if any(
            not row.logical_key.startswith(
                expected_prefix
            )
            for row in self.rows
        ):
            raise ValueError(
                "generic row logical key "
                "does not belong to table."
            )

        if (
            len(set(self.column_names))
            != len(self.column_names)
        ):
            raise ValueError(
                "generic table column_names "
                "must be unique."
            )

        for value in self.column_names:

            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
                or value
                != value.strip()
            ):
                raise ValueError(
                    "generic table column_names "
                    "contains an invalid value."
                )


@dataclass(frozen=True)
class AuthorizedTableVocabulary:
    """
    Structural vocabulary visible inside one already
    authorized table-evidence scope.

    These values may be supplied to an untrusted
    question interpreter.

    Qualified first-column row labels are structural
    labels. Answer/data cell values are deliberately
    excluded.
    """

    # Existing procurement vocabulary.
    section_keys: tuple[
        str,
        ...,
    ]

    column_names: tuple[
        str,
        ...,
    ]

    # Generic tables deliberately preserve table ?
    # row ? column relationships instead of flattening
    # row labels across the authorized corpus.
    generic_tables: tuple[
        AuthorizedGenericTable,
        ...,
    ] = ()

    def __post_init__(
        self,
    ) -> None:

        for (
            field_name,
            values,
        ) in (
            (
                "section_keys",
                self.section_keys,
            ),
            (
                "column_names",
                self.column_names,
            ),
        ):

            if (
                len(set(values))
                != len(values)
            ):
                raise ValueError(
                    f"{field_name} "
                    "must be unique."
                )

            for value in values:

                if (
                    not isinstance(
                        value,
                        str,
                    )
                    or not value.strip()
                    or value
                    != value.strip()
                ):
                    raise ValueError(
                        f"{field_name} contains "
                        "an invalid value."
                    )

        table_keys = tuple(
            table.table_key
            for table
            in self.generic_tables
        )

        if (
            len(set(table_keys))
            != len(table_keys)
        ):
            raise ValueError(
                "generic table keys "
                "must be unique."
            )

        if any(
            not isinstance(
                table,
                AuthorizedGenericTable,
            )
            for table
            in self.generic_tables
        ):
            raise ValueError(
                "generic_tables contains "
                "an invalid value."
            )

    @classmethod
    def empty(
        cls,
    ) -> "AuthorizedTableVocabulary":

        return cls(
            section_keys=(),
            column_names=(),
            generic_tables=(),
        )


class TableEvidenceRepository(
    Protocol
):
    """Persistence boundary for typed table evidence.

    Reads MUST receive an already-authorized document scope.

    Implementations must apply that scope inside the persistence
    query itself; filtering after unrestricted retrieval is not
    considered authorization.
    """

    def replace_version_atoms(
        self,
        *,
        version_id: UUID,
        atoms: tuple[
            TableEvidenceAtom,
            ...,
        ],
        commit: bool = True,
    ) -> None:
        """Atomically replace persisted table evidence for a version."""
        ...

    def list_authorized_vocabulary(
        self,
        *,
        scope: AuthorizationScope,
    ) -> AuthorizedTableVocabulary:
        """
        Return only structural vocabulary visible
        inside the supplied authorization scope.
        """
        ...

    def find_authorized_by_evidence_id(
        self,
        *,
        scope: AuthorizationScope,
        evidence_id: str,
    ) -> TableEvidenceAtom | None:
        """Return one exact canonical atom inside authorization scope."""
        ...

    def find_authorized_by_logical_key(
        self,
        *,
        scope: AuthorizationScope,
        logical_key: str,
    ) -> tuple[
        TableEvidenceAtom,
        ...,
    ]:
        """Return only rows inside scope matching logical_key."""
        ...
