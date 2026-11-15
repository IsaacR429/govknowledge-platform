from __future__ import annotations

from dataclasses import dataclass, replace
import re
import unicodedata

from app.domain.enums.table import (
    TableRowRole,
)
from app.domain.models.table import (
    CanonicalTable,
    CanonicalTableRow,
)


class TableStructureResolutionError(
    ValueError
):
    """Raised when logical table structure cannot be resolved safely."""


@dataclass(frozen=True)
class TableStructureState:
    """State that may cross a table boundary only after continuity approval."""

    section_key: str | None = None
    last_ordinal: int | None = None
    terminated: bool = False


@dataclass(frozen=True)
class TableStructureResolutionResult:
    table: CanonicalTable
    final_state: TableStructureState

    section_count: int
    keyed_row_count: int
    summary_count: int
    promoted_header_count: int


class TableStructureResolver:
    """Resolve deterministic logical row semantics.

    P0 rules:

    - recognized procurement headers become HEADER;
    - "Lot N" / "Lot unique" rows become SECTION;
    - numeric rows inside an active section become DATA rows
      with canonical logical keys such as LOT_2/03;
    - row numbers must progress deterministically;
    - a new section is the only normal numbering reset;
    - "ATTRIBUTAIRE(S) PROVISOIRE(S)" becomes SUMMARY;
    - unresolved numbered rows in an otherwise structured
      procurement table fail closed.

    The resolver never invents cell values and never changes
    cell provenance.
    """

    _SECTION_PATTERN = re.compile(
        r"(?i)\blot\s*(unique|\d+)\s*:"
    )

    _ORDINAL_PATTERN = re.compile(
        r"^\s*0*(\d{1,3})\s*[.)-]?\s*$"
    )

    def resolve_table(
        self,
        table: CanonicalTable,
        *,
        initial_state: (
            TableStructureState | None
        ) = None,
    ) -> TableStructureResolutionResult:

        state = (
            initial_state
            if initial_state is not None
            else TableStructureState()
        )

        active_section = (
            None
            if state.terminated
            else state.section_key
        )

        last_ordinal = (
            None
            if state.terminated
            else state.last_ordinal
        )

        terminated = (
            state.terminated
        )

        rows = list(
            table.rows
        )

        has_local_structure_marker = any(
            (
                self._section_key(row)
                is not None
                or self._is_summary(row)
            )
            for row in rows
        )

        has_structure_context = (
            has_local_structure_marker
            or active_section is not None
        )

        resolved_rows: list[
            CanonicalTableRow
        ] = []

        section_count = 0
        keyed_row_count = 0
        summary_count = 0
        promoted_header_count = 0

        for row in rows:

            # Preserve an already-qualified header.
            if row.role == TableRowRole.HEADER:
                resolved_rows.append(
                    row
                )
                continue

            if self._looks_like_procurement_header(
                row
            ):
                resolved_rows.append(
                    replace(
                        row,
                        role=(
                            TableRowRole.HEADER
                        ),
                        logical_key=None,
                        section_key=None,
                    )
                )

                promoted_header_count += 1
                continue

            # SUMMARY must take precedence over SECTION.
            #
            # Procurement summaries commonly contain phrases such
            # as "Lot 1 : ..." and "Lot 2 : ..." in their descriptive
            # text. If section detection runs first, those references
            # would incorrectly reopen a lot instead of terminating
            # the procurement sequence.
            if self._is_summary(
                row
            ):
                resolved_rows.append(
                    replace(
                        row,
                        role=(
                            TableRowRole.SUMMARY
                        ),
                        logical_key=None,
                        section_key=None,
                    )
                )

                summary_count += 1

                # A summary closes the active
                # numbered procurement sequence.
                active_section = None
                last_ordinal = None
                terminated = True

                continue

            section_key = (
                self._section_key(
                    row
                )
            )

            if section_key is not None:

                if (
                    active_section
                    != section_key
                ):
                    last_ordinal = 0

                active_section = (
                    section_key
                )

                terminated = False
                section_count += 1

                resolved_rows.append(
                    replace(
                        row,
                        role=(
                            TableRowRole.SECTION
                        ),
                        logical_key=None,
                        section_key=(
                            section_key
                        ),
                    )
                )

                continue

            ordinal = (
                self._ordinal(
                    row
                )
            )

            if ordinal is None:
                # Preamble rows, blank rows and other
                # unclassified material remain untouched.
                resolved_rows.append(
                    row
                )
                continue

            if active_section is None:

                if has_structure_context:
                    raise (
                        TableStructureResolutionError(
                            (
                                "Numbered procurement row "
                                f"{ordinal:02d} at row "
                                f"{row.row_index} has no "
                                "active section."
                            )
                        )
                    )

                # Generic numeric table with no procurement
                # section semantics: do not over-classify it.
                resolved_rows.append(
                    row
                )
                continue

            expected = (
                1
                if last_ordinal is None
                else last_ordinal + 1
            )

            if ordinal != expected:
                raise (
                    TableStructureResolutionError(
                        (
                            "Non-deterministic row progression "
                            f"in section {active_section}: "
                            f"expected {expected:02d}, "
                            f"found {ordinal:02d} "
                            f"at row {row.row_index}."
                        )
                    )
                )

            logical_key = (
                f"{active_section}/"
                f"{ordinal:02d}"
            )

            resolved_rows.append(
                replace(
                    row,
                    role=(
                        TableRowRole.DATA
                    ),
                    logical_key=(
                        logical_key
                    ),
                    section_key=(
                        active_section
                    ),
                )
            )

            last_ordinal = ordinal
            keyed_row_count += 1

        resolved_table = (
            replace(
                table,
                rows=tuple(
                    resolved_rows
                ),
            )
        )

        final_state = (
            TableStructureState(
                section_key=(
                    active_section
                ),
                last_ordinal=(
                    last_ordinal
                ),
                terminated=(
                    terminated
                ),
            )
        )

        return (
            TableStructureResolutionResult(
                table=resolved_table,
                final_state=final_state,
                section_count=(
                    section_count
                ),
                keyed_row_count=(
                    keyed_row_count
                ),
                summary_count=(
                    summary_count
                ),
                promoted_header_count=(
                    promoted_header_count
                ),
            )
        )

    @classmethod
    def _section_key(
        cls,
        row: CanonicalTableRow,
    ) -> str | None:

        text = (
            cls._row_text(
                row
            )
        )

        match = (
            cls._SECTION_PATTERN.search(
                text
            )
        )

        if match is None:
            return None

        value = (
            match.group(1)
            .strip()
            .upper()
        )

        if value == "UNIQUE":
            return "LOT_UNIQUE"

        number = int(
            value
        )

        return (
            f"LOT_{number}"
        )

    @classmethod
    def _ordinal(
        cls,
        row: CanonicalTableRow,
    ) -> int | None:

        first = (
            cls._atomic_cell_text(
                row=row,
                column=0,
            )
        )

        if first is None:
            return None

        match = (
            cls._ORDINAL_PATTERN.fullmatch(
                first
            )
        )

        if match is None:
            return None

        ordinal = int(
            match.group(1)
        )

        if ordinal < 1:
            return None

        return ordinal

    @classmethod
    def _looks_like_procurement_header(
        cls,
        row: CanonicalTableRow,
    ) -> bool:

        values = tuple(
            cls._normalized(
                cell.text
            )
            for cell in row.cells
            if cell.text.strip()
        )

        if len(values) < 4:
            return False

        compact = tuple(
            cls._compact(
                value
            )
            for value in values
        )

        has_number = any(
            value
            in {
                "n",
                "no",
                "numero",
                "nordre",
            }
            for value in compact
        )

        has_supplier = any(
            "soumissionnaire"
            in value
            for value in compact
        )

        montant_count = sum(
            "montant"
            in value
            for value in compact
        )

        has_observation = any(
            (
                "observation"
                in value
                or "classement"
                in value
                or "rang"
                in value
            )
            for value in compact
        )

        return (
            has_number
            and has_supplier
            and montant_count >= 2
            and has_observation
        )

    @classmethod
    def _is_summary(
        cls,
        row: CanonicalTableRow,
    ) -> bool:

        # Summary markers must begin the first meaningful
        # cell. Do not match incidental mentions such as:
        #
        # "RECTIFICATIF PORTANT SUR LE NOM DE
        #  L'ATTRIBUTAIRE PROVISOIRE ..."
        #
        # which is document preamble, not a summary row.
        first_text = next(
            (
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            ),
            "",
        )

        if not first_text:
            return False

        compact = (
            cls._compact(
                first_text
            )
        )

        return (
            compact.startswith(
                "attributairesprovisoires"
            )
            or compact.startswith(
                "attributaireprovisoire"
            )
        )

    @staticmethod
    def _atomic_cell_text(
        *,
        row: CanonicalTableRow,
        column: int,
    ) -> str | None:

        matches = tuple(
            cell
            for cell in row.cells
            if (
                cell.column_start
                == column
                and cell.column_end
                == column + 1
            )
        )

        if len(matches) != 1:
            return None

        return (
            matches[0]
            .text
            .strip()
        )

    @classmethod
    def _row_text(
        cls,
        row: CanonicalTableRow,
    ) -> str:

        return " ".join(
            cell.text.strip()
            for cell in row.cells
            if cell.text.strip()
        )

    @staticmethod
    def _normalized(
        value: str,
    ) -> str:

        normalized = (
            unicodedata.normalize(
                "NFKC",
                value,
            )
        )

        normalized = (
            normalized.replace(
                "\u00a0",
                " ",
            )
        )

        return " ".join(
            normalized.split()
        )

    @classmethod
    def _compact(
        cls,
        value: str,
    ) -> str:

        normalized = (
            unicodedata.normalize(
                "NFKD",
                cls._normalized(
                    value
                ),
            )
        )

        without_accents = "".join(
            character
            for character in normalized
            if not unicodedata.combining(
                character
            )
        )

        return "".join(
            character
            for character
            in without_accents.casefold()
            if character.isalnum()
        )
