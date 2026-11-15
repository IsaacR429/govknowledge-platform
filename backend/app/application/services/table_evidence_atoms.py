from __future__ import annotations

from hashlib import sha256

from dataclasses import dataclass
import unicodedata
from uuid import UUID

from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
)
from app.domain.models.table import (
    CanonicalTable,
    CanonicalTableCell,
    CanonicalTableRow,
    TableSourceSpan,
)


class TableEvidenceBuildError(
    ValueError
):
    """Raised when canonical table evidence cannot be exposed safely."""


class TableEvidenceLookupError(
    ValueError
):
    """Raised when typed table lookup is ambiguous or invalid."""


@dataclass(frozen=True)
class TableDocumentContext:
    document_id: UUID
    document_version_id: UUID
    page_ids: tuple[
        tuple[int, UUID],
        ...,
    ]
    language: str = "fr"

    def __post_init__(
        self,
    ) -> None:

        if not self.language.strip():
            raise ValueError(
                "language cannot be empty."
            )

        page_numbers = tuple(
            page_number
            for page_number, _
            in self.page_ids
        )

        if any(
            page_number < 1
            for page_number
            in page_numbers
        ):
            raise ValueError(
                "page numbers must be positive."
            )

        if (
            len(set(page_numbers))
            != len(page_numbers)
        ):
            raise ValueError(
                "page_ids cannot contain "
                "duplicate page numbers."
            )

    def page_id(
        self,
        page_number: int,
    ) -> UUID | None:

        for number, page_id in self.page_ids:

            if number == page_number:
                return page_id

        return None


@dataclass(frozen=True)
class TableCellEvidenceAtom:
    column_start: int
    column_end: int
    column_name: str | None

    text: str

    value_source: (
        TableCellValueSource
    )

    origin_spans: tuple[
        TableSourceSpan,
        ...,
    ] = ()

    structure_spans: tuple[
        TableSourceSpan,
        ...,
    ] = ()

    @property
    def origin_pages(
        self,
    ) -> tuple[int, ...]:

        return tuple(
            sorted(
                {
                    span.page_number
                    for span
                    in self.origin_spans
                }
            )
        )

    @property
    def structure_pages(
        self,
    ) -> tuple[int, ...]:

        return tuple(
            sorted(
                {
                    span.page_number
                    for span
                    in self.structure_spans
                }
            )
        )


@dataclass(frozen=True)
class TableEvidenceAtom:
    """Typed, retrieval-ready logical table evidence.

    search_text is derived retrieval material.

    It is NOT itself canonical citation provenance.
    Citation provenance remains attached to individual
    cells through origin_spans / structure_spans.
    """

    evidence_id: str

    document_id: UUID
    document_version_id: UUID

    page_id: UUID
    page_number: int

    source_table_refs: tuple[
        str,
        ...,
    ]

    row_index: int

    role: TableRowRole

    logical_key: str | None
    section_key: str | None

    column_names: tuple[
        str,
        ...,
    ]

    cells: tuple[
        TableCellEvidenceAtom,
        ...,
    ]

    language: str

    @property
    def origin_pages(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    page
                    for cell in self.cells
                    for page in cell.origin_pages
                }
            )
        )

    @property
    def structure_pages(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    page
                    for cell in self.cells
                    for page in cell.structure_pages
                }
            )
        )

    @property
    def spans_multiple_physical_pages(
        self,
    ) -> bool:
        return (
            len(
                self.structure_pages
            )
            > 1
        )

    @property
    def search_text(
        self,
    ) -> str:

        parts = []

        if self.logical_key:
            parts.append(
                self.logical_key
            )

        if self.section_key:
            parts.append(
                self.section_key
            )

        for cell in self.cells:

            value = (
                cell.text.strip()
            )

            if not value:
                continue

            if cell.column_name:
                parts.append(
                    (
                        f"{cell.column_name}: "
                        f"{value}"
                    )
                )

            else:
                parts.append(
                    value
                )

        return " | ".join(
            parts
        )


@dataclass(frozen=True)
class TableLookupRequest:
    section_key: str
    ordinal: int
    column_name: str

    def __post_init__(
        self,
    ) -> None:

        if not self.section_key.strip():
            raise ValueError(
                "section_key cannot be empty."
            )

        if self.ordinal < 1:
            raise ValueError(
                "ordinal must be positive."
            )

        if not self.column_name.strip():
            raise ValueError(
                "column_name cannot be empty."
            )

    @property
    def logical_key(
        self,
    ) -> str:

        return (
            f"{self.section_key.strip().upper()}/"
            f"{self.ordinal:02d}"
        )



@dataclass(frozen=True)
class LabeledRowLookupRequest:
    """
    Backend-resolved lookup for one labeled generic row.

    The untrusted planner does not invent logical_key.
    A validated interpreter will select this exact
    identity from authorized vocabulary.
    """

    table_key: str
    row_label: str
    logical_key: str
    column_name: str

    def __post_init__(
        self,
    ) -> None:

        for field_name, value in (
            (
                "table_key",
                self.table_key,
            ),
            (
                "row_label",
                self.row_label,
            ),
            (
                "logical_key",
                self.logical_key,
            ),
            (
                "column_name",
                self.column_name,
            ),
        ):

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
                    f"{field_name} cannot "
                    "be empty or padded."
                )

        table_parts = (
            self.table_key.split("/")
        )

        if (
            len(table_parts) != 2
            or table_parts[0]
            != "GENERIC"
            or not table_parts[1]
        ):
            raise ValueError(
                "table_key is not a valid "
                "generic table identity."
            )

        logical_parts = (
            self.logical_key.split("/")
        )

        if (
            len(logical_parts) != 3
            or logical_parts[0]
            != "GENERIC"
            or not logical_parts[1]
            or not logical_parts[2]
        ):
            raise ValueError(
                "logical_key is not a valid "
                "generic row identity."
            )

        if (
            "/".join(
                logical_parts[:2]
            )
            != self.table_key
        ):
            raise ValueError(
                "logical_key does not belong "
                "to table_key."
            )


@dataclass(frozen=True)
class TableLookupSelection:
    request: (
        TableLookupRequest
        | LabeledRowLookupRequest
    )
    row: TableEvidenceAtom
    cell: TableCellEvidenceAtom


class TableEvidenceAtomBuilder:
    """Convert resolved canonical tables into typed evidence atoms.

    Column schema is carried across consecutive canonical
    tables in a previously-approved logical sequence.

    Only table-semantic rows are exposed:
    - keyed DATA
    - SECTION
    - SUMMARY

    Preamble rows and header rows themselves are not emitted
    as retrieval atoms.
    """

    def build_sequence(
        self,
        *,
        tables: tuple[
            CanonicalTable,
            ...,
        ],
        context: TableDocumentContext,
    ) -> tuple[
        TableEvidenceAtom,
        ...,
    ]:

        atoms: list[
            TableEvidenceAtom
        ] = []

        active_column_names: (
            tuple[str, ...] | None
        ) = None

        seen_evidence_ids: set[
            str
        ] = set()

        sequence_identity_refs = (
            self._generic_sequence_identity_refs(
                tables
            )
        )

        for table_index, table in enumerate(
            tables
        ):

            table_schema = (
                self._schema_from_table(
                    table
                )
            )

            if table_schema is not None:
                active_column_names = (
                    table_schema
                )

            elif (
                active_column_names
                is not None
                and len(
                    active_column_names
                )
                != table.column_count
            ):
                raise (
                    TableEvidenceBuildError(
                        (
                            "Carried table schema "
                            "column count does not match "
                            "the next canonical table."
                        )
                    )
                )

            generic_logical_keys = (
                self._generic_logical_keys(
                    table=table,
                    version_id=(
                        context.document_version_id
                    ),
                    identity_refs=(
                        sequence_identity_refs
                    ),
                    allow_carried_schema=(
                        table_index > 0
                        and table_schema is None
                        and active_column_names
                        is not None
                    ),
                )
                if active_column_names is not None
                else {}
            )

            for row in table.rows:

                logical_key = (
                    row.logical_key
                    or generic_logical_keys.get(
                        row.row_index
                    )
                )

                if not self._should_emit(
                    row,
                    logical_key=logical_key,
                ):
                    continue

                page_number = (
                    self._row_page(
                        row
                    )
                )

                page_id = (
                    context.page_id(
                        page_number
                    )
                )

                if page_id is None:
                    raise (
                        TableEvidenceBuildError(
                            (
                                "No page_id mapping exists "
                                f"for physical page "
                                f"{page_number}."
                            )
                        )
                    )

                evidence_id = (
                    self._evidence_id(
                        table=table,
                        row=row,
                        logical_key=logical_key,
                        page_number=(
                            page_number
                        ),
                        version_id=(
                            context
                            .document_version_id
                        ),
                    )
                )

                if (
                    evidence_id
                    in seen_evidence_ids
                ):
                    raise (
                        TableEvidenceBuildError(
                            (
                                "Duplicate table evidence "
                                f"identifier: "
                                f"{evidence_id}"
                            )
                        )
                    )

                seen_evidence_ids.add(
                    evidence_id
                )

                cells = tuple(
                    self._cell_atom(
                        cell=cell,
                        column_names=(
                            active_column_names
                        ),
                    )
                    for cell in row.cells
                )

                atoms.append(
                    TableEvidenceAtom(
                        evidence_id=(
                            evidence_id
                        ),
                        document_id=(
                            context.document_id
                        ),
                        document_version_id=(
                            context
                            .document_version_id
                        ),
                        page_id=page_id,
                        page_number=(
                            page_number
                        ),
                        source_table_refs=(
                            table
                            .source_table_refs
                        ),
                        row_index=(
                            row.row_index
                        ),
                        role=row.role,
                        logical_key=(
                            logical_key
                        ),
                        section_key=(
                            row.section_key
                        ),
                        column_names=(
                            active_column_names
                            or ()
                        ),
                        cells=cells,
                        language=(
                            context.language
                            .strip()
                            .lower()
                        ),
                    )
                )

        return tuple(
            atoms
        )

    @staticmethod
    def _should_emit(
        row: CanonicalTableRow,
        *,
        logical_key: str | None,
    ) -> bool:

        if (
            row.role
            == TableRowRole.DATA
        ):
            return (
                logical_key
                is not None
            )

        return (
            row.role
            in {
                TableRowRole.SECTION,
                TableRowRole.SUMMARY,
            }
        )

    @staticmethod
    def _generic_sequence_identity_refs(
        tables: tuple[
            CanonicalTable,
            ...,
        ],
    ) -> tuple[str, ...]:
        if not tables:
            return ()

        if len(tables) == 1:
            return tuple(
                sorted(
                    {
                        value.strip()
                        for value
                        in tables[0].source_table_refs
                        if value.strip()
                    }
                )
            )

        structural_sets = tuple(
            {
                value.strip()
                for value
                in table.structural_identity_refs
                if value.strip()
            }
            for table in tables
        )

        if all(
            structural_sets
        ):
            shared = set.intersection(
                *structural_sets
            )

            if shared:
                return tuple(
                    sorted(
                        shared
                    )
                )

        # build_sequence() operates only on an already-approved
        # logical continuation sequence. When no parser-neutral
        # stable structural identity is available, use the full
        # sequence provenance rather than any single fragment's
        # page-local provenance.
        return tuple(
            sorted(
                {
                    value.strip()
                    for table in tables
                    for value
                    in table.source_table_refs
                    if value.strip()
                }
            )
        )

    @classmethod
    def _generic_logical_keys(
        cls,
        *,
        table: CanonicalTable,
        version_id: UUID,
        identity_refs: tuple[str, ...],
        allow_carried_schema: bool = False,
    ) -> dict[int, str]:
        """
        Derive addressable logical keys for ordinary
        labeled DATA rows.

        Existing domain-assigned logical keys always win.

        A generic row label is accepted only when its
        first-column cell is:
        - atomic;
        - nonblank;
        - direct SOURCE_TEXT;
        - source-backed.

        Duplicate normalized labels are deliberately
        excluded rather than disambiguated artificially.
        """

        identity_refs = tuple(
            sorted(
                {
                    value.strip()
                    for value
                    in identity_refs
                    if value.strip()
                }
            )
        )

        if not identity_refs:
            return {}

        # A local header normally establishes generic
        # addressability.
        #
        # A headerless fragment may also be addressable
        # when build_sequence() is processing it after
        # another table inside the SAME already-approved
        # continuation sequence. In that case the active
        # schema is trusted only for this sequence.
        header_rows = tuple(
            row
            for row in table.rows
            if (
                row.role
                == TableRowRole.HEADER
            )
        )

        if len(header_rows) > 1:
            return {}

        if (
            not table.column_names
            and not header_rows
            and not allow_carried_schema
        ):
            return {}

        header_row_index = (
            header_rows[0].row_index
            if header_rows
            else None
        )

        candidates: list[
            tuple[int, str]
        ] = []

        for row in table.rows:

            if (
                row.role
                != TableRowRole.DATA
                or row.logical_key
                is not None
                or row.section_key
                is not None
            ):
                continue

            # A DATA-like row occurring before a real
            # header is preamble material, not a generic
            # addressable table row.
            if (
                header_row_index
                is not None
                and row.row_index
                <= header_row_index
            ):
                continue

            first_cells = tuple(
                cell
                for cell
                in row.cells
                if (
                    cell.column_start == 0
                    and cell.column_end == 1
                )
            )

            if len(first_cells) != 1:
                continue

            label_cell = first_cells[0]

            # A heading/preamble containing only its
            # first-column label has no row value to
            # address and must not become evidence.
            has_nonlabel_value = any(
                cell.text.strip()
                for cell in row.cells
                if cell is not label_cell
            )

            if not has_nonlabel_value:
                continue

            if (
                label_cell.value_source
                != TableCellValueSource.SOURCE_TEXT
                or not label_cell.origin_spans
            ):
                continue

            normalized_label = (
                cls._normalize_generic_label(
                    label_cell.text
                )
            )

            if not normalized_label:
                continue

            candidates.append(
                (
                    row.row_index,
                    normalized_label,
                )
            )

        counts: dict[str, int] = {}

        for _, normalized_label in candidates:
            counts[normalized_label] = (
                counts.get(
                    normalized_label,
                    0,
                )
                + 1
            )

        table_key = (
            cls._generic_table_key(
                version_id=version_id,
                source_refs=identity_refs,
            )
        )

        result: dict[int, str] = {}

        for (
            row_index,
            normalized_label,
        ) in candidates:

            if (
                counts[
                    normalized_label
                ]
                != 1
            ):
                continue

            row_key = sha256(
                normalized_label.encode(
                    "utf-8"
                )
            ).hexdigest()

            result[row_index] = (
                "GENERIC/"
                f"{table_key}/"
                f"{row_key}"
            )

        return result

    @staticmethod
    def _generic_table_key(
        *,
        version_id: UUID,
        source_refs: tuple[str, ...],
    ) -> str:

        material = (
            str(version_id)
            + "\x1f"
            + "\x1f".join(
                source_refs
            )
        )

        return sha256(
            material.encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _normalize_generic_label(
        value: str,
    ) -> str:

        decomposed = unicodedata.normalize(
            "NFKD",
            value,
        )

        return "".join(
            character.casefold()
            for character
            in decomposed
            if (
                character.isalnum()
                and not unicodedata.combining(
                    character
                )
            )
        )

    @classmethod
    def _schema_from_table(
        cls,
        table: CanonicalTable,
    ) -> tuple[
        str,
        ...,
    ] | None:

        if table.column_names:

            if (
                len(table.column_names)
                != table.column_count
            ):
                raise (
                    TableEvidenceBuildError(
                        (
                            "Canonical table column_names "
                            "do not match column_count."
                        )
                    )
                )

            if any(
                not value.strip()
                for value
                in table.column_names
            ):
                raise (
                    TableEvidenceBuildError(
                        "Table column names cannot be blank."
                    )
                )

            return tuple(
                value.strip()
                for value
                in table.column_names
            )

        headers = tuple(
            row
            for row in table.rows
            if (
                row.role
                == TableRowRole.HEADER
            )
        )

        if not headers:
            return None

        if len(headers) != 1:
            raise (
                TableEvidenceBuildError(
                    (
                        "Expected at most one "
                        "logical HEADER row per "
                        "canonical table."
                    )
                )
            )

        header = (
            headers[0]
        )

        names = []

        for column in range(
            table.column_count
        ):

            matches = tuple(
                cell
                for cell in header.cells
                if (
                    cell.column_start
                    == column
                    and cell.column_end
                    == column + 1
                )
            )

            if len(matches) != 1:
                raise (
                    TableEvidenceBuildError(
                        (
                            "HEADER schema requires "
                            "exactly one atomic cell "
                            f"for column {column}."
                        )
                    )
                )

            value = (
                matches[0]
                .text
                .strip()
            )

            if not value:
                raise (
                    TableEvidenceBuildError(
                        (
                            "HEADER schema contains "
                            f"a blank column at "
                            f"{column}."
                        )
                    )
                )

            names.append(
                value
            )

        return tuple(
            names
        )

    @staticmethod
    def _row_page(
        row: CanonicalTableRow,
    ) -> int:

        pages = tuple(
            sorted(
                {
                    span.page_number
                    for span
                    in row.source_spans
                }
            )
        )

        if len(pages) != 1:
            raise (
                TableEvidenceBuildError(
                    (
                        "Table evidence rows must "
                        "represent exactly one "
                        "physical page; found "
                        f"{pages!r} for row "
                        f"{row.row_index}."
                    )
                )
            )

        return pages[0]

    @staticmethod
    def _cell_atom(
        *,
        cell: CanonicalTableCell,
        column_names: (
            tuple[str, ...] | None
        ),
    ) -> TableCellEvidenceAtom:

        column_name = None

        if (
            column_names is not None
            and cell.column_end
            == cell.column_start + 1
            and 0
            <= cell.column_start
            < len(column_names)
        ):
            column_name = (
                column_names[
                    cell.column_start
                ]
            )

        return (
            TableCellEvidenceAtom(
                column_start=(
                    cell.column_start
                ),
                column_end=(
                    cell.column_end
                ),
                column_name=(
                    column_name
                ),
                text=cell.text,
                value_source=(
                    cell.value_source
                ),
                origin_spans=(
                    cell.origin_spans
                ),
                structure_spans=(
                    cell.structure_spans
                ),
            )
        )

    @staticmethod
    def _evidence_id(
        *,
        table: CanonicalTable,
        row: CanonicalTableRow,
        logical_key: str | None,
        page_number: int,
        version_id: UUID,
    ) -> str:

        source = (
            table.source_table_refs[0]
            if table.source_table_refs
            else "unknown-table"
        )

        semantic_key = (
            logical_key
            or row.section_key
            or (
                f"{row.role.value}:"
                f"{row.row_index}"
            )
        )

        return (
            "table:"
            f"{version_id}:"
            f"p{page_number}:"
            f"{source}:"
            f"{semantic_key}"
        )


class TableEvidenceLookupService:
    """Deterministic typed lookup over authorized table atoms.

    Authorization is intentionally NOT performed here.
    Only atoms already returned through an authorized boundary
    may be supplied to this service.
    """

    def lookup(
        self,
        *,
        request: (
            TableLookupRequest
            | LabeledRowLookupRequest
        ),
        evidence: tuple[
            TableEvidenceAtom,
            ...,
        ],
    ) -> TableLookupSelection | None:

        logical_key = (
            request.logical_key
        )

        rows = tuple(
            atom
            for atom in evidence
            if (
                atom.role
                == TableRowRole.DATA
                and atom.logical_key
                == logical_key
            )
        )

        if not rows:
            return None

        if len(rows) != 1:
            raise (
                TableEvidenceLookupError(
                    (
                        "Logical table key is "
                        f"ambiguous: {logical_key}"
                    )
                )
            )

        row = rows[0]

        if isinstance(
            request,
            LabeledRowLookupRequest,
        ):

            label_cells = tuple(
                cell
                for cell in row.cells
                if (
                    cell.column_start == 0
                    and cell.column_end == 1
                )
            )

            if len(label_cells) != 1:
                raise (
                    TableEvidenceLookupError(
                        "Generic table row does not "
                        "contain exactly one "
                        "first-column row label."
                    )
                )

            if (
                self._normalize_name(
                    label_cells[0].text
                )
                != self._normalize_name(
                    request.row_label
                )
            ):
                raise (
                    TableEvidenceLookupError(
                        "Selected generic table "
                        "row label does not match "
                        "the lookup request."
                    )
                )

        requested_column = (
            self._normalize_name(
                request.column_name
            )
        )

        matches = tuple(
            cell
            for cell in row.cells
            if (
                cell.column_name
                is not None
                and self._normalize_name(
                    cell.column_name
                )
                == requested_column
            )
        )

        if not matches:
            return None

        if len(matches) != 1:
            raise (
                TableEvidenceLookupError(
                    (
                        "Requested table column is "
                        f"ambiguous: "
                        f"{request.column_name}"
                    )
                )
            )

        cell = (
            matches[0]
        )

        if not cell.text.strip():
            return None

        return (
            TableLookupSelection(
                request=request,
                row=row,
                cell=cell,
            )
        )

    @staticmethod
    def _normalize_name(
        value: str,
    ) -> str:

        normalized = (
            unicodedata.normalize(
                "NFKD",
                value,
            )
        )

        without_accents = "".join(
            character
            for character
            in normalized
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
