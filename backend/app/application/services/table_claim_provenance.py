from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from uuid import UUID

from app.application.services.table_evidence_atoms import (
    LabeledRowLookupRequest,
    TableLookupRequest,
    TableLookupSelection,
)
from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
)
from app.domain.models.table import (
    TableSourceSpan,
)


class TableClaimProvenanceError(
    RuntimeError
):
    pass


def _normalize_name(
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


@dataclass(frozen=True)
class TableClaimOrigin:
    """
    Backend-validated provenance for a claim whose
    canonical evidence is a structured table cell.

    This is intentionally separate from chunk-based
    ClaimOrigin. No synthetic chunk IDs or fake
    character offsets are created.
    """

    evidence_id: str

    document_id: UUID
    document_version_id: UUID

    page_id: UUID
    page_number: int

    logical_key: str
    section_key: str | None
    column_name: str

    text: str
    language: str

    value_source: TableCellValueSource

    origin_spans: tuple[
        TableSourceSpan,
        ...,
    ]

    structure_spans: tuple[
        TableSourceSpan,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:

        if not self.evidence_id.strip():
            raise ValueError(
                "evidence_id cannot be empty."
            )

        if self.page_number < 1:
            raise ValueError(
                "page_number must be positive."
            )

        if not self.logical_key.strip():
            raise ValueError(
                "logical_key cannot be empty."
            )

        if not self.column_name.strip():
            raise ValueError(
                "column_name cannot be empty."
            )

        if not self.text.strip():
            raise ValueError(
                "Table claim text cannot be empty."
            )

        if not self.language.strip():
            raise ValueError(
                "Table claim language cannot be empty."
            )

        if not self.origin_spans:
            raise ValueError(
                "Table claim requires source provenance."
            )

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

    @property
    def source_refs(
        self,
    ) -> tuple[str, ...]:

        return tuple(
            span.source_ref
            for span
            in self.origin_spans
        )


@dataclass(frozen=True)
class TableAtomicClaim:
    claim_id: str
    text: str
    origin: TableClaimOrigin
    language: str = "fr"
    required: bool = True

    def __post_init__(
        self,
    ) -> None:

        if not self.claim_id.strip():
            raise ValueError(
                "claim_id cannot be empty."
            )

        if not self.text.strip():
            raise ValueError(
                "Table atomic claim text "
                "cannot be empty."
            )

        if not self.language.strip():
            raise ValueError(
                "Table atomic claim language "
                "cannot be empty."
            )

        if (
            self.text.strip()
            != self.origin.text
        ):
            raise ValueError(
                "Table claim must match "
                "its validated origin."
            )

        if (
            self.language
            .strip()
            .lower()
            != self.origin
            .language
            .strip()
            .lower()
        ):
            raise ValueError(
                "Table claim language must "
                "match its validated origin."
            )


class DirectTableClaimBuilder:
    """
    P0 deterministic table-claim provenance.

    For the first table-Q&A path we accept only
    directly source-backed cells whose evidence
    originates on the same physical page as the
    semantic row.

    Cross-page inherited values remain represented
    correctly by the table subsystem but fail closed
    here until typed multi-page citation support is
    connected.
    """

    def build(
        self,
        *,
        request: (
            TableLookupRequest
            | LabeledRowLookupRequest
        ),
        selection: TableLookupSelection,
        claim_id: str = "table-claim-1",
    ) -> TableAtomicClaim:

        row = selection.row
        cell = selection.cell

        if (
            row.role
            != TableRowRole.DATA
        ):
            raise (
                TableClaimProvenanceError(
                    "Table claim requires "
                    "a DATA row."
                )
            )

        expected_key = (
            request.logical_key
        )

        if (
            row.logical_key
            != expected_key
        ):
            raise (
                TableClaimProvenanceError(
                    "Selected row does not "
                    "match requested logical key."
                )
            )

        if (
            cell.column_name
            is None
        ):
            raise (
                TableClaimProvenanceError(
                    "Selected table cell "
                    "has no column name."
                )
            )

        if (
            _normalize_name(
                cell.column_name
            )
            != _normalize_name(
                request.column_name
            )
        ):
            raise (
                TableClaimProvenanceError(
                    "Selected table cell "
                    "does not match the "
                    "requested column."
                )
            )

        clean_text = (
            cell.text.strip()
        )

        if not clean_text:
            raise (
                TableClaimProvenanceError(
                    "Selected table cell "
                    "is blank."
                )
            )

        # ------------------------------------------------
        # P0 direct citation path.
        #
        # Do not silently convert inherited or derived
        # semantic values into apparently direct source
        # evidence.
        # ------------------------------------------------

        if (
            cell.value_source
            != TableCellValueSource
            .SOURCE_TEXT
        ):
            raise (
                TableClaimProvenanceError(
                    "Selected table value is not "
                    "direct source text."
                )
            )

        if not cell.origin_spans:
            raise (
                TableClaimProvenanceError(
                    "Selected table cell has "
                    "no source provenance."
                )
            )

        origin_pages = {
            span.page_number
            for span
            in cell.origin_spans
        }

        if origin_pages != {
            row.page_number
        }:
            raise (
                TableClaimProvenanceError(
                    "Cross-page table origin "
                    "requires typed multi-page "
                    "citation support."
                )
            )

        structure_pages = {
            span.page_number
            for span
            in cell.structure_spans
        }

        if (
            structure_pages
            and structure_pages
            != {
                row.page_number
            }
        ):
            raise (
                TableClaimProvenanceError(
                    "Table structure provenance "
                    "does not match the semantic "
                    "row page."
                )
            )

        origin = TableClaimOrigin(
            evidence_id=(
                row.evidence_id
            ),
            document_id=(
                row.document_id
            ),
            document_version_id=(
                row.document_version_id
            ),
            page_id=(
                row.page_id
            ),
            page_number=(
                row.page_number
            ),
            logical_key=(
                expected_key
            ),
            section_key=(
                row.section_key
            ),
            column_name=(
                cell.column_name
            ),
            text=clean_text,
            language=(
                row.language
            ),
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

        return TableAtomicClaim(
            claim_id=claim_id,
            text=clean_text,
            origin=origin,
            language=(
                row.language
            ),
            required=True,
        )
