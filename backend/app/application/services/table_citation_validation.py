from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from uuid import UUID

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.table_evidence_repository import (
    TableEvidenceRepository,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.table_claim_provenance import (
    TableAtomicClaim,
)
from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
)
from app.domain.models.table import (
    TableSourceSpan,
)


class InvalidTableCitationError(
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
class TableCitation:
    """
    Canonical typed citation for structured table evidence.

    Deliberately does NOT contain chunk_id or character offsets.
    Those concepts belong to text-chunk citations and must not be
    fabricated for table evidence.
    """

    evidence_id: str

    document_id: UUID
    document_version_id: UUID

    page_id: UUID
    page_number: int

    logical_key: str
    section_key: str | None
    column_name: str

    cited_text: str
    language: str

    origin_spans: tuple[
        TableSourceSpan,
        ...,
    ]

    structure_spans: tuple[
        TableSourceSpan,
        ...,
    ]

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


class TableClaimCitationValidator:
    """
    Revalidate table provenance against canonical persistence.

    Sequence:

        identity
          ↓
        authorization scope
          ↓
        SQL-scoped evidence_id lookup
          ↓
        compare persisted cell to claim provenance
          ↓
        exact document READ reauthorization
          ↓
        typed canonical citation
    """

    def __init__(
        self,
        *,
        authorization: AuthorizationService,
        repository: TableEvidenceRepository,
    ) -> None:

        self._authorization = (
            authorization
        )

        self._repository = (
            repository
        )

    @staticmethod
    def _invalid() -> None:
        # Keep the external failure deliberately generic.
        # A citation-validation failure must not reveal
        # whether inaccessible evidence exists.
        raise InvalidTableCitationError(
            "Table citation source is invalid."
        )

    def validate(
        self,
        *,
        identity: CurrentIdentity,
        claim: TableAtomicClaim,
    ) -> TableCitation:

        origin = claim.origin

        # ------------------------------------------------
        # Authorization BEFORE persistence retrieval.
        # ------------------------------------------------

        scope = (
            self._authorization
            .retrieval_scope(
                identity
            )
        )

        if not scope.document_ids:
            self._invalid()

        row = (
            self._repository
            .find_authorized_by_evidence_id(
                scope=scope,
                evidence_id=(
                    origin.evidence_id
                ),
            )
        )

        if row is None:
            self._invalid()

        # ------------------------------------------------
        # Canonical row identity.
        # ------------------------------------------------

        if (
            row.document_id
            != origin.document_id
        ):
            self._invalid()

        if (
            row.document_version_id
            != origin.document_version_id
        ):
            self._invalid()

        if (
            row.page_id
            != origin.page_id
        ):
            self._invalid()

        if (
            row.page_number
            != origin.page_number
        ):
            self._invalid()

        if (
            row.role
            != TableRowRole.DATA
        ):
            self._invalid()

        if (
            row.logical_key
            != origin.logical_key
        ):
            self._invalid()

        if (
            row.section_key
            != origin.section_key
        ):
            self._invalid()

        if (
            row.language
            .strip()
            .lower()
            != origin.language
            .strip()
            .lower()
        ):
            self._invalid()

        # ------------------------------------------------
        # Re-select the canonical persisted column.
        # ------------------------------------------------

        requested_column = (
            _normalize_name(
                origin.column_name
            )
        )

        matching_cells = tuple(
            cell
            for cell in row.cells
            if (
                cell.column_name
                is not None
                and _normalize_name(
                    cell.column_name
                )
                == requested_column
            )
        )

        if len(matching_cells) != 1:
            self._invalid()

        cell = matching_cells[0]

        # ------------------------------------------------
        # P0 citation policy:
        # direct source text only.
        # ------------------------------------------------

        if (
            cell.value_source
            != TableCellValueSource
            .SOURCE_TEXT
        ):
            self._invalid()

        if (
            origin.value_source
            != cell.value_source
        ):
            self._invalid()

        if (
            cell.text.strip()
            != claim.text.strip()
        ):
            self._invalid()

        if (
            cell.text.strip()
            != origin.text
        ):
            self._invalid()

        if (
            cell.origin_spans
            != origin.origin_spans
        ):
            self._invalid()

        if (
            cell.structure_spans
            != origin.structure_spans
        ):
            self._invalid()

        if not cell.origin_spans:
            self._invalid()

        origin_pages = {
            span.page_number
            for span
            in cell.origin_spans
        }

        if origin_pages != {
            row.page_number
        }:
            self._invalid()

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
            self._invalid()

        # ------------------------------------------------
        # Final exact authorization check immediately
        # before disclosure.
        # ------------------------------------------------

        self._authorization.require_document_read(
            identity,
            row.document_id,
        )

        return TableCitation(
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
                row.logical_key
            ),
            section_key=(
                row.section_key
            ),
            column_name=(
                cell.column_name
            ),
            cited_text=(
                cell.text.strip()
            ),
            language=(
                row.language
            ),
            origin_spans=(
                cell.origin_spans
            ),
            structure_spans=(
                cell.structure_spans
            ),
        )
