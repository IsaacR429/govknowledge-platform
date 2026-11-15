from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.application.ports.table_evidence_repository import (
    TableEvidenceRepository,
)
from app.application.services.table_evidence_atoms import (
    TableEvidenceAtom,
)


class TableEvidenceIndexingError(
    ValueError
):
    pass


@dataclass(frozen=True)
class TableEvidenceIndexingResult:
    version_id: UUID
    indexed_count: int


class TableEvidenceIndexingService:
    """Persist qualified typed table evidence.

    This service is used by the trusted document-processing
    pipeline, not directly by end users.
    """

    def __init__(
        self,
        *,
        repository: TableEvidenceRepository,
    ) -> None:
        self._repository = repository

    def index_version(
        self,
        *,
        version_id: UUID,
        atoms: tuple[
            TableEvidenceAtom,
            ...,
        ],
    ) -> TableEvidenceIndexingResult:

        items = tuple(
            atoms
        )

        evidence_ids = tuple(
            item.evidence_id
            for item in items
        )

        if (
            len(set(evidence_ids))
            != len(evidence_ids)
        ):
            raise (
                TableEvidenceIndexingError(
                    "Duplicate table evidence IDs."
                )
            )

        wrong_versions = tuple(
            item
            for item in items
            if (
                item.document_version_id
                != version_id
            )
        )

        if wrong_versions:
            raise (
                TableEvidenceIndexingError(
                    (
                        "Table evidence contains an atom "
                        "outside the requested document version."
                    )
                )
            )

        document_ids = {
            item.document_id
            for item in items
        }

        if len(document_ids) > 1:
            raise (
                TableEvidenceIndexingError(
                    (
                        "A document version cannot contain "
                        "table evidence from multiple documents."
                    )
                )
            )

        # Deliberately call even for an empty tuple.
        #
        # An empty qualified result must be capable of clearing
        # stale evidence left by a previous processing run.
        self._repository.replace_version_atoms(
            version_id=version_id,
            atoms=items,
        )

        return (
            TableEvidenceIndexingResult(
                version_id=version_id,
                indexed_count=len(
                    items
                ),
            )
        )
