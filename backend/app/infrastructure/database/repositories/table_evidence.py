from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import (
    delete,
    select,
)
from sqlalchemy.orm import Session

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.application.ports.table_evidence_repository import (
    AuthorizedGenericTable,
    AuthorizedLabeledRow,
    AuthorizedTableVocabulary,
    TableEvidenceRepository,
)
from app.application.services.table_evidence_atoms import (
    TableCellEvidenceAtom,
    TableEvidenceAtom,
)
from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    TableBBox,
    TableSourceSpan,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.permissions import (
    DocumentPermission,
)
from app.infrastructure.database.models.table_evidence import (
    TableEvidenceRecord,
)


class TableEvidencePersistenceError(
    RuntimeError
):
    pass


def _serialize_span(
    span: TableSourceSpan,
) -> dict[str, Any]:

    bbox = None

    if span.bbox is not None:
        bbox = {
            "x0": span.bbox.x0,
            "y0": span.bbox.y0,
            "x1": span.bbox.x1,
            "y1": span.bbox.y1,
        }

    return {
        "page_number": span.page_number,
        "kind": span.kind.value,
        "source_ref": span.source_ref,
        "bbox": bbox,
        "text": span.text,
    }


def _deserialize_span(
    payload: dict[str, Any],
) -> TableSourceSpan:

    bbox_payload = payload.get(
        "bbox"
    )

    bbox = None

    if bbox_payload is not None:
        bbox = TableBBox(
            x0=float(
                bbox_payload["x0"]
            ),
            y0=float(
                bbox_payload["y0"]
            ),
            x1=float(
                bbox_payload["x1"]
            ),
            y1=float(
                bbox_payload["y1"]
            ),
        )

    return TableSourceSpan(
        page_number=int(
            payload["page_number"]
        ),
        kind=TableSpanKind(
            payload["kind"]
        ),
        source_ref=str(
            payload["source_ref"]
        ),
        bbox=bbox,
        text=payload.get(
            "text"
        ),
    )


def _serialize_cell(
    cell: TableCellEvidenceAtom,
) -> dict[str, Any]:

    return {
        "column_start": (
            cell.column_start
        ),
        "column_end": (
            cell.column_end
        ),
        "column_name": (
            cell.column_name
        ),
        "text": cell.text,
        "value_source": (
            cell.value_source.value
        ),
        "origin_spans": [
            _serialize_span(span)
            for span
            in cell.origin_spans
        ],
        "structure_spans": [
            _serialize_span(span)
            for span
            in cell.structure_spans
        ],
    }


def _deserialize_cell(
    payload: dict[str, Any],
) -> TableCellEvidenceAtom:

    return TableCellEvidenceAtom(
        column_start=int(
            payload["column_start"]
        ),
        column_end=int(
            payload["column_end"]
        ),
        column_name=(
            None
            if payload.get(
                "column_name"
            )
            is None
            else str(
                payload[
                    "column_name"
                ]
            )
        ),
        text=str(
            payload["text"]
        ),
        value_source=(
            TableCellValueSource(
                payload[
                    "value_source"
                ]
            )
        ),
        origin_spans=tuple(
            _deserialize_span(
                item
            )
            for item
            in payload.get(
                "origin_spans",
                [],
            )
        ),
        structure_spans=tuple(
            _deserialize_span(
                item
            )
            for item
            in payload.get(
                "structure_spans",
                [],
            )
        ),
    )


def _record_to_atom(
    record: TableEvidenceRecord,
) -> TableEvidenceAtom:

    try:
        atom = TableEvidenceAtom(
            evidence_id=(
                record.evidence_id
            ),
            document_id=(
                record.document_id
            ),
            document_version_id=(
                record
                .document_version_id
            ),
            page_id=(
                record.page_id
            ),
            page_number=(
                record.page_number
            ),
            source_table_refs=tuple(
                str(value)
                for value
                in (
                    record
                    .source_table_refs_json
                )
            ),
            row_index=(
                record.row_index
            ),
            role=TableRowRole(
                record.role
            ),
            logical_key=(
                record.logical_key
            ),
            section_key=(
                record.section_key
            ),
            column_names=tuple(
                str(value)
                for value
                in (
                    record
                    .column_names_json
                )
            ),
            cells=tuple(
                _deserialize_cell(
                    item
                )
                for item
                in record.cells_json
            ),
            language=(
                record.language
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise (
            TableEvidencePersistenceError(
                (
                    "Persisted table evidence "
                    "is malformed."
                )
            )
        ) from exc

    if (
        atom.search_text
        != record.search_text
    ):
        raise (
            TableEvidencePersistenceError(
                (
                    "Persisted table search text "
                    "does not match canonical "
                    "table evidence."
                )
            )
        )

    return atom


class PostgresTableEvidenceRepository(
    TableEvidenceRepository
):
    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

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

        items = tuple(
            atoms
        )

        try:
            document_id = (
                self._db.scalar(
                    select(
                        DocumentVersion
                        .document_id
                    )
                    .where(
                        DocumentVersion.id
                        == version_id
                    )
                )
            )

            if document_id is None:
                raise ValueError(
                    (
                        "Document version does "
                        "not exist."
                    )
                )

            if any(
                atom.document_version_id
                != version_id
                for atom in items
            ):
                raise ValueError(
                    (
                        "Table evidence contains "
                        "an atom outside the "
                        "requested version."
                    )
                )

            if any(
                atom.document_id
                != document_id
                for atom in items
            ):
                raise ValueError(
                    (
                        "Table evidence document "
                        "does not match the "
                        "document version."
                    )
                )

            page_ids = {
                atom.page_id
                for atom in items
            }

            page_map = {}

            if page_ids:
                page_rows = (
                    self._db.execute(
                        select(
                            DocumentPage.id,
                            DocumentPage
                            .page_number,
                        )
                        .where(
                            DocumentPage.id.in_(
                                page_ids
                            ),
                            DocumentPage
                            .document_version_id
                            == version_id,
                        )
                    )
                    .all()
                )

                page_map = {
                    page_id: page_number
                    for (
                        page_id,
                        page_number,
                    )
                    in page_rows
                }

                if (
                    set(page_map)
                    != page_ids
                ):
                    raise ValueError(
                        (
                            "Table evidence "
                            "references a page "
                            "outside the requested "
                            "document version."
                        )
                    )

                for atom in items:

                    if (
                        page_map[
                            atom.page_id
                        ]
                        != atom.page_number
                    ):
                        raise ValueError(
                            (
                                "Table evidence "
                                "page number does "
                                "not match its "
                                "canonical page_id."
                            )
                        )

            self._db.execute(
                delete(
                    TableEvidenceRecord
                )
                .where(
                    TableEvidenceRecord
                    .document_version_id
                    == version_id
                )
            )

            for atom in items:
                self._db.add(
                    TableEvidenceRecord(
                        evidence_id=(
                            atom.evidence_id
                        ),
                        document_id=(
                            atom.document_id
                        ),
                        document_version_id=(
                            atom
                            .document_version_id
                        ),
                        page_id=(
                            atom.page_id
                        ),
                        page_number=(
                            atom.page_number
                        ),
                        row_index=(
                            atom.row_index
                        ),
                        role=(
                            atom.role.value
                        ),
                        logical_key=(
                            atom.logical_key
                        ),
                        section_key=(
                            atom.section_key
                        ),
                        language=(
                            atom.language
                        ),
                        source_table_refs_json=list(
                            atom
                            .source_table_refs
                        ),
                        column_names_json=list(
                            atom.column_names
                        ),
                        cells_json=[
                            _serialize_cell(
                                cell
                            )
                            for cell
                            in atom.cells
                        ],
                        search_text=(
                            atom.search_text
                        ),
                    )
                )

            if commit:
                self._db.commit()

        except Exception:
            self._db.rollback()
            raise

    def list_authorized_vocabulary(
        self,
        *,
        scope: AuthorizationScope,
    ) -> AuthorizedTableVocabulary:

        # --------------------------------------------
        # Empty authorization => no corpus query.
        # --------------------------------------------

        if not scope.document_ids:
            return (
                AuthorizedTableVocabulary
                .empty()
            )

        # --------------------------------------------
        # Authorization is applied INSIDE the query.
        #
        # Every scope is constrained to its authorized
        # document IDs and ACTIVE documents.
        #
        # ACL-backed scopes additionally re-check the
        # current DocumentPermission so stale or
        # poisoned scopes cannot disclose structure.
        #
        # ADMIN scopes are produced from the current
        # active-document set and therefore do not
        # require per-document ACL rows.
        # --------------------------------------------

        statement = (
            select(
                TableEvidenceRecord
                .section_key,
                TableEvidenceRecord
                .logical_key,
                TableEvidenceRecord
                .column_names_json,
                TableEvidenceRecord
                .cells_json,
            )
            .join(
                Document,
                Document.id
                == (
                    TableEvidenceRecord
                    .document_id
                ),
            )
        )

        if (
            scope
            .requires_document_permission
        ):
            statement = (
                statement
                .join(
                    DocumentPermission,
                    DocumentPermission
                    .document_id
                    == (
                        TableEvidenceRecord
                        .document_id
                    ),
                )
                .where(
                    DocumentPermission
                    .user_id
                    == scope.user_id,
                    DocumentPermission
                    .action
                    == scope.action.value,
                )
            )

        statement = statement.where(
            TableEvidenceRecord
            .document_id.in_(
                scope.document_ids
            ),
            Document.status
            == "ACTIVE",
        )

        rows = tuple(
            self._db.execute(
                statement
            )
            .all()
        )

        section_keys: set[str] = set()

        # Existing procurement/global columns remain
        # isolated from generic-table columns.
        column_names: set[str] = set()

        generic: dict[
            str,
            dict[str, object],
        ] = {}

        try:

            for (
                section_key,
                logical_key,
                raw_column_names,
                raw_cells,
            ) in rows:

                # ------------------------------------
                # Existing section vocabulary.
                # ------------------------------------

                if section_key is not None:

                    if (
                        not isinstance(
                            section_key,
                            str,
                        )
                        or not section_key.strip()
                        or section_key
                        != section_key.strip()
                    ):
                        raise ValueError(
                            "Invalid persisted "
                            "table section key."
                        )

                    section_keys.add(
                        section_key
                    )

                # ------------------------------------
                # Column schema.
                # ------------------------------------

                if not isinstance(
                    raw_column_names,
                    list,
                ):
                    raise ValueError(
                        "Invalid persisted "
                        "table column names."
                    )

                clean_columns: list[str] = []

                for value in raw_column_names:

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
                            "Invalid persisted "
                            "table column name."
                        )

                    clean_columns.append(
                        value
                    )

                is_generic = (
                    isinstance(
                        logical_key,
                        str,
                    )
                    and logical_key.startswith(
                        "GENERIC/"
                    )
                )

                # ------------------------------------
                # Existing procurement vocabulary
                # remains globally shaped exactly as
                # before.
                # ------------------------------------

                if not is_generic:

                    column_names.update(
                        clean_columns
                    )

                    continue

                # ------------------------------------
                # Generic table identity.
                # ------------------------------------

                logical_parts = (
                    logical_key.split("/")
                )

                if (
                    len(logical_parts) != 3
                    or logical_parts[0]
                    != "GENERIC"
                    or not logical_parts[1]
                    or not logical_parts[2]
                ):
                    raise ValueError(
                        "Invalid persisted "
                        "generic logical key."
                    )

                table_key = (
                    "/".join(
                        logical_parts[:2]
                    )
                )

                if not clean_columns:
                    raise ValueError(
                        "Generic table has "
                        "no column schema."
                    )

                # ------------------------------------
                # Only the qualified first-column
                # label becomes structural vocabulary.
                #
                # Other cell values are never copied
                # into the vocabulary model.
                # ------------------------------------

                if not isinstance(
                    raw_cells,
                    list,
                ):
                    raise ValueError(
                        "Invalid persisted "
                        "generic table cells."
                    )

                if any(
                    not isinstance(
                        item,
                        dict,
                    )
                    for item in raw_cells
                ):
                    raise ValueError(
                        "Invalid persisted "
                        "generic table cell."
                    )

                label_cells = tuple(
                    item
                    for item in raw_cells
                    if (
                        item.get(
                            "column_start"
                        )
                        == 0
                        and item.get(
                            "column_end"
                        )
                        == 1
                    )
                )

                if len(label_cells) != 1:
                    raise ValueError(
                        "Generic table row "
                        "must contain exactly "
                        "one first-column label."
                    )

                label = (
                    label_cells[0]
                    .get("text")
                )

                if (
                    not isinstance(
                        label,
                        str,
                    )
                    or not label.strip()
                    or label
                    != label.strip()
                ):
                    raise ValueError(
                        "Invalid persisted "
                        "generic row label."
                    )

                row_descriptor = (
                    AuthorizedLabeledRow(
                        label=label,
                        logical_key=(
                            logical_key
                        ),
                    )
                )

                existing = generic.get(
                    table_key
                )

                if existing is None:

                    generic[
                        table_key
                    ] = {
                        "column_names": (
                            tuple(
                                clean_columns
                            )
                        ),
                        "rows": [
                            row_descriptor
                        ],
                    }

                    continue

                existing_columns = (
                    existing[
                        "column_names"
                    ]
                )

                if (
                    existing_columns
                    != tuple(
                        clean_columns
                    )
                ):
                    raise ValueError(
                        "Generic table rows "
                        "disagree on column schema."
                    )

                existing_rows = (
                    existing["rows"]
                )

                if not isinstance(
                    existing_rows,
                    list,
                ):
                    raise ValueError(
                        "Invalid generic table "
                        "vocabulary accumulator."
                    )

                existing_rows.append(
                    row_descriptor
                )

            generic_tables = []

            for table_key in sorted(
                generic,
                key=lambda value: (
                    value.casefold(),
                    value,
                ),
            ):

                entry = generic[
                    table_key
                ]

                rows_value = entry[
                    "rows"
                ]

                columns_value = entry[
                    "column_names"
                ]

                if (
                    not isinstance(
                        rows_value,
                        list,
                    )
                    or not isinstance(
                        columns_value,
                        tuple,
                    )
                ):
                    raise ValueError(
                        "Invalid generic table "
                        "vocabulary accumulator."
                    )

                generic_tables.append(
                    AuthorizedGenericTable(
                        table_key=(
                            table_key
                        ),
                        rows=tuple(
                            sorted(
                                rows_value,
                                key=lambda row: (
                                    row.label
                                    .casefold(),
                                    row.label,
                                    row.logical_key,
                                ),
                            )
                        ),
                        column_names=(
                            columns_value
                        ),
                    )
                )

            result = (
                AuthorizedTableVocabulary(
                    section_keys=tuple(
                        sorted(
                            section_keys,
                            key=lambda value: (
                                value.casefold(),
                                value,
                            ),
                        )
                    ),
                    column_names=tuple(
                        sorted(
                            column_names,
                            key=lambda value: (
                                value.casefold(),
                                value,
                            ),
                        )
                    ),
                    generic_tables=tuple(
                        generic_tables
                    ),
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            self._db.rollback()

            raise (
                TableEvidencePersistenceError(
                    "Persisted table vocabulary "
                    "is malformed."
                )
            ) from exc

        return result


    def find_authorized_by_evidence_id(
        self,
        *,
        scope: AuthorizationScope,
        evidence_id: str,
    ) -> TableEvidenceAtom | None:

        cleaned = (
            evidence_id.strip()
        )

        if not cleaned:
            raise ValueError(
                "evidence_id cannot be empty."
            )

        if not scope.document_ids:
            return None

        records = tuple(
            self._db.scalars(
                select(
                    TableEvidenceRecord
                )
                .join(
                    Document,
                    Document.id
                    == (
                        TableEvidenceRecord
                        .document_id
                    ),
                )
                .where(
                    TableEvidenceRecord
                    .document_id.in_(
                        scope.document_ids
                    ),
                    TableEvidenceRecord
                    .evidence_id
                    == cleaned,
                    Document.status
                    == "ACTIVE",
                )
            )
        )

        if not records:
            return None

        if len(records) != 1:
            self._db.rollback()

            raise (
                TableEvidencePersistenceError(
                    "Canonical table evidence "
                    "is ambiguous."
                )
            )

        try:
            return _record_to_atom(
                records[0]
            )

        except Exception:
            self._db.rollback()
            raise

    def find_authorized_by_logical_key(
        self,
        *,
        scope: AuthorizationScope,
        logical_key: str,
    ) -> tuple[
        TableEvidenceAtom,
        ...,
    ]:

        cleaned = (
            logical_key.strip()
        )

        if not cleaned:
            raise ValueError(
                (
                    "logical_key cannot "
                    "be empty."
                )
            )

        if not scope.document_ids:
            return ()

        records = tuple(
            self._db.scalars(
                select(
                    TableEvidenceRecord
                )
                .join(
                    Document,
                    Document.id
                    == (
                        TableEvidenceRecord
                        .document_id
                    ),
                )
                .where(
                    TableEvidenceRecord
                    .document_id.in_(
                        scope.document_ids
                    ),
                    TableEvidenceRecord
                    .logical_key
                    == cleaned,
                    Document.status
                    == "ACTIVE",
                )
                .order_by(
                    TableEvidenceRecord
                    .document_id,
                    TableEvidenceRecord
                    .page_number,
                    TableEvidenceRecord
                    .row_index,
                )
            )
        )

        try:
            return tuple(
                _record_to_atom(
                    record
                )
                for record
                in records
            )

        except Exception:
            # End any read transaction after
            # malformed persistence is detected.
            self._db.rollback()
            raise
