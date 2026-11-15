from __future__ import annotations

import os
import re
from uuid import uuid4

from sqlalchemy import (
    delete,
    func,
    select,
    text,
)
from sqlalchemy.engine import make_url

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.authorized_table_evidence import (
    AuthorizedTableEvidenceService,
)
from app.application.services.table_evidence_atoms import (
    TableCellEvidenceAtom,
    TableEvidenceAtom,
    TableEvidenceLookupService,
    TableLookupRequest,
)
from app.application.services.table_evidence_indexing import (
    TableEvidenceIndexingService,
)
from app.core.config import (
    get_settings,
)
from app.domain.enums.permission import (
    PermissionAction,
)
from app.domain.enums.processing import (
    ProcessingStatus,
)
from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    TableSourceSpan,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.identity import (
    User,
)
from app.infrastructure.database.models.permissions import (
    DocumentPermission,
)
from app.infrastructure.database.models.table_evidence import (
    TableEvidenceRecord,
)
from app.infrastructure.database.repositories.authorization import (
    PostgresAuthorizationRepository,
)
from app.infrastructure.database.repositories.table_evidence import (
    PostgresTableEvidenceRepository,
)
from app.infrastructure.database.session import (
    SessionLocal,
    engine,
)


EXPECTED_SYSTEM_ID = "7686228710959620134"

LOGICAL_KEY = "LOT_2/05"

AUTHORIZED_SUPPLIER = (
    "ENTREPRISE ZERBO AWA "
    "ET FRERES (EZAF)"
)

SECRET_SUPPLIER = (
    "SECRET SUPPLIER - "
    "MUST NEVER LEAK"
)


def _assert_isolated_qa() -> None:
    expected_system_id = os.environ.get(
        "GOVKNOWLEDGE_QA_SYSTEM_ID",
        "",
    )

    if (
        expected_system_id
        != EXPECTED_SYSTEM_ID
    ):
        raise RuntimeError(
            "STOP: QA system identifier "
            "guard is missing."
        )

    raw_url = os.environ.get(
        "DATABASE_URL",
        "",
    )

    expected = make_url(
        raw_url
    )

    actual = make_url(
        get_settings().database_url
    )

    if not (
        expected.drivername
        == "postgresql+psycopg"
        and expected.host
        == "127.0.0.1"
        and expected.port
        == 15432
        and expected.database
        == "govknowledge_qa"
        and expected.username
        == "govknowledge_qa"
        and bool(expected.password)
        and actual == expected
        and engine.url == expected
        and SessionLocal.kw.get(
            "bind"
        )
        is engine
    ):
        raise RuntimeError(
            "STOP: test is not bound "
            "to isolated QA PostgreSQL."
        )

    with engine.connect() as connection:
        connection.execute(
            text(
                "SET TRANSACTION READ ONLY"
            )
        )

        (
            database,
            username,
            system_id,
        ) = (
            connection.execute(
                text(
                    "SELECT "
                    "current_database(), "
                    "current_user, "
                    "system_identifier::text "
                    "FROM pg_control_system()"
                )
            )
            .one()
        )

        connection.rollback()

    if (
        database,
        username,
        system_id,
    ) != (
        "govknowledge_qa",
        "govknowledge_qa",
        EXPECTED_SYSTEM_ID,
    ):
        raise RuntimeError(
            "STOP: actual PostgreSQL "
            "instance is not QA."
        )


def _cell(
    *,
    column_index: int,
    column_name: str,
    value: str,
    page_number: int,
    source_label: str,
) -> TableCellEvidenceAtom:

    return TableCellEvidenceAtom(
        column_start=column_index,
        column_end=(
            column_index + 1
        ),
        column_name=column_name,
        text=value,
        value_source=(
            TableCellValueSource
            .SOURCE_TEXT
        ),
        origin_spans=(
            TableSourceSpan(
                page_number=page_number,
                kind=TableSpanKind.TEXT,
                source_ref=(
                    f"{source_label}:"
                    f"{column_index}"
                ),
                bbox=None,
                text=value,
            ),
        ),
        structure_spans=(
            TableSourceSpan(
                page_number=page_number,
                kind=(
                    TableSpanKind
                    .STRUCTURE
                ),
                source_ref=(
                    f"{source_label}:"
                    "structure"
                ),
                bbox=None,
                text=None,
            ),
        ),
    )


def _atom(
    *,
    document_id,
    version_id,
    page_id,
    supplier: str,
    source_label: str,
) -> TableEvidenceAtom:

    column_names = (
        "N°",
        "Soumissionnaires",
        "Montants lus en FCFA",
        "Montants corrigés en FCFA",
        "Observations",
    )

    values = (
        "05",
        supplier,
        "3972300 HTVA",
        "4417530 TTC",
        "Conforme",
    )

    cells = tuple(
        _cell(
            column_index=index,
            column_name=column_name,
            value=value,
            page_number=4,
            source_label=(
                source_label
            ),
        )
        for index, (
            column_name,
            value,
        )
        in enumerate(
            zip(
                column_names,
                values,
                strict=True,
            )
        )
    )

    return TableEvidenceAtom(
        evidence_id=(
            f"table:{version_id}:"
            f"{LOGICAL_KEY}"
        ),
        document_id=document_id,
        document_version_id=(
            version_id
        ),
        page_id=page_id,
        page_number=4,
        source_table_refs=(
            source_label,
        ),
        row_index=5,
        role=TableRowRole.DATA,
        logical_key=LOGICAL_KEY,
        section_key="LOT_2",
        column_names=(
            column_names
        ),
        cells=cells,
        language="fr",
    )


def _document(
    *,
    document_id,
    title,
    confidentiality,
):
    return Document(
        id=document_id,
        title=title,
        description=(
            "Typed table evidence "
            "ACL qualification fixture."
        ),
        document_type="PROCUREMENT",
        issuing_organization=(
            "GovKnowledge QA"
        ),
        jurisdiction=(
            "Burkina Faso"
        ),
        language="fr",
        publication_date=None,
        source_url=None,
        status="ACTIVE",
        confidentiality_level=(
            confidentiality
        ),
    )


def _version(
    *,
    version_id,
    document_id,
    filename,
    file_hash,
):
    return DocumentVersion(
        id=version_id,
        document_id=document_id,
        version_number=1,
        original_filename=filename,
        storage_path=(
            f"qa-table-acl/"
            f"{filename}"
        ),
        file_hash=file_hash,
        processing_status=(
            ProcessingStatus
            .SUCCEEDED
            .value
        ),
        effective_date=None,
        superseded_at=None,
    )


def _page(
    *,
    page_id,
    version_id,
    text_value,
):
    return DocumentPage(
        id=page_id,
        document_version_id=(
            version_id
        ),
        page_number=4,
        extracted_text=(
            text_value
        ),
        metadata_json={},
    )


def test_real_postgres_table_evidence_acl_isolation():
    _assert_isolated_qa()

    db = SessionLocal()

    user_id = uuid4()

    allowed_document_id = uuid4()
    allowed_version_id = uuid4()
    allowed_page_id = uuid4()

    secret_document_id = uuid4()
    secret_version_id = uuid4()
    secret_page_id = uuid4()

    permission_id = uuid4()

    username = (
        "table-acl-"
        f"{user_id}"
    )

    try:
        # ====================================================
        # REAL DATABASE IDENTITY
        # ====================================================

        db.add(
            User(
                id=user_id,
                username=username,
                display_name=(
                    "Table ACL "
                    "Qualification User"
                ),
                email=None,
                department="QA",
                language_preference="fr",
                is_active=True,
            )
        )

        # ====================================================
        # AUTHORIZED DOCUMENT
        # ====================================================

        db.add(
            _document(
                document_id=(
                    allowed_document_id
                ),
                title=(
                    "Authorized "
                    "Procurement Table"
                ),
                confidentiality="PUBLIC",
            )
        )

        db.add(
            _version(
                version_id=(
                    allowed_version_id
                ),
                document_id=(
                    allowed_document_id
                ),
                filename=(
                    "authorized-table.pdf"
                ),
                file_hash=(
                    "a" * 64
                ),
            )
        )

        db.add(
            _page(
                page_id=(
                    allowed_page_id
                ),
                version_id=(
                    allowed_version_id
                ),
                text_value=(
                    "Authorized procurement "
                    "table page."
                ),
            )
        )

        # ====================================================
        # UNAUTHORIZED DOCUMENT
        #
        # Same exact logical key, but deliberately
        # sensitive supplier value.
        # ====================================================

        db.add(
            _document(
                document_id=(
                    secret_document_id
                ),
                title=(
                    "Unauthorized "
                    "Procurement Table"
                ),
                confidentiality="SECRET",
            )
        )

        db.add(
            _version(
                version_id=(
                    secret_version_id
                ),
                document_id=(
                    secret_document_id
                ),
                filename=(
                    "secret-table.pdf"
                ),
                file_hash=(
                    "b" * 64
                ),
            )
        )

        db.add(
            _page(
                page_id=(
                    secret_page_id
                ),
                version_id=(
                    secret_version_id
                ),
                text_value=(
                    "Secret procurement "
                    "table page."
                ),
            )
        )

        # ====================================================
        # USER GETS READ ONLY ON AUTHORIZED DOCUMENT
        # ====================================================

        db.add(
            DocumentPermission(
                id=permission_id,
                user_id=user_id,
                document_id=(
                    allowed_document_id
                ),
                action=(
                    PermissionAction
                    .READ
                    .value
                ),
            )
        )

        db.commit()

        # ====================================================
        # REAL TABLE-EVIDENCE PERSISTENCE
        # ====================================================

        table_repository = (
            PostgresTableEvidenceRepository(
                db
            )
        )

        indexer = (
            TableEvidenceIndexingService(
                repository=(
                    table_repository
                )
            )
        )

        allowed_atom = _atom(
            document_id=(
                allowed_document_id
            ),
            version_id=(
                allowed_version_id
            ),
            page_id=(
                allowed_page_id
            ),
            supplier=(
                AUTHORIZED_SUPPLIER
            ),
            source_label=(
                "authorized-table"
            ),
        )

        secret_atom = _atom(
            document_id=(
                secret_document_id
            ),
            version_id=(
                secret_version_id
            ),
            page_id=(
                secret_page_id
            ),
            supplier=(
                SECRET_SUPPLIER
            ),
            source_label=(
                "secret-table"
            ),
        )

        first_index = (
            indexer.index_version(
                version_id=(
                    allowed_version_id
                ),
                atoms=(
                    allowed_atom,
                ),
            )
        )

        second_index = (
            indexer.index_version(
                version_id=(
                    secret_version_id
                ),
                atoms=(
                    secret_atom,
                ),
            )
        )

        assert (
            first_index.indexed_count
            == 1
        )

        assert (
            second_index.indexed_count
            == 1
        )

        # ====================================================
        # PROVE BOTH COMPETING ROWS REALLY EXIST
        # ====================================================

        raw_rows = tuple(
            db.scalars(
                select(
                    TableEvidenceRecord
                )
                .where(
                    TableEvidenceRecord
                    .logical_key
                    == LOGICAL_KEY,
                    TableEvidenceRecord
                    .document_id.in_(
                        {
                            allowed_document_id,
                            secret_document_id,
                        }
                    ),
                )
            )
        )

        assert len(raw_rows) == 2

        raw_documents = {
            row.document_id
            for row in raw_rows
        }

        assert raw_documents == {
            allowed_document_id,
            secret_document_id,
        }

        raw_search_text = "\n".join(
            row.search_text
            for row in raw_rows
        )

        assert (
            "EZAF"
            in raw_search_text
        )

        assert (
            "SECRET SUPPLIER"
            in raw_search_text
        )

        print()
        print(
            "RAW DATABASE ROWS:",
            len(raw_rows),
        )

        print(
            "RAW DATABASE CONTAINS "
            "AUTHORIZED + SECRET: PASS"
        )

        # ====================================================
        # REAL AUTHORIZATION SCOPE
        # ====================================================

        identity = CurrentIdentity(
            user_id=user_id,
            username=username,
            display_name=(
                "Table ACL "
                "Qualification User"
            ),
            email=None,
            department="QA",
            language_preference="fr",
            roles=(
                "EMPLOYEE",
            ),
        )

        authorization = (
            AuthorizationService(
                PostgresAuthorizationRepository(
                    db
                )
            )
        )

        scope = (
            authorization
            .retrieval_scope(
                identity
            )
        )

        assert (
            scope.document_ids
            == frozenset(
                {
                    allowed_document_id,
                }
            )
        )

        assert scope.allows(
            allowed_document_id
        )

        assert not scope.allows(
            secret_document_id
        )

        print(
            "AUTHORIZATION SCOPE:",
            len(
                scope.document_ids
            ),
            "document",
        )

        print(
            "SECRET DOCUMENT EXCLUDED "
            "FROM SCOPE: PASS"
        )

        # ====================================================
        # CALL REAL POSTGRES TABLE REPOSITORY DIRECTLY
        #
        # This is the key SQL-boundary proof.
        # ====================================================

        scoped_rows = (
            table_repository
            .find_authorized_by_logical_key(
                scope=scope,
                logical_key=LOGICAL_KEY,
            )
        )

        assert (
            len(scoped_rows)
            == 1
        )

        assert (
            scoped_rows[0]
            .document_id
            == allowed_document_id
        )

        assert (
            scoped_rows[0]
            .document_version_id
            == allowed_version_id
        )

        assert (
            scoped_rows[0]
            .page_id
            == allowed_page_id
        )

        assert (
            SECRET_SUPPLIER
            not in scoped_rows[0]
            .search_text
        )

        assert (
            "EZAF"
            in scoped_rows[0]
            .search_text
        )

        print(
            "SQL-SCOPED ROWS:",
            len(scoped_rows),
        )

        print(
            "UNAUTHORIZED TABLE ROW "
            "NEVER RETURNED BY REPOSITORY: PASS"
        )

        # ====================================================
        # FULL AUTHORIZATION-FIRST SERVICE
        # ====================================================

        service = (
            AuthorizedTableEvidenceService(
                authorization=(
                    authorization
                ),
                repository=(
                    table_repository
                ),
                lookup=(
                    TableEvidenceLookupService()
                ),
            )
        )

        selection = service.lookup(
            identity=identity,
            request=TableLookupRequest(
                section_key="LOT_2",
                ordinal=5,
                column_name=(
                    "Soumissionnaires"
                ),
            ),
        )

        assert (
            selection
            is not None
        )

        assert (
            selection.row
            .logical_key
            == LOGICAL_KEY
        )

        assert (
            selection.row
            .document_id
            == allowed_document_id
        )

        assert (
            selection.row
            .page_number
            == 4
        )

        assert (
            selection.cell.text
            == AUTHORIZED_SUPPLIER
        )

        assert (
            SECRET_SUPPLIER
            not in selection.cell.text
        )

        assert (
            selection.cell
            .origin_pages
            == (4,)
        )

        assert (
            selection.cell
            .structure_pages
            == (4,)
        )

        print()
        print(
            "SELECTED KEY:",
            selection.row.logical_key,
        )

        print(
            "SELECTED SUPPLIER:",
            selection.cell.text,
        )

        print(
            "SELECTED PAGE:",
            selection.row.page_number,
        )

        print(
            "ORIGIN PAGES:",
            selection.cell.origin_pages,
        )

        print(
            "STRUCTURE PAGES:",
            selection.cell.structure_pages,
        )

        # ====================================================
        # SECOND USER WITH NO READ PERMISSION
        # ====================================================

        no_access_identity = (
            CurrentIdentity(
                user_id=uuid4(),
                username=(
                    "no-access-"
                    f"{uuid4()}"
                ),
                display_name=(
                    "No Access User"
                ),
                email=None,
                department="QA",
                language_preference="fr",
                roles=(
                    "EMPLOYEE",
                ),
            )
        )

        no_access_result = (
            service.lookup(
                identity=(
                    no_access_identity
                ),
                request=(
                    TableLookupRequest(
                        section_key="LOT_2",
                        ordinal=5,
                        column_name=(
                            "Soumissionnaires"
                        ),
                    )
                ),
            )
        )

        assert (
            no_access_result
            is None
        )

        print(
            "NO-READ IDENTITY "
            "RECEIVES NO EVIDENCE: PASS"
        )

        # ====================================================
        # FINAL DATABASE COUNTS
        # ====================================================

        persisted_count = (
            db.scalar(
                select(
                    func.count(
                        TableEvidenceRecord.id
                    )
                )
                .where(
                    TableEvidenceRecord
                    .document_id.in_(
                        {
                            allowed_document_id,
                            secret_document_id,
                        }
                    )
                )
            )
        )

        assert persisted_count == 2

        print()
        print(
            "=" * 100
        )

        print(
            "REAL POSTGRES TABLE-EVIDENCE "
            "ACL ISOLATION: PASS"
        )

        print(
            "=" * 100
        )

    finally:
        # ====================================================
        # CLEANUP
        #
        # The table-evidence indexer commits internally,
        # so cleanup must also be committed explicitly.
        # ====================================================

        db.rollback()

        try:
            db.execute(
                delete(
                    TableEvidenceRecord
                )
                .where(
                    TableEvidenceRecord
                    .document_id.in_(
                        {
                            allowed_document_id,
                            secret_document_id,
                        }
                    )
                )
            )

            db.execute(
                delete(
                    DocumentPermission
                )
                .where(
                    DocumentPermission
                    .user_id
                    == user_id
                )
            )

            db.execute(
                delete(
                    DocumentPage
                )
                .where(
                    DocumentPage.id.in_(
                        {
                            allowed_page_id,
                            secret_page_id,
                        }
                    )
                )
            )

            db.execute(
                delete(
                    DocumentVersion
                )
                .where(
                    DocumentVersion.id.in_(
                        {
                            allowed_version_id,
                            secret_version_id,
                        }
                    )
                )
            )

            db.execute(
                delete(
                    Document
                )
                .where(
                    Document.id.in_(
                        {
                            allowed_document_id,
                            secret_document_id,
                        }
                    )
                )
            )

            db.execute(
                delete(
                    User
                )
                .where(
                    User.id
                    == user_id
                )
            )

            db.commit()

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()
