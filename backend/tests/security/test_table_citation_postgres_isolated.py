from __future__ import annotations

import os
from dataclasses import replace
from uuid import uuid4

import pytest

from sqlalchemy import (
    delete,
    select,
    text,
)
from sqlalchemy.engine import make_url

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.table_citation_validation import (
    InvalidTableCitationError,
    TableClaimCitationValidator,
)
from app.application.services.table_claim_provenance import (
    DirectTableClaimBuilder,
    TableAtomicClaim,
)
from app.application.services.table_evidence_atoms import (
    TableCellEvidenceAtom,
    TableEvidenceAtom,
    TableLookupRequest,
    TableLookupSelection,
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

SUPPLIER = (
    "ENTREPRISE ZERBO AWA "
    "ET FRERES (EZAF)"
)

LOGICAL_KEY = "LOT_2/05"


def assert_isolated_qa():
    if (
        os.environ.get(
            "GOVKNOWLEDGE_QA_SYSTEM_ID"
        )
        != EXPECTED_SYSTEM_ID
    ):
        raise RuntimeError(
            "STOP: QA system ID guard missing."
        )

    expected = make_url(
        os.environ["DATABASE_URL"]
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
            "STOP: test is not isolated QA."
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

    assert database == "govknowledge_qa"
    assert username == "govknowledge_qa"
    assert system_id == EXPECTED_SYSTEM_ID


def build_atom(
    *,
    document_id,
    version_id,
    page_id,
):
    supplier_span = TableSourceSpan(
        page_number=4,
        kind=TableSpanKind.TEXT,
        source_ref=(
            "quotidien:xref781:"
            "lot2:row05:supplier"
        ),
        bbox=None,
        text=SUPPLIER,
    )

    structure_span = TableSourceSpan(
        page_number=4,
        kind=TableSpanKind.STRUCTURE,
        source_ref=(
            "quotidien:xref781:"
            "lot2:row05"
        ),
        bbox=None,
        text=None,
    )

    ordinal = TableCellEvidenceAtom(
        column_start=0,
        column_end=1,
        column_name="N°",
        text="05",
        value_source=(
            TableCellValueSource
            .SOURCE_TEXT
        ),
        origin_spans=(
            TableSourceSpan(
                page_number=4,
                kind=TableSpanKind.TEXT,
                source_ref=(
                    "quotidien:xref781:"
                    "lot2:row05:ordinal"
                ),
                bbox=None,
                text="05",
            ),
        ),
        structure_spans=(
            structure_span,
        ),
    )

    supplier = TableCellEvidenceAtom(
        column_start=1,
        column_end=2,
        column_name="Soumissionnaires",
        text=SUPPLIER,
        value_source=(
            TableCellValueSource
            .SOURCE_TEXT
        ),
        origin_spans=(
            supplier_span,
        ),
        structure_spans=(
            structure_span,
        ),
    )

    atom = TableEvidenceAtom(
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
            "xref:781",
        ),
        row_index=10,
        role=TableRowRole.DATA,
        logical_key=LOGICAL_KEY,
        section_key="LOT_2",
        column_names=(
            "N°",
            "Soumissionnaires",
        ),
        cells=(
            ordinal,
            supplier,
        ),
        language="fr",
    )

    request = TableLookupRequest(
        section_key="LOT_2",
        ordinal=5,
        column_name="Soumissionnaires",
    )

    selection = TableLookupSelection(
        request=request,
        row=atom,
        cell=supplier,
    )

    return (
        atom,
        request,
        selection,
    )


def test_real_postgres_table_citation_revalidation():
    assert_isolated_qa()

    db = SessionLocal()

    user_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    page_id = uuid4()
    permission_id = uuid4()

    username = (
        "table-citation-"
        f"{user_id}"
    )

    try:
        # ====================================================
        # USER + DOCUMENT GRAPH
        # ====================================================

        db.add(
            User(
                id=user_id,
                username=username,
                display_name=(
                    "Table Citation QA User"
                ),
                email=None,
                department="QA",
                language_preference="fr",
                is_active=True,
            )
        )

        db.add(
            Document(
                id=document_id,
                title=(
                    "Quotidien QA "
                    "Table Citation Fixture"
                ),
                description=(
                    "Typed table citation "
                    "qualification fixture."
                ),
                document_type=(
                    "PROCUREMENT"
                ),
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
                    "PUBLIC"
                ),
            )
        )

        db.add(
            DocumentVersion(
                id=version_id,
                document_id=document_id,
                version_number=1,
                original_filename=(
                    "quotidien-table-"
                    "citation.pdf"
                ),
                storage_path=(
                    "qa-table-citation/"
                    "quotidien.pdf"
                ),
                file_hash="c" * 64,
                processing_status=(
                    ProcessingStatus
                    .SUCCEEDED
                    .value
                ),
                effective_date=None,
                superseded_at=None,
            )
        )

        db.add(
            DocumentPage(
                id=page_id,
                document_version_id=(
                    version_id
                ),
                page_number=4,
                extracted_text=(
                    "Quotidien procurement "
                    "table page."
                ),
                metadata_json={},
            )
        )

        db.add(
            DocumentPermission(
                id=permission_id,
                user_id=user_id,
                document_id=document_id,
                action=(
                    PermissionAction
                    .READ
                    .value
                ),
            )
        )

        db.commit()

        # ====================================================
        # PERSIST TYPED TABLE EVIDENCE
        # ====================================================

        repository = (
            PostgresTableEvidenceRepository(
                db
            )
        )

        atom, request, selection = (
            build_atom(
                document_id=document_id,
                version_id=version_id,
                page_id=page_id,
            )
        )

        result = (
            TableEvidenceIndexingService(
                repository=repository
            )
            .index_version(
                version_id=version_id,
                atoms=(
                    atom,
                ),
            )
        )

        assert (
            result.indexed_count
            == 1
        )

        persisted = (
            db.scalar(
                select(
                    TableEvidenceRecord
                )
                .where(
                    TableEvidenceRecord
                    .evidence_id
                    == atom.evidence_id
                )
            )
        )

        assert persisted is not None

        assert (
            persisted.search_text
            == atom.search_text
        )

        print()
        print(
            "PERSISTED EVIDENCE ID:",
            persisted.evidence_id,
        )

        print(
            "PERSISTED PAGE:",
            persisted.page_number,
        )

        print(
            "PERSISTED CANONICAL "
            "TABLE EVIDENCE: PASS"
        )

        # ====================================================
        # BUILD CLAIM FROM TYPED SELECTION
        # ====================================================

        claim = (
            DirectTableClaimBuilder()
            .build(
                request=request,
                selection=selection,
            )
        )

        assert claim.text == SUPPLIER

        print(
            "CLAIM:",
            claim.text,
        )

        print(
            "CLAIM ORIGIN PAGE:",
            claim.origin.page_number,
        )

        # ====================================================
        # REAL ACL + REAL POSTGRES RE-READ
        # ====================================================

        identity = CurrentIdentity(
            user_id=user_id,
            username=username,
            display_name=(
                "Table Citation QA User"
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

        validator = (
            TableClaimCitationValidator(
                authorization=(
                    authorization
                ),
                repository=(
                    repository
                ),
            )
        )

        citation = (
            validator.validate(
                identity=identity,
                claim=claim,
            )
        )

        assert (
            citation.evidence_id
            == atom.evidence_id
        )

        assert (
            citation.document_id
            == document_id
        )

        assert (
            citation.document_version_id
            == version_id
        )

        assert (
            citation.page_id
            == page_id
        )

        assert (
            citation.page_number
            == 4
        )

        assert (
            citation.logical_key
            == "LOT_2/05"
        )

        assert (
            citation.column_name
            == "Soumissionnaires"
        )

        assert (
            citation.cited_text
            == SUPPLIER
        )

        assert (
            citation.origin_pages
            == (4,)
        )

        assert (
            citation.structure_pages
            == (4,)
        )

        assert (
            citation.source_refs
            == (
                (
                    "quotidien:xref781:"
                    "lot2:row05:supplier"
                ),
            )
        )

        print()
        print(
            "CITATION TEXT:",
            citation.cited_text,
        )

        print(
            "CITATION PAGE:",
            citation.page_number,
        )

        print(
            "CITATION SOURCE REFS:",
            citation.source_refs,
        )

        print(
            "POSTGRES CANONICAL "
            "REVALIDATION: PASS"
        )

        # ====================================================
        # TAMPER TEST
        #
        # Claim and origin agree with one another,
        # but no longer agree with persisted evidence.
        # Persistence revalidation MUST reject it.
        # ====================================================

        tampered_origin = replace(
            claim.origin,
            text=(
                "FAKE SUPPLIER"
            ),
        )

        tampered_claim = (
            TableAtomicClaim(
                claim_id=(
                    claim.claim_id
                ),
                text="FAKE SUPPLIER",
                origin=(
                    tampered_origin
                ),
                language=(
                    claim.language
                ),
                required=True,
            )
        )

        with pytest.raises(
            InvalidTableCitationError
        ):
            validator.validate(
                identity=identity,
                claim=tampered_claim,
            )

        print(
            "PERSISTENCE TAMPER "
            "REJECTION: PASS"
        )

        # ====================================================
        # REMOVE READ ACL
        #
        # Same canonical evidence still exists,
        # but citation disclosure must now fail.
        # ====================================================

        db.execute(
            delete(
                DocumentPermission
            )
            .where(
                DocumentPermission.id
                == permission_id
            )
        )

        db.commit()

        still_persisted = (
            db.scalar(
                select(
                    TableEvidenceRecord.id
                )
                .where(
                    TableEvidenceRecord
                    .evidence_id
                    == atom.evidence_id
                )
            )
        )

        assert still_persisted is not None

        with pytest.raises(
            InvalidTableCitationError
        ):
            validator.validate(
                identity=identity,
                claim=claim,
            )

        print(
            "EVIDENCE STILL EXISTS "
            "AFTER ACL REMOVAL: PASS"
        )

        print(
            "NO-READ CITATION "
            "DISCLOSURE BLOCKED: PASS"
        )

        print()
        print(
            "=" * 100
        )

        print(
            "REAL POSTGRES TABLE "
            "CITATION REVALIDATION: PASS"
        )

        print(
            "=" * 100
        )

    finally:
        db.rollback()

        try:
            db.execute(
                delete(
                    TableEvidenceRecord
                )
                .where(
                    TableEvidenceRecord
                    .document_id
                    == document_id
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
                    DocumentPage.id
                    == page_id
                )
            )

            db.execute(
                delete(
                    DocumentVersion
                )
                .where(
                    DocumentVersion.id
                    == version_id
                )
            )

            db.execute(
                delete(
                    Document
                )
                .where(
                    Document.id
                    == document_id
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
