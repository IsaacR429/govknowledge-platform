from __future__ import annotations

import os
from uuid import uuid4

from sqlalchemy import delete, text
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
from app.core.config import get_settings
from app.domain.enums.permission import PermissionAction
from app.infrastructure.database.models.documents import (
    Document,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.identity import User
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


def assert_isolated_qa():
    assert (
        os.environ.get("GOVKNOWLEDGE_QA_SYSTEM_ID")
        == EXPECTED_SYSTEM_ID
    )

    expected = make_url(
        os.environ["DATABASE_URL"]
    )

    actual = make_url(
        get_settings().database_url
    )

    assert expected.host == "127.0.0.1"
    assert expected.port == 15432
    assert expected.database == "govknowledge_qa"
    assert expected.username == "govknowledge_qa"
    assert actual == expected
    assert engine.url == expected

    with engine.connect() as connection:
        database, username, system_id = (
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

    assert database == "govknowledge_qa"
    assert username == "govknowledge_qa"
    assert system_id == EXPECTED_SYSTEM_ID


def add_document(
    db,
    *,
    document_id,
    version_id,
    page_id,
    title,
):
    db.add(
        Document(
            id=document_id,
            title=title,
            description="Vocabulary security fixture",
            document_type="PROCUREMENT",
            issuing_organization="GovKnowledge QA",
            jurisdiction="Burkina Faso",
            language="fr",
            publication_date=None,
            source_url=None,
            status="ACTIVE",
            confidentiality_level="PUBLIC",
        )
    )

    db.add(
        DocumentVersion(
            id=version_id,
            document_id=document_id,
            version_number=1,
            original_filename=f"{title}.pdf",
            storage_path=f"qa/{version_id}.pdf",
            file_hash=uuid4().hex + uuid4().hex,
            processing_status="SUCCEEDED",
            effective_date=None,
            superseded_at=None,
        )
    )

    db.add(
        DocumentPage(
            id=page_id,
            document_version_id=version_id,
            page_number=1,
            extracted_text="QA table vocabulary fixture",
            metadata_json={},
        )
    )


def add_table_row(
    db,
    *,
    evidence_id,
    document_id,
    version_id,
    page_id,
    section_key,
    logical_key,
    columns,
):
    db.add(
        TableEvidenceRecord(
            evidence_id=evidence_id,
            document_id=document_id,
            document_version_id=version_id,
            page_id=page_id,
            page_number=1,
            row_index=1,
            role="DATA",
            logical_key=logical_key,
            section_key=section_key,
            language="fr",
            source_table_refs_json=["qa:test"],
            column_names_json=list(columns),
            cells_json=[],
            search_text="qa vocabulary security test",
        )
    )


def test_authorized_vocabulary_does_not_leak():
    assert_isolated_qa()

    db = SessionLocal()

    user_id = uuid4()

    authorized_document_id = uuid4()
    authorized_version_id = uuid4()
    authorized_page_id = uuid4()

    secret_document_id = uuid4()
    secret_version_id = uuid4()
    secret_page_id = uuid4()

    permission_id = uuid4()

    username = f"table-vocab-{user_id}"

    try:
        # ----------------------------------------------------
        # User
        # ----------------------------------------------------

        db.add(
            User(
                id=user_id,
                username=username,
                display_name="Table Vocabulary QA",
                email=None,
                department="QA",
                language_preference="fr",
                is_active=True,
            )
        )

        # ----------------------------------------------------
        # Two ACTIVE documents
        # ----------------------------------------------------

        add_document(
            db,
            document_id=authorized_document_id,
            version_id=authorized_version_id,
            page_id=authorized_page_id,
            title="Authorized Procurement",
        )

        add_document(
            db,
            document_id=secret_document_id,
            version_id=secret_version_id,
            page_id=secret_page_id,
            title="Secret Procurement",
        )

        db.flush()

        # ----------------------------------------------------
        # READ permission ONLY on authorized document
        # ----------------------------------------------------

        db.add(
            DocumentPermission(
                id=permission_id,
                user_id=user_id,
                document_id=authorized_document_id,
                action="READ",
            )
        )

        # ----------------------------------------------------
        # Authorized vocabulary
        # ----------------------------------------------------

        add_table_row(
            db,
            evidence_id=f"table:{authorized_version_id}:LOT_2/05",
            document_id=authorized_document_id,
            version_id=authorized_version_id,
            page_id=authorized_page_id,
            section_key="LOT_2",
            logical_key="LOT_2/05",
            columns=(
                "N°",
                "Soumissionnaires",
            ),
        )

        # ----------------------------------------------------
        # Unauthorized vocabulary
        # ----------------------------------------------------

        add_table_row(
            db,
            evidence_id=f"table:{secret_version_id}:SECRET_LOT/05",
            document_id=secret_document_id,
            version_id=secret_version_id,
            page_id=secret_page_id,
            section_key="SECRET_LOT",
            logical_key="SECRET_LOT/05",
            columns=(
                "N°",
                "Confidential Salary",
            ),
        )

        db.commit()

        print()
        print("BOTH TABLE-EVIDENCE ROWS PHYSICALLY PRESENT: PASS")

        # ----------------------------------------------------
        # Real authorization path
        # ----------------------------------------------------

        identity = CurrentIdentity(
            user_id=user_id,
            username=username,
            display_name="Table Vocabulary QA",
            email=None,
            department="QA",
            language_preference="fr",
            roles=("EMPLOYEE",),
        )

        authorization = AuthorizationService(
            PostgresAuthorizationRepository(db)
        )

        repository = PostgresTableEvidenceRepository(db)

        service = AuthorizedTableEvidenceService(
            authorization=authorization,
            repository=repository,
        )

        scope = authorization.retrieval_scope(
            identity
        )

        assert scope.document_ids == frozenset(
            (authorized_document_id,)
        )

        result = service.vocabulary(
            identity=identity
        )

        print(
            "AUTHORIZED SECTION KEYS:",
            result.section_keys,
        )

        print(
            "AUTHORIZED COLUMN NAMES:",
            result.column_names,
        )

        assert result.section_keys == (
            "LOT_2",
        )

        assert result.column_names == (
            "N°",
            "Soumissionnaires",
        )

        assert (
            "SECRET_LOT"
            not in result.section_keys
        )

        assert (
            "Confidential Salary"
            not in result.column_names
        )

        print("NORMAL ACL ISOLATION: PASS")

        # ----------------------------------------------------
        # Poisoned scope:
        #
        # Even if upstream gives repository BOTH document IDs,
        # live permission join must still exclude secret one.
        # ----------------------------------------------------

        poisoned_scope = AuthorizationScope(
            user_id=user_id,
            action=PermissionAction.READ,
            document_ids=frozenset(
                (
                    authorized_document_id,
                    secret_document_id,
                )
            ),
        )

        poisoned = (
            repository.list_authorized_vocabulary(
                scope=poisoned_scope
            )
        )

        print(
            "POISONED-SCOPE SECTION KEYS:",
            poisoned.section_keys,
        )

        print(
            "POISONED-SCOPE COLUMN NAMES:",
            poisoned.column_names,
        )

        assert poisoned.section_keys == (
            "LOT_2",
        )

        assert poisoned.column_names == (
            "N°",
            "Soumissionnaires",
        )

        assert (
            "SECRET_LOT"
            not in poisoned.section_keys
        )

        assert (
            "Confidential Salary"
            not in poisoned.column_names
        )

        print("POISONED-SCOPE ATTACK BLOCKED: PASS")

        # ----------------------------------------------------
        # Revoke READ but retain old poisoned scope object.
        # ----------------------------------------------------

        db.execute(
            delete(DocumentPermission)
            .where(
                DocumentPermission.id
                == permission_id
            )
        )

        db.commit()

        stale = (
            repository.list_authorized_vocabulary(
                scope=poisoned_scope
            )
        )

        print(
            "STALE-SCOPE SECTION KEYS:",
            stale.section_keys,
        )

        print(
            "STALE-SCOPE COLUMN NAMES:",
            stale.column_names,
        )

        assert stale.section_keys == ()
        assert stale.column_names == ()

        print("STALE SCOPE AFTER ACL REVOCATION: BLOCKED")

        # ----------------------------------------------------
        # Normal service must also now see no vocabulary.
        # ----------------------------------------------------

        no_read = service.vocabulary(
            identity=identity
        )

        assert no_read.section_keys == ()
        assert no_read.column_names == ()

        print("NO-READ SERVICE VOCABULARY: EMPTY")

        print()
        print("=" * 90)
        print(
            "REAL POSTGRES AUTHORIZED "
            "TABLE VOCABULARY: PASS"
        )
        print("=" * 90)

    finally:
        db.rollback()

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        try:
            db.execute(
                delete(TableEvidenceRecord)
                .where(
                    TableEvidenceRecord.document_id.in_(
                        (
                            authorized_document_id,
                            secret_document_id,
                        )
                    )
                )
            )

            db.execute(
                delete(DocumentPermission)
                .where(
                    DocumentPermission.user_id
                    == user_id
                )
            )

            db.execute(
                delete(DocumentPage)
                .where(
                    DocumentPage.id.in_(
                        (
                            authorized_page_id,
                            secret_page_id,
                        )
                    )
                )
            )

            db.execute(
                delete(DocumentVersion)
                .where(
                    DocumentVersion.id.in_(
                        (
                            authorized_version_id,
                            secret_version_id,
                        )
                    )
                )
            )

            db.execute(
                delete(Document)
                .where(
                    Document.id.in_(
                        (
                            authorized_document_id,
                            secret_document_id,
                        )
                    )
                )
            )

            db.execute(
                delete(User)
                .where(
                    User.id == user_id
                )
            )

            db.commit()

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()
