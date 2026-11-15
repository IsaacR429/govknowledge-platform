from __future__ import annotations

import os
import secrets
from uuid import uuid4

from fastapi.testclient import TestClient
from pwdlib import PasswordHash
from sqlalchemy import (
    delete,
    select,
    text,
)
from sqlalchemy.engine import make_url

from app.core.config import (
    get_settings,
)
from app.infrastructure.database.models.audit import (
    AuditEventRecord,
)
from app.infrastructure.database.models.authentication import (
    AuthSession,
    LocalCredential,
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
from app.infrastructure.database.session import (
    SessionLocal,
    engine,
)
from app.main import (
    create_app,
)


EXPECTED_SYSTEM_ID = (
    "7686228710959620134"
)


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
        os.environ[
            "DATABASE_URL"
        ]
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
        and bool(
            expected.password
        )
        and actual == expected
        and engine.url == expected
    ):
        raise RuntimeError(
            "STOP: test is not isolated QA."
        )

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

    assert (
        database
        == "govknowledge_qa"
    )

    assert (
        username
        == "govknowledge_qa"
    )

    assert (
        system_id
        == EXPECTED_SYSTEM_ID
    )


def test_logged_in_user_without_read_gets_404():
    assert_isolated_qa()

    db = SessionLocal()

    user_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    page_id = uuid4()

    username = (
        "g29-no-read-"
        + user_id.hex
    )

    password = (
        "G29-"
        + secrets.token_urlsafe(
            24
        )
    )

    try:

        # ====================================================
        # REAL ACTIVE USER + REAL LOCAL CREDENTIAL
        # ====================================================

        db.add(
            User(
                id=user_id,
                username=username,
                display_name=(
                    "G29 No-Read User"
                ),
                email=None,
                department="QA",
                language_preference="fr",
                is_active=True,
            )
        )

        db.flush()

        db.add(
            LocalCredential(
                user_id=user_id,
                password_hash=(
                    PasswordHash
                    .recommended()
                    .hash(
                        password
                    )
                ),
            )
        )

        # ====================================================
        # REAL EXISTING DOCUMENT / VERSION / PAGE
        #
        # IMPORTANT:
        # deliberately NO DocumentPermission is inserted.
        # ====================================================

        db.add(
            Document(
                id=document_id,
                title=(
                    "G29 Restricted "
                    "Source Page Fixture"
                ),
                description=(
                    "Existing page inaccessible "
                    "to the test user."
                ),
                document_type="QA_POLICY",
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
                    "RESTRICTED"
                ),
            )
        )

        db.add(
            DocumentVersion(
                id=version_id,
                document_id=document_id,
                version_number=1,
                original_filename=(
                    "g29-restricted.pdf"
                ),
                storage_path=(
                    "qa-g29/"
                    "g29-restricted.pdf"
                ),
                file_hash=(
                    uuid4().hex
                    + uuid4().hex
                ),
                processing_status=(
                    "SUCCEEDED"
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
                page_number=7,
                extracted_text=(
                    "This page physically exists "
                    "but the current user must "
                    "not be allowed to read it."
                ),
                metadata_json={},
            )
        )

        db.commit()

        # Explicitly prove there is no READ permission.
        permission = db.scalar(
            select(
                DocumentPermission.id
            )
            .where(
                DocumentPermission.user_id
                == user_id,
                DocumentPermission.document_id
                == document_id,
                DocumentPermission.action
                == "READ",
            )
        )

        assert permission is None

        print()
        print(
            "EXISTING SOURCE PAGE CREATED: PASS"
        )

        print(
            "USER HAS NO READ PERMISSION: PASS"
        )

        # ====================================================
        # REAL FASTAPI APPLICATION
        # ====================================================

        application = create_app()

        prefix = (
            get_settings().api_prefix
        )

        page_url = (
            f"{prefix}/source-pages/"
            f"{page_id}"
        )

        missing_page_id = uuid4()

        missing_url = (
            f"{prefix}/source-pages/"
            f"{missing_page_id}"
        )

        with TestClient(
            application,
            raise_server_exceptions=True,
        ) as client:

            # =================================================
            # ANONYMOUS ACCESS MUST BE 401
            # =================================================

            anonymous = client.get(
                page_url
            )

            print()
            print(
                "ANONYMOUS STATUS:",
                anonymous.status_code,
            )

            assert (
                anonymous.status_code
                == 401
            )

            # =================================================
            # REAL LOGIN
            # =================================================

            login = client.post(
                f"{prefix}/auth/login",
                json={
                    "username": username,
                    "password": password,
                },
            )

            print()
            print(
                "LOGIN STATUS:",
                login.status_code,
            )

            assert (
                login.status_code
                == 200
            ), login.text

            token = (
                login.json()[
                    "access_token"
                ]
            )

            assert token

            bearer = {
                "Authorization": (
                    f"Bearer {token}"
                )
            }

            # =================================================
            # PROVE TOKEN RESOLVES AS LOGGED-IN USER
            # =================================================

            me = client.get(
                f"{prefix}/me",
                headers=bearer,
            )

            print(
                "ME STATUS:",
                me.status_code,
            )

            assert (
                me.status_code
                == 200
            ), me.text

            assert (
                me.json()[
                    "user_id"
                ]
                == str(
                    user_id
                )
            )

            print(
                "REAL AUTHENTICATED SESSION: PASS"
            )

            # =================================================
            # EXISTING PAGE, BUT USER HAS NO READ
            #
            # REQUIRED G29 RESULT: 404
            # =================================================

            denied = client.get(
                page_url,
                headers=bearer,
            )

            print()
            print(
                "NO-READ EXISTING PAGE STATUS:",
                denied.status_code,
            )

            print(
                "NO-READ EXISTING PAGE BODY:",
                denied.json(),
            )

            assert (
                denied.status_code
                == 404
            )

            assert (
                denied.json()
                == {
                    "detail": (
                        "Source page not found."
                    )
                }
            )

            # =================================================
            # GENUINELY NONEXISTENT PAGE
            #
            # Response must be indistinguishable.
            # =================================================

            missing = client.get(
                missing_url,
                headers=bearer,
            )

            print()
            print(
                "NONEXISTENT PAGE STATUS:",
                missing.status_code,
            )

            print(
                "NONEXISTENT PAGE BODY:",
                missing.json(),
            )

            assert (
                missing.status_code
                == 404
            )

            assert (
                missing.json()
                == denied.json()
            )

            print(
                "ANTI-ENUMERATION RESPONSE: PASS"
            )

        # ====================================================
        # AUDIT VERIFICATION
        # ====================================================

        db.expire_all()

        denied_events = tuple(
            db.scalars(
                select(
                    AuditEventRecord
                )
                .where(
                    AuditEventRecord
                    .event_type
                    == "SOURCE_PAGE_DENIED",
                    AuditEventRecord
                    .actor_user_id
                    == user_id,
                    AuditEventRecord
                    .resource_id
                    == page_id,
                    AuditEventRecord
                    .document_id
                    == document_id,
                )
            )
        )

        assert denied_events

        latest_denied = (
            denied_events[-1]
        )

        assert (
            latest_denied.outcome
            == "DENIED"
        )

        assert (
            latest_denied.error_code
            == "DOCUMENT_READ_DENIED"
        )

        assert (
            latest_denied.metadata_json
            == {
                "permission": "READ"
            }
        )

        viewed_events = tuple(
            db.scalars(
                select(
                    AuditEventRecord
                )
                .where(
                    AuditEventRecord
                    .event_type
                    == "SOURCE_PAGE_VIEWED",
                    AuditEventRecord
                    .actor_user_id
                    == user_id,
                    AuditEventRecord
                    .resource_id
                    == page_id,
                )
            )
        )

        assert viewed_events == ()

        print()
        print(
            "SOURCE_PAGE_DENIED AUDIT: PASS"
        )

        print(
            "SOURCE_PAGE_VIEWED ABSENT: PASS"
        )

        print()
        print(
            "=" * 90
        )

        print(
            "G29 LIVE LOGGED-IN "
            "NO-READ SOURCE PAGE 404: PASS"
        )

        print(
            "=" * 90
        )

    finally:

        db.rollback()

        try:

            # Audit records do not need to survive
            # this isolated qualification fixture.
            db.execute(
                delete(
                    AuditEventRecord
                )
                .where(
                    AuditEventRecord
                    .actor_user_id
                    == user_id
                )
            )

            db.execute(
                delete(
                    AuthSession
                )
                .where(
                    AuthSession.user_id
                    == user_id
                )
            )

            db.execute(
                delete(
                    LocalCredential
                )
                .where(
                    LocalCredential.user_id
                    == user_id
                )
            )

            db.execute(
                delete(
                    DocumentPermission
                )
                .where(
                    DocumentPermission.user_id
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
