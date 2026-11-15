
"""Real-session HTTP QA + ACL test against isolated PostgreSQL only.

Required environment:
  DATABASE_URL -> 127.0.0.1:15432/govknowledge_qa
  GOVKNOWLEDGE_QA_SYSTEM_ID -> the QA container's PostgreSQL system ID

Never run this test against the original qualification database.
No Anthropic calls or model downloads occur.
"""
from __future__ import annotations

import os
import re
import secrets
from hashlib import sha256
from types import SimpleNamespace
from typing import Annotated
from uuid import uuid4

from fastapi import Depends
from fastapi.testclient import TestClient
from pwdlib import PasswordHash
from sqlalchemy import delete, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.api.dependencies import document_processing as processing_dependencies
from app.api.dependencies import qa as qa_dependencies
from app.api.dependencies.qa import get_question_answer_service
from app.api.routes import documents as document_routes
from app.application.ports.question_claim_planner import PlannedClaim
from app.application.services.audit import AuditService
from app.application.services.authorization import AuthorizationService
from app.application.services.authorized_search import AuthorizedSearchService
from app.application.services.grounded_answer_generation import (
    GroundedAnswerResult,
)
from app.application.services.question_answer import QuestionAnswerService
from app.application.services.validated_citations import ValidatedCitationSet
from app.core.config import get_settings
from app.domain.enums.permission import PermissionAction
from app.domain.enums.processing import ProcessingStatus
from app.infrastructure.database.models.audit import AuditEventRecord
from app.infrastructure.database.models.authentication import (
    AuthSession,
    LocalCredential,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.embeddings import (
    ChunkEmbedding,
    EMBEDDING_DIMENSION,
)
from app.infrastructure.database.models.identity import Role, User
from app.infrastructure.database.models.permissions import DocumentPermission
from app.infrastructure.database.models.processing import DocumentProcessingJob
from app.infrastructure.database.repositories.authorization import (
    PostgresAuthorizationRepository,
)
from app.infrastructure.database.session import SessionLocal, engine, get_db
from app.infrastructure.retrieval.postgres_vector_retrieval import (
    PostgresVectorRetrievalAdapter,
)
from app.main import create_app


MODEL_NAME = "qa-auth-acl-isolation-test"
MODEL_REVISION = "revision-1"

FACT = "The Court shall consist of fifteen members."
ALLOWED_TEXT = "Article 3\n\n1. " + FACT
SECRET_TEXT = (
    "SECRET-DO-NOT-DISCLOSE: "
    "this unauthorized vector is the strongest match."
)
REFUSAL = (
    "Les sources autorisees ne permettent pas "
    "de repondre de maniere fiable."
)


def _unit_vector(first: float, second: float) -> list[float]:
    return [first, second] + [0.0] * (EMBEDDING_DIMENSION - 2)


def _assert_isolated_target() -> None:
    """Guard evaluated before the first fixture write."""
    expected_system_id = os.environ.get(
        "GOVKNOWLEDGE_QA_SYSTEM_ID", ""
    )

    if not re.fullmatch(r"\d+", expected_system_id):
        raise RuntimeError(
            "STOP: isolated PostgreSQL system ID not provided."
        )

    expected = make_url(os.environ.get("DATABASE_URL", ""))
    actual = make_url(get_settings().database_url)

    if (
        expected.drivername != "postgresql+psycopg"
        or expected.host != "127.0.0.1"
        or expected.port != 15432
        or expected.database != "govknowledge_qa"
        or expected.username != "govknowledge_qa"
        or not expected.password
        or actual != expected
        or engine.url != expected
        or SessionLocal.kw.get("bind") is not engine
    ):
        raise RuntimeError(
            "STOP: QA test is not bound to isolated PostgreSQL."
        )

    with engine.connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))

        identity = connection.execute(
            text("""
                SELECT
                    current_database(),
                    current_user,
                    system_identifier::text
                FROM pg_control_system()
            """)
        ).one()

        connection.rollback()

    if identity != (
        "govknowledge_qa",
        "govknowledge_qa",
        expected_system_id,
    ):
        raise RuntimeError(
            "STOP: actual PostgreSQL instance does not match "
            "the QA container."
        )


class FixedEmbedding:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, text: str):
        self.calls.append(text)

        return SimpleNamespace(
            values=tuple(_unit_vector(1.0, 0.0)),
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
        )


class RecordingPlanner:
    def __init__(self, expected_chunk_id) -> None:
        self.expected_chunk_id = expected_chunk_id
        self.calls = []

    def plan(self, request):
        self.calls.append(request)

        # These are REAL pgvector results constrained by
        # the REAL PostgreSQL authorization scope.
        assert {
            item.chunk_id for item in request.evidence
        } == {self.expected_chunk_id}

        assert all(
            SECRET_TEXT not in item.content
            for item in request.evidence
        )

        assert FACT in request.evidence[0].content
        assert request.source_language == "en"

        # Must literally occur in authorized evidence.
        return (PlannedClaim(text=FACT),)


class RecordingEvidence:
    def __init__(self) -> None:
        self.calls = []
        self.result = object()

    def evaluate(self, *, identity, claims, correlation_id):
        self.calls.append((identity, claims, correlation_id))
        return self.result


class DeterministicRefusal:
    def __init__(self, expected_evidence) -> None:
        self.calls = []
        self.expected_evidence = expected_evidence

    def generate(self, *, question, evidence_result, language):
        assert evidence_result is self.expected_evidence
        self.calls.append((question, language))

        return GroundedAnswerResult(
            correlation_id=uuid4(),
            status="INSUFFICIENT_EVIDENCE",
            answer=REFUSAL,
            citations=ValidatedCitationSet(citations=()),
            cited_evidence_ids=(),
        )


class MemoryAuditRepository:
    def __init__(self) -> None:
        self.events = []

    def append(self, event):
        self.events.append(event)
        return uuid4()


def _seed(
    db: Session,
    *,
    user_id,
    allowed,
    secret,
    password_hash: str,
) -> None:
    db.add(
        User(
            id=user_id,
            username=f"isolated-qa-{user_id.hex}",
            display_name="Isolated QA Test User",
            email=None,
            department="Test",
            language_preference="fr",
            is_active=True,
        )
    )
    db.flush()

    db.add(
        LocalCredential(
            user_id=user_id,
            password_hash=password_hash,
        )
    )

    for spec in (allowed, secret):
        (
            doc_id,
            version_id,
            page_id,
            chunk_id,
            embedding_id,
            label,
            content,
            vector,
        ) = spec

        db.add(
            Document(
                id=doc_id,
                title=label,
                description="Synthetic isolated QA test document",
                document_type="POLICY",
                issuing_organization="QA Integration Test",
                jurisdiction="Test",
                language="en",
                publication_date=None,
                source_url=None,
                status="ACTIVE",
                confidentiality_level=(
                    "PUBLIC" if spec is allowed else "SECRET"
                ),
            )
        )

        db.add(
            DocumentVersion(
                id=version_id,
                document_id=doc_id,
                version_number=1,
                original_filename=f"{label}.pdf",
                storage_path=f"isolated-qa/{doc_id}.pdf",
                file_hash=(
                    "a" if spec is allowed else "b"
                ) * 64,
                processing_status=ProcessingStatus.SUCCEEDED.value,
                effective_date=None,
                superseded_at=None,
            )
        )

        db.add(
            DocumentPage(
                id=page_id,
                document_version_id=version_id,
                page_number=1,
                extracted_text=content,
                metadata_json={},
            )
        )

        db.add(
            DocumentChunk(
                id=chunk_id,
                document_id=doc_id,
                document_version_id=version_id,
                page_id=page_id,
                page_start=1,
                page_end=1,
                chunk_index=0,
                section_title=None,
                content=content,
                language="en",
                metadata_json={},
            )
        )

        # Persist the document graph before embedding FK.
        db.flush()

        db.add(
            ChunkEmbedding(
                id=embedding_id,
                chunk_id=chunk_id,
                model_name=MODEL_NAME,
                model_revision=MODEL_REVISION,
                dimension=EMBEDDING_DIMENSION,
                normalized=True,
                embedding=vector,
            )
        )

    db.commit()


def _cleanup(*, user_id, employee_id, allowed, secret) -> None:
    """Delete only this test's UUIDs in foreign-key-safe order."""
    specs = (allowed, secret)
    test_user_ids = [user_id, employee_id]

    document_ids = [item[0] for item in specs]
    version_ids = [item[1] for item in specs]
    page_ids = [item[2] for item in specs]
    chunk_ids = [item[3] for item in specs]
    embedding_ids = [item[4] for item in specs]

    with SessionLocal() as db:
        db.execute(
            delete(AuditEventRecord).where(
                AuditEventRecord.actor_user_id.in_(test_user_ids),
                AuditEventRecord.document_id.in_(document_ids),
            )
        )
        db.execute(
            delete(AuthSession).where(
                AuthSession.user_id.in_(test_user_ids)
            )
        )
        db.execute(
            delete(LocalCredential).where(
                LocalCredential.user_id.in_(test_user_ids)
            )
        )
        db.execute(
            delete(ChunkEmbedding).where(
                ChunkEmbedding.chunk_id.in_(chunk_ids)
                | ChunkEmbedding.id.in_(embedding_ids)
            )
        )
        db.execute(
            delete(DocumentPermission).where(
                DocumentPermission.user_id.in_(test_user_ids)
            )
        )
        db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.id.in_(chunk_ids)
            )
        )
        db.execute(
            delete(DocumentPage).where(
                DocumentPage.id.in_(page_ids)
            )
        )
        db.execute(
            delete(DocumentProcessingJob).where(
                DocumentProcessingJob.document_version_id.in_(version_ids)
            )
        )
        db.execute(
            delete(DocumentVersion).where(
                DocumentVersion.id.in_(version_ids)
            )
        )
        db.execute(
            delete(Document).where(
                Document.id.in_(document_ids)
            )
        )
        db.execute(
            delete(User).where(
                User.id.in_(test_user_ids)
            )
        )
        db.commit()


def test_real_login_and_acl_restrict_qa_discovery(monkeypatch):
    _assert_isolated_target()

    user_id = uuid4()
    employee_id = uuid4()

    # Authorized vector has similarity 0.8 to the query.
    allowed = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        "Permitted Policy",
        ALLOWED_TEXT,
        _unit_vector(0.8, 0.6),
    )

    # Unauthorized vector is an exact query match (1.0).
    secret = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        "Secret Policy",
        SECRET_TEXT,
        _unit_vector(1.0, 0.0),
    )

    password = secrets.token_urlsafe(24)
    employee_password = secrets.token_urlsafe(24)
    password_hash = PasswordHash.recommended().hash(password)

    embedding = FixedEmbedding()
    planner = RecordingPlanner(allowed[3])
    evidence = RecordingEvidence()
    answer = DeterministicRefusal(evidence.result)
    memory_audit = MemoryAuditRepository()

    try:
        with SessionLocal() as db:
            # The dedicated database was empty after migration
            # and the prior retrieval security test's cleanup.
            assert db.scalar(
                select(func.count()).select_from(User)
            ) == 0

            assert db.scalar(
                select(func.count()).select_from(Document)
            ) == 0

            assert db.scalar(
                select(func.count()).select_from(ChunkEmbedding)
            ) == 0

            _seed(
                db,
                user_id=user_id,
                allowed=allowed,
                secret=secret,
                password_hash=password_hash,
            )

        # Give the generated test user the real ADMIN role.
        # This is needed to exercise the grant/revoke HTTP endpoints.
        with SessionLocal() as db:
            admin_role = db.scalar(
                select(Role).where(Role.name == "ADMIN")
            )

            if admin_role is None:
                admin_role = Role(
                    id=uuid4(),
                    name="ADMIN",
                    description="Isolated integration-test role",
                )
                db.add(admin_role)
                db.flush()

            admin_user = db.get(User, user_id)
            admin_user.roles.append(admin_role)

            employee_role = db.scalar(
                select(Role).where(Role.name == "EMPLOYEE")
            )
            if employee_role is None:
                employee_role = Role(
                    id=uuid4(),
                    name="EMPLOYEE",
                    description="Isolated integration-test role",
                )
                db.add(employee_role)
                db.flush()

            employee = User(
                id=employee_id,
                username=f"isolated-employee-{employee_id.hex}",
                display_name="Isolation Test Employee",
                email=None,
                department="Test",
                language_preference="fr",
                is_active=True,
                roles=[employee_role],
            )
            db.add(employee)
            db.flush()

            db.add(
                LocalCredential(
                    user_id=employee_id,
                    password_hash=PasswordHash.recommended().hash(
                        employee_password
                    ),
                )
            )
            db.commit()

        app = create_app()

        def override_qa_service(
            db: Annotated[Session, Depends(get_db)],
        ):
            auth_service = AuthorizationService(
                PostgresAuthorizationRepository(db)
            )

            real_search = AuthorizedSearchService(
                authorization=auth_service,
                embedding_provider=embedding,
                retrieval=PostgresVectorRetrievalAdapter(db),
            )

            return QuestionAnswerService(
                search=real_search,
                planner=planner,
                evidence=evidence,
                answer=answer,
                authorization=auth_service,
                audit=AuditService(memory_audit),
                discovery_top_k=5,
            )

        app.dependency_overrides[
            get_question_answer_service
        ] = override_qa_service

        # Deliberately DO NOT override:
        # get_current_identity, login, get_db,
        # AuthorizationService, or pgvector retrieval.

        prefix = get_settings().api_prefix
        question = {
            "question": "How many members?",
            "language": "fr",
        }

        login_url = f"{prefix}/auth/login"
        qa_url = f"{prefix}/qa"
        me_url = f"{prefix}/me"
        logout_url = f"{prefix}/auth/logout"
        permission_url = (
            f"{prefix}/documents/{allowed[0]}"
            f"/permissions/read/{user_id}"
        )
        source_page_url = f"{prefix}/source-pages/{allowed[2]}"

        with TestClient(app) as client:
            # 1. No token: reject before model work.
            anonymous = client.post(qa_url, json=question)
            assert anonymous.status_code == 401
            assert not planner.calls
            assert not embedding.calls

            # 2. Incorrect password: reject.
            bad_login = client.post(
                login_url,
                json={
                    "username": f"isolated-qa-{user_id.hex}",
                    "password": "incorrect-password",
                },
            )
            assert bad_login.status_code == 401

            # 3. Real login: session is stored in PostgreSQL.
            login = client.post(
                login_url,
                json={
                    "username": f"isolated-qa-{user_id.hex}",
                    "password": password,
                },
            )
            assert login.status_code == 200

            token = login.json()["access_token"]
            assert len(token) >= 30
            bearer = {"Authorization": f"Bearer {token}"}

            with SessionLocal() as db:
                sessions = db.scalars(
                    select(AuthSession).where(
                        AuthSession.user_id == user_id
                    )
                ).all()

                assert len(sessions) == 1
                assert sessions[0].token_hash == sha256(
                    token.encode()
                ).hexdigest()
                assert sessions[0].token_hash != token

            # 4. Real token resolves the correct identity.
            me = client.get(me_url, headers=bearer)
            assert me.status_code == 200
            assert me.json()["user_id"] == str(user_id)

            assert client.get(
                me_url,
                headers={
                    "Authorization": "Bearer definitely-invalid"
                },
            ).status_code == 401

            # 5. Valid session, but no READ permission:
            # no embedding, retrieval, or planning.
            empty = client.post(
                qa_url,
                json=question,
                headers=bearer,
            )

            assert empty.status_code == 200
            assert empty.json()["status"] == (
                "INSUFFICIENT_EVIDENCE"
            )
            assert empty.json()["citations"] == []
            assert not embedding.calls
            assert not planner.calls
            assert len(evidence.calls) == 1
            assert evidence.calls[0][1] == ()

            no_page_access = client.get(
                source_page_url,
                headers=bearer,
            )
            assert no_page_access.status_code == 404

            # Grant access through the real administrator API.
            grant = client.post(
                permission_url,
                headers=bearer,
            )

            assert grant.status_code == 200, grant.text
            assert grant.json()["changed"] is True
            assert grant.json()["permission"] == "READ"

            # 6. Strongest vector is unauthorized.
            # Only authorized evidence reaches planning.
            permitted_page = client.get(
                source_page_url,
                headers=bearer,
            )
            assert permitted_page.status_code == 200, permitted_page.text
            assert permitted_page.json()["page_id"] == str(allowed[2])
            assert permitted_page.json()["extracted_text"] == ALLOWED_TEXT

            permitted = client.post(
                qa_url,
                json=question,
                headers=bearer,
            )

            assert permitted.status_code == 200
            assert permitted.json()["status"] == (
                "INSUFFICIENT_EVIDENCE"
            )
            assert permitted.json()["citations"] == []

            assert len(embedding.calls) == 1
            assert len(planner.calls) == 1

            assert {
                item.chunk_id
                for item in planner.calls[0].evidence
            } == {allowed[3]}

            assert SECRET_TEXT not in repr(
                planner.calls[0].evidence
            )

            assert len(evidence.calls) == 2
            assert len(evidence.calls[1][1]) == 1
            assert evidence.calls[1][1][0].text == FACT

            # A second, authenticated employee cannot access the
            # document while the administrator's READ grant is active.
            employee_login = client.post(
                login_url,
                json={
                    "username": f"isolated-employee-{employee_id.hex}",
                    "password": employee_password,
                },
            )
            assert employee_login.status_code == 200, employee_login.text

            employee_bearer = {
                "Authorization": (
                    f"Bearer {employee_login.json()['access_token']}"
                )
            }

            employee_me = client.get(me_url, headers=employee_bearer)
            assert employee_me.status_code == 200
            assert employee_me.json()["user_id"] == str(employee_id)
            assert employee_me.json()["roles"] == ["EMPLOYEE"]

            # An employee cannot grant themselves READ access.
            employee_grant_attempt = client.post(
                (
                    f"{prefix}/documents/{allowed[0]}"
                    f"/permissions/read/{employee_id}"
                ),
                headers=employee_bearer,
            )
            assert employee_grant_attempt.status_code == 403, (
                employee_grant_attempt.text
            )

            employee_page = client.get(
                source_page_url,
                headers=employee_bearer,
            )
            assert employee_page.status_code == 404

            employee_qa = client.post(
                qa_url,
                json=question,
                headers=employee_bearer,
            )
            assert employee_qa.status_code == 200, employee_qa.text
            assert employee_qa.json()["status"] == "INSUFFICIENT_EVIDENCE"
            assert employee_qa.json()["citations"] == []
            assert FACT not in employee_qa.json()["answer"]
            assert SECRET_TEXT not in employee_qa.json()["answer"]

            # The employee's request must not reach embedding or planning.
            assert len(embedding.calls) == 1
            assert len(planner.calls) == 1
            assert len(evidence.calls) == 3
            assert evidence.calls[2][1] == ()

            # Exercise the retry HTTP endpoint without running an
            # actual embedding task in this authorization test.
            retry_job_id = uuid4()
            with SessionLocal() as db:
                db.add(
                    DocumentProcessingJob(
                        id=retry_job_id,
                        document_version_id=allowed[1],
                        status="SUCCEEDED",
                        parsing_status="SUCCESS",
                        attempt_number=1,
                        indexing_status="FAILED",
                        indexing_attempt_number=1,
                        indexing_error_code="INDEXING_FAILED",
                    )
                )
                db.commit()

            scheduled_retries = []

            def record_retry(job_id, version_id):
                scheduled_retries.append((job_id, version_id))

            monkeypatch.setattr(
                document_routes,
                "run_document_indexing_retry",
                record_retry,
            )

            retry_url = (
                f"{prefix}/documents/jobs/{retry_job_id}/retry-indexing"
            )

            # A non-admin cannot request a retry.
            employee_retry = client.post(
                retry_url,
                headers=employee_bearer,
            )
            assert employee_retry.status_code == 403, employee_retry.text
            assert scheduled_retries == []

            # The administrator claims the failed job.
            accepted_retry = client.post(
                retry_url,
                headers=bearer,
            )
            assert accepted_retry.status_code == 202, accepted_retry.text
            assert accepted_retry.json()["retry_accepted"] is True
            assert accepted_retry.json()["indexing_status"] == "INDEXING"
            assert scheduled_retries == [(retry_job_id, allowed[1])]

            # A second request must not schedule another retry.
            duplicate_retry = client.post(
                retry_url,
                headers=bearer,
            )
            assert duplicate_retry.status_code == 409
            assert scheduled_retries == [(retry_job_id, allowed[1])]

            with SessionLocal() as db:
                retry_job = db.get(DocumentProcessingJob, retry_job_id)
                assert retry_job.indexing_status == "INDEXING"
                assert retry_job.indexing_attempt_number == 2
                assert retry_job.indexing_error_code is None

            # 7. Revoking permission takes effect
            # for the SAME still-valid session.
            revoke = client.delete(
                permission_url,
                headers=bearer,
            )

            assert revoke.status_code == 200, revoke.text
            assert revoke.json()["changed"] is True
            assert revoke.json()["permission"] == "READ"

            revoked_acl = client.post(
                qa_url,
                json=question,
                headers=bearer,
            )

            assert revoked_acl.status_code == 200
            assert revoked_acl.json()["status"] == (
                "INSUFFICIENT_EVIDENCE"
            )
            assert revoked_acl.json()["citations"] == []

            assert len(planner.calls) == 1
            assert len(embedding.calls) == 1
            assert len(evidence.calls) == 4
            assert evidence.calls[3][1] == ()

            revoked_page = client.get(
                source_page_url,
                headers=bearer,
            )
            assert revoked_page.status_code == 404

            # 8. Logout invalidates the real session.
            logout = client.post(
                logout_url,
                headers=bearer,
            )

            assert logout.status_code == 204
            assert client.get(
                me_url,
                headers=bearer,
            ).status_code == 401
            assert client.post(
                qa_url,
                json=question,
                headers=bearer,
            ).status_code == 401

            assert len(planner.calls) == 1
            assert len(embedding.calls) == 1

        # QA used an in-memory audit repository.
        assert memory_audit.events == []

        with SessionLocal() as db:
            permission_events = db.scalars(
                select(AuditEventRecord).where(
                    AuditEventRecord.actor_user_id == user_id,
                    AuditEventRecord.document_id == allowed[0],
                    AuditEventRecord.event_type.in_(
                        ["DOCUMENT_READ_GRANTED", "DOCUMENT_READ_REVOKED"]
                    ),
                )
            ).all()

            assert sorted(
                event.event_type for event in permission_events
            ) == [
                "DOCUMENT_READ_GRANTED",
                "DOCUMENT_READ_REVOKED",
            ]
            assert all(
                event.outcome == "SUCCEEDED"
                for event in permission_events
            )

    finally:
        _cleanup(
            user_id=user_id,
            employee_id=employee_id,
            allowed=allowed,
            secret=secret,
        )

    with SessionLocal() as db:
        assert db.scalar(
            select(func.count()).select_from(User)
        ) == 0
        assert db.scalar(
            select(func.count()).select_from(Document)
        ) == 0
        assert db.scalar(
            select(func.count()).select_from(ChunkEmbedding)
        ) == 0
        assert db.scalar(
            select(func.count()).select_from(AuthSession)
        ) == 0


def test_indexing_failure_is_durable_after_parsing(monkeypatch):
    _assert_isolated_target()

    user_id = uuid4()
    employee_id = uuid4()
    job_id = uuid4()

    allowed = (
        uuid4(), uuid4(), uuid4(), uuid4(), uuid4(),
        "Indexing Failure Test", ALLOWED_TEXT,
        _unit_vector(0.8, 0.6),
    )
    secret = (
        uuid4(), uuid4(), uuid4(), uuid4(), uuid4(),
        "Second Test Document", SECRET_TEXT,
        _unit_vector(1.0, 0.0),
    )

    try:
        with SessionLocal() as db:
            assert db.scalar(
                select(func.count()).select_from(Document)
            ) == 0

            _seed(
                db,
                user_id=user_id,
                allowed=allowed,
                secret=secret,
                password_hash=PasswordHash.recommended().hash(
                    secrets.token_urlsafe(24)
                ),
            )

            # Leave the parsed chunk without an embedding.
            db.execute(
                delete(ChunkEmbedding).where(
                    ChunkEmbedding.id == allowed[4]
                )
            )

            # Represent a job whose parsing has already succeeded.
            db.add(
                DocumentProcessingJob(
                    id=job_id,
                    document_version_id=allowed[1],
                    status="SUCCEEDED",
                    parsing_status="SUCCESS",
                    attempt_number=1,
                )
            )
            db.commit()

        process_calls = []

        def fake_completed_parsing(self, request):
            process_calls.append(request.processing_job_id)
            return SimpleNamespace(
                version_id=allowed[1],
                chunk_count=1,
            )

        def failing_embedding_provider():
            raise RuntimeError("SIMULATED_EMBEDDING_PROVIDER_OUTAGE")

        # Exercise the worker's post-parsing handoff without
        # opening a PDF, loading a model, or writing a file.
        monkeypatch.setattr(
            processing_dependencies.DocumentProcessingService,
            "process",
            fake_completed_parsing,
        )
        monkeypatch.setattr(
            processing_dependencies,
            "LocalFileStorage",
            lambda _root: object(),
        )
        monkeypatch.setattr(
            qa_dependencies,
            "get_qa_embedding_provider",
            failing_embedding_provider,
        )

        processing_dependencies.run_document_processing_job(
            job_id,
            object(),
        )

        assert process_calls == [job_id]

        # A NEW session must see the committed failure state.
        with SessionLocal() as db:
            job = db.get(DocumentProcessingJob, job_id)
            version = db.get(DocumentVersion, allowed[1])

            assert job is not None
            assert version is not None
            assert job.status == "SUCCEEDED"
            assert job.parsing_status == "SUCCESS"
            assert version.processing_status == "SUCCEEDED"

            assert job.indexing_status == "FAILED"
            assert job.indexing_attempt_number == 1
            assert job.indexing_error_code == "INDEXING_FAILED"

            assert db.scalar(
                select(func.count())
                .select_from(ChunkEmbedding)
                .where(ChunkEmbedding.chunk_id == allowed[3])
            ) == 0

        # Claim the failed job without parsing the document again.
        claimed_version = processing_dependencies._claim_indexing_retry(job_id)
        assert claimed_version == allowed[1]

        # A job already in INDEXING cannot be claimed a second time.
        assert processing_dependencies._claim_indexing_retry(job_id) is None

        with SessionLocal() as db:
            job = db.get(DocumentProcessingJob, job_id)
            assert job.indexing_status == "INDEXING"
            assert job.indexing_attempt_number == 2
            assert job.indexing_error_code is None

        # The provider is still monkeypatched to fail.
        processing_dependencies.run_document_indexing_retry(
            job_id,
            allowed[1],
        )

        # Verify the failed retry from a fresh database session.
        with SessionLocal() as db:
            job = db.get(DocumentProcessingJob, job_id)
            assert job.status == "SUCCEEDED"
            assert job.parsing_status == "SUCCESS"
            assert job.indexing_status == "FAILED"
            assert job.indexing_attempt_number == 2
            assert job.indexing_error_code == "INDEXING_FAILED"

        # Neither retry should have invoked the parser.
        assert process_calls == [job_id]

        # The embedding provider recovers. Use deterministic vectors,
        # not a downloaded model or an external API.
        class WorkingEmbeddingProvider:
            model_name = MODEL_NAME
            model_revision = MODEL_REVISION
            dimension = EMBEDDING_DIMENSION

            def embed_passages(self, texts):
                assert list(texts) == [ALLOWED_TEXT]
                return tuple(
                    SimpleNamespace(
                        values=tuple(_unit_vector(0.8, 0.6)),
                        model_name=self.model_name,
                        model_revision=self.model_revision,
                        dimension=self.dimension,
                    )
                    for _ in texts
                )

        monkeypatch.setattr(
            qa_dependencies,
            "get_qa_embedding_provider",
            lambda: WorkingEmbeddingProvider(),
        )

        # Third attempt: reuse the existing parsed version.
        assert (
            processing_dependencies._claim_indexing_retry(job_id)
            == allowed[1]
        )
        processing_dependencies.run_document_indexing_retry(
            job_id,
            allowed[1],
        )

        # Verify the successful outcome and persisted embedding
        # using a fresh database session.
        with SessionLocal() as db:
            job = db.get(DocumentProcessingJob, job_id)
            version = db.get(DocumentVersion, allowed[1])

            assert job.indexing_status == "SUCCEEDED"
            assert job.indexing_attempt_number == 3
            assert job.indexing_error_code is None
            assert job.status == "SUCCEEDED"
            assert job.parsing_status == "SUCCESS"
            assert version.processing_status == "SUCCEEDED"

            embeddings = db.scalars(
                select(ChunkEmbedding).where(
                    ChunkEmbedding.chunk_id == allowed[3],
                    ChunkEmbedding.model_name == MODEL_NAME,
                    ChunkEmbedding.model_revision == MODEL_REVISION,
                )
            ).all()
            assert len(embeddings) == 1

        # No retry has invoked the parser.
        assert process_calls == [job_id]

    finally:
        _cleanup(
            user_id=user_id,
            employee_id=employee_id,
            allowed=allowed,
            secret=secret,
        )