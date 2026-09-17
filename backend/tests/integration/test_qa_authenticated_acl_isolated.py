
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

from app.api.dependencies.qa import get_question_answer_service
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
from app.infrastructure.database.models.identity import User
from app.infrastructure.database.models.permissions import DocumentPermission
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


def _cleanup(*, user_id, allowed, secret) -> None:
    """Delete only this test's UUIDs in foreign-key-safe order."""
    specs = (allowed, secret)

    document_ids = [item[0] for item in specs]
    version_ids = [item[1] for item in specs]
    page_ids = [item[2] for item in specs]
    chunk_ids = [item[3] for item in specs]
    embedding_ids = [item[4] for item in specs]

    with SessionLocal() as db:
        db.execute(
            delete(AuthSession).where(
                AuthSession.user_id == user_id
            )
        )
        db.execute(
            delete(LocalCredential).where(
                LocalCredential.user_id == user_id
            )
        )
        db.execute(
            delete(ChunkEmbedding).where(
                ChunkEmbedding.id.in_(embedding_ids)
            )
        )
        db.execute(
            delete(DocumentPermission).where(
                DocumentPermission.user_id == user_id
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
                User.id == user_id
            )
        )
        db.commit()


def test_real_login_and_acl_restrict_qa_discovery():
    _assert_isolated_target()

    user_id = uuid4()

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

        app = create_app()

        def override_qa_service(
            db: Annotated[Session, Depends(get_db)],
        ):
            real_search = AuthorizedSearchService(
                authorization=AuthorizationService(
                    PostgresAuthorizationRepository(db)
                ),
                embedding_provider=embedding,
                retrieval=PostgresVectorRetrievalAdapter(db),
            )

            return QuestionAnswerService(
                search=real_search,
                planner=planner,
                evidence=evidence,
                answer=answer,
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

            # Grant READ permission only for the weaker
            # authorized document.
            with SessionLocal() as db:
                db.add(
                    DocumentPermission(
                        id=uuid4(),
                        user_id=user_id,
                        document_id=allowed[0],
                        action=PermissionAction.READ.value,
                    )
                )
                db.commit()

            # 6. Strongest vector is unauthorized.
            # Only authorized evidence reaches planning.
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

            # 7. Revoking permission takes effect
            # for the SAME still-valid session.
            with SessionLocal() as db:
                db.execute(
                    delete(DocumentPermission).where(
                        DocumentPermission.user_id == user_id
                    )
                )
                db.commit()

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
            assert len(evidence.calls) == 3
            assert evidence.calls[2][1] == ()

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
            assert db.scalar(
                text("SELECT count(*) FROM audit_events")
            ) == 0

    finally:
        _cleanup(
            user_id=user_id,
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