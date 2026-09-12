from uuid import uuid4

from sqlalchemy import delete, select

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.domain.enums.permission import (
    PermissionAction,
)
from app.domain.enums.processing import (
    ProcessingStatus,
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
from app.infrastructure.database.models.identity import (
    User,
)
from app.infrastructure.database.models.permissions import (
    DocumentPermission,
)
from app.infrastructure.database.repositories.authorization import (
    PostgresAuthorizationRepository,
)
from app.infrastructure.database.session import (
    SessionLocal,
)
from app.infrastructure.retrieval.postgres_vector_retrieval import (
    PostgresVectorRetrievalAdapter,
)


MODEL_NAME = "security-test-model"
MODEL_REVISION = "security-test-revision"


def make_unit_vector(
    first: float,
    second: float,
) -> list[float]:
    return [
        first,
        second,
        *(
            [0.0]
            * (
                EMBEDDING_DIMENSION
                - 2
            )
        ),
    ]


def test_unauthorized_strongest_vector_is_never_retrieved():
    db = SessionLocal()

    user_id = uuid4()

    allowed_document_id = uuid4()
    allowed_version_id = uuid4()
    allowed_page_id = uuid4()
    allowed_chunk_id = uuid4()
    allowed_embedding_id = uuid4()

    secret_document_id = uuid4()
    secret_version_id = uuid4()
    secret_page_id = uuid4()
    secret_chunk_id = uuid4()
    secret_embedding_id = uuid4()

    archived_document_id = uuid4()
    archived_version_id = uuid4()
    archived_page_id = uuid4()
    archived_chunk_id = uuid4()
    archived_embedding_id = uuid4()

    archived_permission_id = uuid4()

    permission_id = uuid4()

    try:
        # -----------------------------------------
        # Real database identity.
        # -----------------------------------------
        db.add(
            User(
                id=user_id,
                username=(
                    f"security-user-{user_id}"
                ),
                display_name=(
                    "Security Test Employee"
                ),
                email=None,
                department="Security Test",
                language_preference="fr",
                is_active=True,
            )
        )

        # -----------------------------------------
        # Authorized document.
        #
        # Its vector has cosine similarity 0.8
        # with the test query.
        # -----------------------------------------
        db.add(
            Document(
                id=allowed_document_id,
                title="Authorized Policy",
                description=(
                    "Authorized security test "
                    "document."
                ),
                document_type="POLICY",
                issuing_organization=(
                    "GovKnowledge Test"
                ),
                jurisdiction="Test",
                language="en",
                publication_date=None,
                source_url=None,
                status="ACTIVE",
                confidentiality_level="PUBLIC",
            )
        )

        db.add(
            DocumentVersion(
                id=allowed_version_id,
                document_id=allowed_document_id,
                version_number=1,
                original_filename=(
                    "authorized.pdf"
                ),
                storage_path=(
                    "security/authorized.pdf"
                ),
                file_hash="a" * 64,
                processing_status=(
                    ProcessingStatus.SUCCEEDED.value
                ),
                effective_date=None,
                superseded_at=None,
            )
        )

        db.add(
            DocumentPage(
                id=allowed_page_id,
                document_version_id=(
                    allowed_version_id
                ),
                page_number=1,
                extracted_text=(
                    "Authorized policy evidence."
                ),
                metadata_json={},
            )
        )

        db.add(
            DocumentChunk(
                id=allowed_chunk_id,
                document_id=allowed_document_id,
                document_version_id=(
                    allowed_version_id
                ),
                page_id=allowed_page_id,
                page_start=1,
                page_end=1,
                chunk_index=0,
                section_title=None,
                content=(
                    "AUTHORIZED POLICY: "
                    "ordinary permitted evidence."
                ),
                language="en",
                metadata_json={},
            )
        )

        # Persist the document graph before
        # inserting an embedding that references
        # the canonical chunk by FK.
        db.flush()

        # Unit vector:
        # 0.8^2 + 0.6^2 = 1.
        db.add(
            ChunkEmbedding(
                id=allowed_embedding_id,
                chunk_id=allowed_chunk_id,
                model_name=MODEL_NAME,
                model_revision=MODEL_REVISION,
                dimension=EMBEDDING_DIMENSION,
                normalized=True,
                embedding=make_unit_vector(
                    0.8,
                    0.6,
                ),
            )
        )

        # -----------------------------------------
        # Secret document.
        #
        # Exact query-vector match => similarity 1.
        # This MUST still be excluded.
        # -----------------------------------------
        db.add(
            Document(
                id=secret_document_id,
                title="Secret Policy",
                description=(
                    "Unauthorized security test "
                    "document."
                ),
                document_type="POLICY",
                issuing_organization=(
                    "GovKnowledge Test"
                ),
                jurisdiction="Test",
                language="en",
                publication_date=None,
                source_url=None,
                status="ACTIVE",
                confidentiality_level="SECRET",
            )
        )

        db.add(
            DocumentVersion(
                id=secret_version_id,
                document_id=secret_document_id,
                version_number=1,
                original_filename="secret.pdf",
                storage_path="security/secret.pdf",
                file_hash="b" * 64,
                processing_status=(
                    ProcessingStatus.SUCCEEDED.value
                ),
                effective_date=None,
                superseded_at=None,
            )
        )

        db.add(
            DocumentPage(
                id=secret_page_id,
                document_version_id=(
                    secret_version_id
                ),
                page_number=1,
                extracted_text=(
                    "Highly relevant secret "
                    "policy evidence."
                ),
                metadata_json={},
            )
        )

        db.add(
            DocumentChunk(
                id=secret_chunk_id,
                document_id=secret_document_id,
                document_version_id=(
                    secret_version_id
                ),
                page_id=secret_page_id,
                page_start=1,
                page_end=1,
                chunk_index=0,
                section_title=None,
                content=(
                    "SECRET POLICY: this is the "
                    "mathematically strongest match."
                ),
                language="en",
                metadata_json={},
            )
        )

        # Persist the secret document graph
        # before inserting its embedding FK.
        db.flush()

        db.add(
            ChunkEmbedding(
                id=secret_embedding_id,
                chunk_id=secret_chunk_id,
                model_name=MODEL_NAME,
                model_revision=MODEL_REVISION,
                dimension=EMBEDDING_DIMENSION,
                normalized=True,
                embedding=make_unit_vector(
                    1.0,
                    0.0,
                ),
            )
        )

        # -----------------------------------------
        # Archived document.
        #
        # The employee DOES have READ permission,
        # and its vector is an exact query match.
        #
        # It must still be excluded because
        # archived documents are not active
        # knowledge.
        # -----------------------------------------
        db.add(
            Document(
                id=archived_document_id,
                title="Archived Policy",
                description=(
                    "Archived security test "
                    "document."
                ),
                document_type="POLICY",
                issuing_organization=(
                    "GovKnowledge Test"
                ),
                jurisdiction="Test",
                language="en",
                publication_date=None,
                source_url=None,
                status="ARCHIVED",
                confidentiality_level="PUBLIC",
            )
        )

        db.add(
            DocumentVersion(
                id=archived_version_id,
                document_id=archived_document_id,
                version_number=1,
                original_filename="archived.pdf",
                storage_path="security/archived.pdf",
                file_hash="c" * 64,
                processing_status=(
                    ProcessingStatus.SUCCEEDED.value
                ),
                effective_date=None,
                superseded_at=None,
            )
        )

        db.add(
            DocumentPage(
                id=archived_page_id,
                document_version_id=(
                    archived_version_id
                ),
                page_number=1,
                extracted_text=(
                    "Archived policy evidence."
                ),
                metadata_json={},
            )
        )

        db.add(
            DocumentChunk(
                id=archived_chunk_id,
                document_id=archived_document_id,
                document_version_id=(
                    archived_version_id
                ),
                page_id=archived_page_id,
                page_start=1,
                page_end=1,
                chunk_index=0,
                section_title=None,
                content=(
                    "ARCHIVED POLICY: this is a "
                    "perfect vector match."
                ),
                language="en",
                metadata_json={},
            )
        )

        db.flush()

        db.add(
            ChunkEmbedding(
                id=archived_embedding_id,
                chunk_id=archived_chunk_id,
                model_name=MODEL_NAME,
                model_revision=MODEL_REVISION,
                dimension=EMBEDDING_DIMENSION,
                normalized=True,
                embedding=make_unit_vector(
                    1.0,
                    0.0,
                ),
            )
        )

        # The employee gets READ permission for
        # both the active allowed document and
        # the archived document.
        #
        # The SECRET document remains unauthorized.
        db.add(
            DocumentPermission(
                id=permission_id,
                user_id=user_id,
                document_id=(
                    allowed_document_id
                ),
                action=PermissionAction.READ.value,
            )
        )

        db.add(
            DocumentPermission(
                id=archived_permission_id,
                user_id=user_id,
                document_id=(
                    archived_document_id
                ),
                action=PermissionAction.READ.value,
            )
        )

        db.commit()

        query_vector = tuple(
            make_unit_vector(
                1.0,
                0.0,
            )
        )

        # -----------------------------------------
        # Control proof:
        # without ACL filtering, the SECRET
        # vector really is the strongest result.
        # -----------------------------------------
        distance = (
            ChunkEmbedding.embedding
            .cosine_distance(
                list(query_vector)
            )
        )

        unrestricted = db.execute(
            select(
                DocumentChunk.document_id,
                DocumentChunk.id,
                distance.label("distance"),
            )
            .join(
                ChunkEmbedding,
                ChunkEmbedding.chunk_id
                == DocumentChunk.id,
            )
            .where(
                ChunkEmbedding.model_name
                == MODEL_NAME,
                ChunkEmbedding.model_revision
                == MODEL_REVISION,
                DocumentChunk.document_id.in_(
                    (
                        allowed_document_id,
                        secret_document_id,
                    )
                ),
            )
            .order_by(
                distance
            )
        ).all()

        assert len(unrestricted) == 2

        assert (
            unrestricted[0].document_id
            == secret_document_id
        )

        assert (
            unrestricted[0].id
            == secret_chunk_id
        )

        assert (
            float(
                unrestricted[0].distance
            )
            < float(
                unrestricted[1].distance
            )
        )

        archived_control = db.execute(
            select(
                DocumentChunk.document_id,
                DocumentChunk.id,
                distance.label("distance"),
            )
            .join(
                ChunkEmbedding,
                ChunkEmbedding.chunk_id
                == DocumentChunk.id,
            )
            .where(
                ChunkEmbedding.model_name
                == MODEL_NAME,
                ChunkEmbedding.model_revision
                == MODEL_REVISION,
                DocumentChunk.document_id.in_(
                    (
                        allowed_document_id,
                        archived_document_id,
                    )
                ),
            )
            .order_by(
                distance
            )
        ).all()

        assert len(archived_control) == 2

        assert (
            archived_control[0].document_id
            == archived_document_id
        )

        assert (
            archived_control[0].id
            == archived_chunk_id
        )

        db.rollback()

        # -----------------------------------------
        # Real GovKnowledge authorization scope.
        # -----------------------------------------
        identity = CurrentIdentity(
            user_id=user_id,
            username="security-test-user",
            display_name=(
                "Security Test Employee"
            ),
            email=None,
            department="Security Test",
            language_preference="fr",
            roles=("EMPLOYEE",),
        )

        authorization = AuthorizationService(
            PostgresAuthorizationRepository(
                db
            )
        )

        scope = authorization.retrieval_scope(
            identity
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

        assert not scope.allows(
            archived_document_id
        )

        # -----------------------------------------
        # Real pgvector retrieval adapter.
        # -----------------------------------------
        retrieval = (
            PostgresVectorRetrievalAdapter(
                db
            )
        )

        results = retrieval.search_authorized(
            scope=scope,
            query_vector=query_vector,
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            top_k=1,
        )

        assert len(results) == 1

        assert (
            results[0].document_id
            == allowed_document_id
        )

        assert (
            results[0].chunk_id
            == allowed_chunk_id
        )

        assert results[0].score > 0.79

        assert all(
            result.document_id
            != secret_document_id
            for result in results
        )

        assert all(
            result.chunk_id
            != secret_chunk_id
            for result in results
        )

        # -----------------------------------------
        # Defense-in-depth attack:
        #
        # Fabricate a stale/malformed scope that
        # incorrectly includes an archived document.
        #
        # The retrieval adapter must independently
        # enforce Document.status == ACTIVE.
        # -----------------------------------------
        stale_scope = AuthorizationScope(
            user_id=user_id,
            action=PermissionAction.READ,
            document_ids=frozenset(
                {
                    allowed_document_id,
                    archived_document_id,
                }
            ),
        )

        stale_scope_results = (
            retrieval.search_authorized(
                scope=stale_scope,
                query_vector=query_vector,
                model_name=MODEL_NAME,
                model_revision=MODEL_REVISION,
                top_k=1,
            )
        )

        assert len(stale_scope_results) == 1

        assert (
            stale_scope_results[0].document_id
            == allowed_document_id
        )

        assert (
            stale_scope_results[0].chunk_id
            == allowed_chunk_id
        )

        assert all(
            result.document_id
            != archived_document_id
            for result in stale_scope_results
        )

        assert all(
            result.chunk_id
            != archived_chunk_id
            for result in stale_scope_results
        )

        # -----------------------------------------
        # Non-SUCCEEDED version defense.
        #
        # The document remains ACTIVE and the user
        # remains authorized, but its version is now
        # FAILED. Its persisted embedding must no
        # longer be retrievable.
        # -----------------------------------------
        allowed_version = db.get(
            DocumentVersion,
            allowed_version_id,
        )

        assert allowed_version is not None

        allowed_version.processing_status = (
            ProcessingStatus.FAILED.value
        )

        db.flush()

        failed_version_results = (
            retrieval.search_authorized(
                scope=scope,
                query_vector=query_vector,
                model_name=MODEL_NAME,
                model_revision=MODEL_REVISION,
                top_k=1,
            )
        )

        assert failed_version_results == ()

    finally:
        db.rollback()

        db.execute(
            delete(
                ChunkEmbedding
            ).where(
                ChunkEmbedding.id.in_(
                    (
                        allowed_embedding_id,
                        secret_embedding_id,
                        archived_embedding_id,
                    )
                )
            )
        )

        db.execute(
            delete(
                DocumentPermission
            ).where(
                DocumentPermission.id.in_(
                    (
                        permission_id,
                        archived_permission_id,
                    )
                )
            )
        )

        db.execute(
            delete(
                DocumentChunk
            ).where(
                DocumentChunk.id.in_(
                    (
                        allowed_chunk_id,
                        secret_chunk_id,
                        archived_chunk_id,
                    )
                )
            )
        )

        db.execute(
            delete(
                DocumentPage
            ).where(
                DocumentPage.id.in_(
                    (
                        allowed_page_id,
                        secret_page_id,
                        archived_page_id,
                    )
                )
            )
        )

        db.execute(
            delete(
                DocumentVersion
            ).where(
                DocumentVersion.id.in_(
                    (
                        allowed_version_id,
                        secret_version_id,
                        archived_version_id,
                    )
                )
            )
        )

        db.execute(
            delete(
                Document
            ).where(
                Document.id.in_(
                    (
                        allowed_document_id,
                        secret_document_id,
                        archived_document_id,
                    )
                )
            )
        )

        db.execute(
            delete(
                User
            ).where(
                User.id == user_id
            )
        )

        db.commit()
        db.close()
