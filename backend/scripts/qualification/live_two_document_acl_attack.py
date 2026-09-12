from __future__ import annotations

import os
from uuid import UUID

from sqlalchemy import select

# Qualification must remain offline and pinned.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.authorized_search import (
    AuthorizedSearchService,
)

from app.domain.enums.permission import (
    PermissionAction,
)

from app.infrastructure.database.models.documents import (
    Document,
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
from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


UNICEF_DOCUMENT_ID = UUID(
    "ba4f8e8c-f165-4940-9148-611fb488b653"
)

OXFAM_DOCUMENT_ID = UUID(
    "80895834-aced-41a9-91d5-5cf3108df0c1"
)

UNICEF_USERNAME = (
    "govknowledge-unicef-qualification"
)

OXFAM_USERNAME = (
    "govknowledge-oxfam-qualification"
)

BOTH_USERNAME = (
    "govknowledge-two-doc-qualification"
)


EMBEDDING_MODEL = (
    "intfloat/multilingual-e5-base"
)

EMBEDDING_REVISION = (
    "d128750597153bb5987e10b1c3493a34e5a4502a"
)


OXFAM_QUERY = (
    "What are Oxfam GB's five principles "
    "of partnership?"
)

UNICEF_QUERY = (
    "En cas d'urgence de niveau 3, "
    "EMOPS recommande une allocation de "
    "5 750 000 dollars."
)


def ensure_user(
    db,
    *,
    username: str,
    display_name: str,
    language: str,
) -> User:
    user = db.scalar(
        select(User).where(
            User.username == username
        )
    )

    if user is None:
        user = User(
            username=username,
            display_name=display_name,
            email=None,
            department="Qualification",
            language_preference=language,
            is_active=True,
        )

        db.add(user)
        db.flush()

        print(
            "Created user:",
            username,
            user.id,
        )

    else:
        print(
            "Reusing user:",
            username,
            user.id,
        )

    return user


def ensure_read_permission(
    db,
    *,
    user: User,
    document_id: UUID,
) -> None:
    permission = db.scalar(
        select(DocumentPermission).where(
            DocumentPermission.user_id
            == user.id,
            DocumentPermission.document_id
            == document_id,
            DocumentPermission.action
            == PermissionAction.READ.value,
        )
    )

    if permission is not None:
        return

    db.add(
        DocumentPermission(
            user_id=user.id,
            document_id=document_id,
            action=PermissionAction.READ.value,
        )
    )

    db.flush()


def identity_for(
    user: User,
) -> CurrentIdentity:
    return CurrentIdentity(
        user_id=user.id,
        username=user.username,
        display_name=user.display_name,
        email=user.email,
        department=user.department,
        language_preference=(
            user.language_preference
        ),
        roles=(),
    )


def print_results(
    title: str,
    results,
) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    for rank, chunk in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank:02d} "
            f"document={chunk.document_id} "
            f"page={chunk.page_number} "
            f"chunk={chunk.chunk_index} "
            f"score={chunk.score:.6f}"
        )

        preview = " ".join(
            chunk.content.split()
        )

        print(
            "   ",
            preview[:220],
        )


def assert_only(
    results,
    *,
    allowed: frozenset[UUID],
    label: str,
) -> None:
    leaked = tuple(
        chunk
        for chunk in results
        if chunk.document_id not in allowed
    )

    if leaked:
        raise RuntimeError(
            f"{label}: unauthorized document "
            "reached retrieval output."
        )


def main() -> None:
    print("=" * 72)
    print(
        "GOVKNOWLEDGE — TWO-DOCUMENT ACL ATTACK"
    )
    print("=" * 72)

    print()
    print(
        "Loading pinned multilingual E5..."
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider(
            model_name=EMBEDDING_MODEL,
            model_revision=EMBEDDING_REVISION,
        )
    )

    db = SessionLocal()

    try:
        # --------------------------------------------------
        # 1. Verify both real documents
        # --------------------------------------------------

        unicef_document = db.get(
            Document,
            UNICEF_DOCUMENT_ID,
        )

        oxfam_document = db.get(
            Document,
            OXFAM_DOCUMENT_ID,
        )

        if unicef_document is None:
            raise RuntimeError(
                "UNICEF document missing."
            )

        if oxfam_document is None:
            raise RuntimeError(
                "Oxfam document missing."
            )

        if unicef_document.status != "ACTIVE":
            raise RuntimeError(
                "UNICEF document is not ACTIVE."
            )

        if oxfam_document.status != "ACTIVE":
            raise RuntimeError(
                "Oxfam document is not ACTIVE."
            )

        print()
        print(
            "UNICEF:",
            UNICEF_DOCUMENT_ID,
        )

        print(
            "OXFAM :",
            OXFAM_DOCUMENT_ID,
        )

        # --------------------------------------------------
        # 2. Dedicated identities
        # --------------------------------------------------

        unicef_user = db.scalar(
            select(User).where(
                User.username
                == UNICEF_USERNAME
            )
        )

        if unicef_user is None:
            raise RuntimeError(
                "Existing UNICEF qualification "
                "user was not found."
            )

        oxfam_user = ensure_user(
            db,
            username=OXFAM_USERNAME,
            display_name=(
                "Oxfam Qualification User"
            ),
            language="en",
        )

        both_user = ensure_user(
            db,
            username=BOTH_USERNAME,
            display_name=(
                "Two Document Qualification User"
            ),
            language="en",
        )

        # IMPORTANT:
        # Never grant Oxfam to the UNICEF-only user.

        ensure_read_permission(
            db,
            user=oxfam_user,
            document_id=OXFAM_DOCUMENT_ID,
        )

        ensure_read_permission(
            db,
            user=both_user,
            document_id=UNICEF_DOCUMENT_ID,
        )

        ensure_read_permission(
            db,
            user=both_user,
            document_id=OXFAM_DOCUMENT_ID,
        )

        db.commit()

        # --------------------------------------------------
        # 3. Production authorization/search wiring
        # --------------------------------------------------

        authorization = AuthorizationService(
            PostgresAuthorizationRepository(
                db
            )
        )

        search = AuthorizedSearchService(
            authorization=authorization,
            embedding_provider=(
                embedding_provider
            ),
            retrieval=(
                PostgresVectorRetrievalAdapter(
                    db
                )
            ),
        )

        unicef_identity = identity_for(
            unicef_user
        )

        oxfam_identity = identity_for(
            oxfam_user
        )

        both_identity = identity_for(
            both_user
        )

        # --------------------------------------------------
        # 4. Assert exact ACL scopes
        # --------------------------------------------------

        unicef_scope = (
            authorization.retrieval_scope(
                unicef_identity
            )
        )

        oxfam_scope = (
            authorization.retrieval_scope(
                oxfam_identity
            )
        )

        both_scope = (
            authorization.retrieval_scope(
                both_identity
            )
        )

        print()
        print("=" * 72)
        print("AUTHORIZATION SCOPES")
        print("=" * 72)

        print(
            "UNICEF-only:",
            unicef_scope.document_ids,
        )

        print(
            "Oxfam-only :",
            oxfam_scope.document_ids,
        )

        print(
            "Both       :",
            both_scope.document_ids,
        )

        assert (
            unicef_scope.document_ids
            == frozenset(
                {UNICEF_DOCUMENT_ID}
            )
        )

        assert (
            oxfam_scope.document_ids
            == frozenset(
                {OXFAM_DOCUMENT_ID}
            )
        )

        assert (
            both_scope.document_ids
            == frozenset(
                {
                    UNICEF_DOCUMENT_ID,
                    OXFAM_DOCUMENT_ID,
                }
            )
        )

        # --------------------------------------------------
        # 5. BOTH-AUTHORIZED competition controls
        #
        # This proves each target corpus is actually
        # retrievable when access is permitted.
        # --------------------------------------------------

        both_oxfam = search.search(
            identity=both_identity,
            query=OXFAM_QUERY,
            top_k=10,
        )

        print_results(
            "BOTH AUTHORIZED — OXFAM QUERY",
            both_oxfam.results,
        )

        if not any(
            chunk.document_id
            == OXFAM_DOCUMENT_ID
            for chunk in both_oxfam.results
        ):
            raise RuntimeError(
                "Competition control failed: "
                "Oxfam did not surface when "
                "authorized."
            )

        both_unicef = search.search(
            identity=both_identity,
            query=UNICEF_QUERY,
            top_k=10,
        )

        print_results(
            "BOTH AUTHORIZED — UNICEF QUERY",
            both_unicef.results,
        )

        if not any(
            chunk.document_id
            == UNICEF_DOCUMENT_ID
            for chunk in both_unicef.results
        ):
            raise RuntimeError(
                "Competition control failed: "
                "UNICEF did not surface when "
                "authorized."
            )

        # --------------------------------------------------
        # 6. ATTACK A
        #
        # UNICEF-only user asks a highly specific
        # Oxfam question.
        # --------------------------------------------------

        attack_a = search.search(
            identity=unicef_identity,
            query=OXFAM_QUERY,
            top_k=10,
        )

        print_results(
            "ATTACK A — UNICEF USER / OXFAM QUERY",
            attack_a.results,
        )

        assert_only(
            attack_a.results,
            allowed=frozenset(
                {UNICEF_DOCUMENT_ID}
            ),
            label="ATTACK A",
        )

        if any(
            chunk.document_id
            == OXFAM_DOCUMENT_ID
            for chunk in attack_a.results
        ):
            raise RuntimeError(
                "ATTACK A FAILED: Oxfam leaked "
                "to UNICEF-only user."
            )

        # --------------------------------------------------
        # 7. ATTACK B
        #
        # Oxfam-only user asks a highly specific
        # UNICEF question.
        # --------------------------------------------------

        attack_b = search.search(
            identity=oxfam_identity,
            query=UNICEF_QUERY,
            top_k=10,
        )

        print_results(
            "ATTACK B — OXFAM USER / UNICEF QUERY",
            attack_b.results,
        )

        assert_only(
            attack_b.results,
            allowed=frozenset(
                {OXFAM_DOCUMENT_ID}
            ),
            label="ATTACK B",
        )

        if any(
            chunk.document_id
            == UNICEF_DOCUMENT_ID
            for chunk in attack_b.results
        ):
            raise RuntimeError(
                "ATTACK B FAILED: UNICEF leaked "
                "to Oxfam-only user."
            )

        # --------------------------------------------------
        # 8. Verdict
        # --------------------------------------------------

        print()
        print("=" * 72)
        print(
            "TWO-DOCUMENT ACL RETRIEVAL ATTACK: PASS"
        )
        print("=" * 72)

        print()
        print(
            "Both-authorized control:"
            " target corpora retrievable."
        )

        print(
            "UNICEF-only → Oxfam excluded: PASS"
        )

        print(
            "Oxfam-only → UNICEF excluded: PASS"
        )

        print(
            "Unauthorized document IDs reaching "
            "retrieval output: 0"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
