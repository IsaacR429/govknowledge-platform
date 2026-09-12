from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pprint import pprint
from uuid import UUID

from sqlalchemy import select

# Force the qualification to use already-cached,
# pinned model artifacts only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.application.ports.identity_provider import CurrentIdentity

from app.application.services.audit import AuditService
from app.application.services.authorization import AuthorizationService
from app.application.services.authorized_search import AuthorizedSearchService
from app.application.services.atomic_evidence_orchestration import (
    AtomicClaim,
    AtomicEvidenceOrchestrationService,
)
from app.application.services.atomic_evidence_policy import (
    AtomicEvidencePolicy,
)
from app.application.services.sentence_evidence_verification import (
    SentenceEvidenceVerificationService,
)

from app.domain.enums.permission import PermissionAction

from app.infrastructure.database.models.documents import Document
from app.infrastructure.database.models.identity import User
from app.infrastructure.database.models.permissions import DocumentPermission

from app.infrastructure.database.repositories.audit import (
    PostgresAuditRepository,
)
from app.infrastructure.database.repositories.authorization import (
    PostgresAuthorizationRepository,
)

from app.infrastructure.database.session import SessionLocal

from app.infrastructure.retrieval.postgres_vector_retrieval import (
    PostgresVectorRetrievalAdapter,
)
from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)

from app.infrastructure.verification.mdeberta_entailment_verifier import (
    MDebertaEntailmentVerifier,
)


UNICEF_DOCUMENT_ID = UUID(
    "ba4f8e8c-f165-4940-9148-611fb488b653"
)

USERNAME = "govknowledge-unicef-qualification"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_REVISION = (
    "d128750597153bb5987e10b1c3493a34e5a4502a"
)

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NLI_REVISION = (
    "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)


CLAIMS = (
    AtomicClaim(
        claim_id="level3-total-allocation",
        text=(
            "En cas d'urgence de niveau 3, "
            "EMOPS recommande une allocation de "
            "5 750 000 dollars."
        ),
        required=True,
    ),
    AtomicClaim(
        claim_id="level3-peas-allocation",
        text=(
            "750 000 dollars de cette allocation "
            "sont affectés à la PEAS."
        ),
        required=True,
    ),
)


def get_attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def print_retrieved_chunk(index, chunk) -> None:
    document_id = get_attr(
        chunk,
        "document_id",
    )

    chunk_id = get_attr(
        chunk,
        "chunk_id",
        "id",
    )

    page = get_attr(
        chunk,
        "page_number",
        "page_start",
    )

    page_end = get_attr(
        chunk,
        "page_end",
    )

    score = get_attr(
        chunk,
        "score",
        "similarity_score",
        "cosine_similarity",
    )

    content = get_attr(
        chunk,
        "content",
        "text",
    )

    print()
    print(f"  RESULT {index}")
    print(f"  document_id : {document_id}")
    print(f"  chunk_id    : {chunk_id}")
    print(f"  page        : {page}")
    if page_end is not None:
        print(f"  page_end    : {page_end}")
    print(f"  score       : {score}")

    if content:
        print("  content:")
        print("  " + "-" * 60)
        print(str(content)[:1200])
        print("  " + "-" * 60)


def print_dataclass(title: str, value) -> None:
    print()
    print(title)
    print("-" * len(title))

    if is_dataclass(value):
        pprint(
            asdict(value),
            width=120,
            sort_dicts=False,
        )
    else:
        pprint(
            value,
            width=120,
            sort_dicts=False,
        )


def main() -> None:
    print("=" * 72)
    print("GOVKNOWLEDGE — UNICEF LIVE EVIDENCE CONTROL")
    print("=" * 72)

    print()
    print("Loading pinned embedding model...")
    embedding_provider = (
        SentenceTransformerEmbeddingProvider(
            model_name=EMBEDDING_MODEL,
            model_revision=EMBEDDING_REVISION,
        )
    )

    print("Loading pinned NLI model...")
    entailment_verifier = (
        MDebertaEntailmentVerifier(
            model_name=NLI_MODEL,
            model_revision=NLI_REVISION,
            local_files_only=True,
        )
    )

    db = SessionLocal()

    try:
        # --------------------------------------------------
        # 1. Verify target document
        # --------------------------------------------------

        document = db.get(
            Document,
            UNICEF_DOCUMENT_ID,
        )

        if document is None:
            raise RuntimeError(
                "UNICEF qualification document "
                "does not exist in the database."
            )

        print()
        print(
            "Target document:",
            UNICEF_DOCUMENT_ID,
        )
        print(
            "Document status:",
            document.status,
        )

        if document.status != "ACTIVE":
            raise RuntimeError(
                "UNICEF document is not ACTIVE."
            )

        # --------------------------------------------------
        # 2. Create/reuse dedicated qualification user
        # --------------------------------------------------

        user = db.scalar(
            select(User).where(
                User.username == USERNAME
            )
        )

        if user is None:
            user = User(
                username=USERNAME,
                display_name=(
                    "UNICEF Qualification User"
                ),
                email=None,
                department="Qualification",
                language_preference="fr",
                is_active=True,
            )

            db.add(user)
            db.flush()

            print()
            print(
                "Created qualification user:",
                user.id,
            )
        else:
            print()
            print(
                "Reusing qualification user:",
                user.id,
            )

        # --------------------------------------------------
        # 3. Grant explicit READ permission to UNICEF only
        # --------------------------------------------------

        permission = db.scalar(
            select(DocumentPermission).where(
                DocumentPermission.user_id
                == user.id,
                DocumentPermission.document_id
                == UNICEF_DOCUMENT_ID,
                DocumentPermission.action
                == PermissionAction.READ.value,
            )
        )

        if permission is None:
            permission = DocumentPermission(
                user_id=user.id,
                document_id=UNICEF_DOCUMENT_ID,
                action=PermissionAction.READ.value,
            )

            db.add(permission)
            db.flush()

            print(
                "Created UNICEF READ permission."
            )
        else:
            print(
                "UNICEF READ permission already exists."
            )

        db.commit()

        # --------------------------------------------------
        # 4. Build real application identity
        # --------------------------------------------------

        identity = CurrentIdentity(
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

        # --------------------------------------------------
        # 5. Wire real production services
        # --------------------------------------------------

        authorization_repository = (
            PostgresAuthorizationRepository(db)
        )

        authorization = AuthorizationService(
            authorization_repository
        )

        retrieval = PostgresVectorRetrievalAdapter(
            db
        )

        search = AuthorizedSearchService(
            authorization=authorization,
            embedding_provider=embedding_provider,
            retrieval=retrieval,
        )

        verification = (
            SentenceEvidenceVerificationService(
                verifier=entailment_verifier
            )
        )

        policy = AtomicEvidencePolicy()

        audit_repository = (
            PostgresAuditRepository(db)
        )

        audit = AuditService(
            audit_repository
        )

        orchestration = (
            AtomicEvidenceOrchestrationService(
                search=search,
                verification=verification,
                policy=policy,
                audit=audit,
                top_k=5,
            )
        )

        # --------------------------------------------------
        # 6. Verify authorization scope BEFORE query
        # --------------------------------------------------

        scope = authorization.retrieval_scope(
            identity
        )

        print()
        print("=" * 72)
        print("AUTHORIZATION SCOPE")
        print("=" * 72)

        print(
            "Allowed document IDs:",
            scope.document_ids,
        )

        if UNICEF_DOCUMENT_ID not in (
            scope.document_ids
        ):
            raise RuntimeError(
                "UNICEF document is missing from "
                "the authorization scope."
            )

        # Dedicated qualification user should
        # currently have UNICEF only.
        if scope.document_ids != frozenset(
            {UNICEF_DOCUMENT_ID}
        ):
            raise RuntimeError(
                "Qualification user has unexpected "
                "additional document permissions: "
                f"{scope.document_ids}"
            )

        print(
            "Authorization scope: PASS "
            "(UNICEF only)"
        )

        # --------------------------------------------------
        # 7. Execute REAL evidence pipeline
        # --------------------------------------------------

        print()
        print("=" * 72)
        print("RUNNING ATOMIC EVIDENCE PIPELINE")
        print("=" * 72)

        result = orchestration.evaluate(
            identity=identity,
            claims=CLAIMS,
        )

        problems: list[str] = []

        # --------------------------------------------------
        # 8. Inspect retrieval + verification
        # --------------------------------------------------

        for claim_result in (
            result.claim_results
        ):
            print()
            print("=" * 72)
            print(
                "CLAIM:",
                claim_result.claim.claim_id,
            )
            print("=" * 72)

            print(
                claim_result.claim.text
            )

            retrieved = (
                claim_result.retrieved_evidence
            )

            print()
            print(
                "Retrieved chunks:",
                len(retrieved),
            )

            pages = []

            for index, chunk in enumerate(
                retrieved,
                start=1,
            ):
                print_retrieved_chunk(
                    index,
                    chunk,
                )

                document_id = get_attr(
                    chunk,
                    "document_id",
                )

                if (
                    document_id
                    != UNICEF_DOCUMENT_ID
                ):
                    problems.append(
                        f"{claim_result.claim.claim_id}: "
                        "unauthorized/non-UNICEF "
                        f"document retrieved: {document_id}"
                    )

                page = get_attr(
                    chunk,
                    "page_number",
                    "page_start",
                )

                if page is not None:
                    pages.append(page)

            print()
            print(
                "Retrieved pages:",
                pages,
            )

            if 8 not in pages:
                problems.append(
                    f"{claim_result.claim.claim_id}: "
                    "page 8 was not retrieved."
                )

            print_dataclass(
                "VERIFICATION RESULT",
                claim_result.verification,
            )

        # --------------------------------------------------
        # 9. Final evidence-policy decision
        # --------------------------------------------------

        print()
        print("=" * 72)
        print("FINAL EVIDENCE ASSESSMENT")
        print("=" * 72)

        print_dataclass(
            "ASSESSMENT",
            result.assessment,
        )

        print()
        print(
            "is_sufficient:",
            result.is_sufficient,
        )
        print(
            "correlation_id:",
            result.correlation_id,
        )

        if not result.is_sufficient:
            problems.append(
                "Atomic evidence policy returned "
                "INSUFFICIENT."
            )

        # --------------------------------------------------
        # 10. Qualification verdict
        # --------------------------------------------------

        print()
        print("=" * 72)

        if problems:
            print(
                "UNICEF LIVE EVIDENCE CONTROL: FAIL"
            )
            print("=" * 72)

            for problem in problems:
                print(
                    " -",
                    problem,
                )

            raise SystemExit(1)

        print(
            "UNICEF LIVE EVIDENCE CONTROL: PASS"
        )
        print("=" * 72)

    finally:
        db.close()


if __name__ == "__main__":
    main()
