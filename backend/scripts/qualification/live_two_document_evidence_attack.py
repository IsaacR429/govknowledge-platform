from __future__ import annotations

import os
from uuid import UUID

from sqlalchemy import select

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault(
    "TRANSFORMERS_OFFLINE",
    "1",
)

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.audit import (
    AuditService,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.authorized_search import (
    AuthorizedSearchService,
)
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

from app.infrastructure.database.models.identity import (
    User,
)
from app.infrastructure.database.repositories.audit import (
    PostgresAuditRepository,
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
from app.infrastructure.verification.mdeberta_entailment_verifier import (
    MDebertaEntailmentVerifier,
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

NLI_MODEL = (
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

NLI_REVISION = (
    "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)


OXFAM_CLAIM = AtomicClaim(
    claim_id="oxfam-partnership-policy",
    text=(
        "Oxfam GB's partner relations are "
        "informed by and managed to a set "
        "of clear principles."
    ),
    language="en",
    required=True,
)

UNICEF_CLAIM = AtomicClaim(
    claim_id="unicef-level3-allocation",
    text=(
        "En cas d'urgence de niveau 3, "
        "EMOPS recommande une allocation "
        "de 5 750 000 dollars."
    ),
    language="fr",
    required=True,
)


def get_user(
    db,
    username: str,
) -> User:
    user = db.scalar(
        select(User).where(
            User.username == username
        )
    )

    if user is None:
        raise RuntimeError(
            f"Missing qualification user: "
            f"{username}"
        )

    return user


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


def run_case(
    *,
    title: str,
    orchestration,
    identity,
    claim: AtomicClaim,
    allowed_documents: frozenset[UUID],
    expected_document: UUID | None,
    expect_sufficient: bool,
) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    result = orchestration.evaluate(
        identity=identity,
        claims=(claim,),
    )

    retrieved_ids = {
        chunk.document_id
        for claim_result in result.claim_results
        for chunk in (
            claim_result.retrieved_evidence
        )
    }

    nli_ids = {
        span.chunk.document_id
        for claim_result in result.claim_results
        for span in (
            claim_result.verification.spans
        )
    }

    selected_ids = {
        selected.span.chunk.document_id
        for selected in (
            result.assessment.selected_evidence
        )
    }

    print(
        "Retrieved document IDs:",
        retrieved_ids,
    )

    print(
        "NLI span document IDs:",
        nli_ids,
    )

    print(
        "Selected evidence IDs:",
        selected_ids,
    )

    print(
        "Decision:",
        result.assessment.decision,
    )

    print(
        "is_sufficient:",
        result.is_sufficient,
    )

    if not expect_sufficient:
        print()
        print("Retrieved chunks actually sent to NLI:")

        for claim_result in result.claim_results:
            for index, chunk in enumerate(
                claim_result.retrieved_evidence,
                start=1,
            ):
                preview = " ".join(
                    chunk.content.split()
                )

                print(
                    f"  [{index}] "
                    f"document={chunk.document_id} "
                    f"page={chunk.page_number} "
                    f"chunk={chunk.chunk_index} "
                    f"score={chunk.score:.6f}"
                )
                print(
                    "      ",
                    preview[:500],
                )

        print()
        print("NLI sentence spans:")

        for claim_result in result.claim_results:
            for index, span in enumerate(
                claim_result.verification.spans,
                start=1,
            ):
                print(
                    f"  [{index}] "
                    f"document={span.chunk.document_id} "
                    f"page={span.chunk.page_number} "
                    f"chunk={span.chunk.chunk_index}"
                )
                print(
                    "      ",
                    span.text[:500],
                )
                print(
                    "      label=",
                    span.verification.label,
                    " entailment=",
                    round(
                        span.verification
                        .entailment_probability,
                        6,
                    ),
                )

    if not retrieved_ids.issubset(
        allowed_documents
    ):
        raise RuntimeError(
            f"{title}: unauthorized document "
            "reached retrieval."
        )

    if not nli_ids.issubset(
        allowed_documents
    ):
        raise RuntimeError(
            f"{title}: unauthorized document "
            "reached NLI."
        )

    if not selected_ids.issubset(
        allowed_documents
    ):
        raise RuntimeError(
            f"{title}: unauthorized document "
            "reached selected evidence."
        )

    if expect_sufficient:
        if not result.is_sufficient:
            raise RuntimeError(
                f"{title}: expected SUFFICIENT."
            )

        if selected_ids != {
            expected_document
        }:
            raise RuntimeError(
                f"{title}: selected evidence "
                "has incorrect provenance."
            )

    else:
        if result.is_sufficient:
            raise RuntimeError(
                f"{title}: restricted claim "
                "was incorrectly supported."
            )

        if selected_ids:
            raise RuntimeError(
                f"{title}: restricted attack "
                "produced selected evidence."
            )

    print("CASE: PASS")


def main() -> None:
    print("=" * 72)
    print(
        "GOVKNOWLEDGE — TWO-DOCUMENT "
        "EVIDENCE ATTACK"
    )
    print("=" * 72)

    print()
    print("Loading pinned E5...")

    embedding_provider = (
        SentenceTransformerEmbeddingProvider(
            model_name=EMBEDDING_MODEL,
            model_revision=EMBEDDING_REVISION,
        )
    )

    print("Loading pinned NLI...")

    verifier = MDebertaEntailmentVerifier(
        model_name=NLI_MODEL,
        model_revision=NLI_REVISION,
        local_files_only=True,
    )

    db = SessionLocal()

    try:
        unicef_user = get_user(
            db,
            UNICEF_USERNAME,
        )

        oxfam_user = get_user(
            db,
            OXFAM_USERNAME,
        )

        both_user = get_user(
            db,
            BOTH_USERNAME,
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

        verification = (
            SentenceEvidenceVerificationService(
                verifier=verifier
            )
        )

        audit = AuditService(
            PostgresAuditRepository(
                db
            )
        )

        orchestration = (
            AtomicEvidenceOrchestrationService(
                search=search,
                verification=verification,
                policy=AtomicEvidencePolicy(),
                audit=audit,
                top_k=5,
            )
        )

        # Exact scopes remain part of the test.
        assert (
            authorization.retrieval_scope(
                unicef_identity
            ).document_ids
            == frozenset(
                {UNICEF_DOCUMENT_ID}
            )
        )

        assert (
            authorization.retrieval_scope(
                oxfam_identity
            ).document_ids
            == frozenset(
                {OXFAM_DOCUMENT_ID}
            )
        )

        assert (
            authorization.retrieval_scope(
                both_identity
            ).document_ids
            == frozenset(
                {
                    UNICEF_DOCUMENT_ID,
                    OXFAM_DOCUMENT_ID,
                }
            )
        )

        # Positive provenance controls.
        run_case(
            title=(
                "CONTROL — BOTH AUTHORIZED "
                "/ OXFAM CLAIM"
            ),
            orchestration=orchestration,
            identity=both_identity,
            claim=OXFAM_CLAIM,
            allowed_documents=frozenset(
                {
                    UNICEF_DOCUMENT_ID,
                    OXFAM_DOCUMENT_ID,
                }
            ),
            expected_document=(
                OXFAM_DOCUMENT_ID
            ),
            expect_sufficient=True,
        )

        run_case(
            title=(
                "CONTROL — BOTH AUTHORIZED "
                "/ UNICEF CLAIM"
            ),
            orchestration=orchestration,
            identity=both_identity,
            claim=UNICEF_CLAIM,
            allowed_documents=frozenset(
                {
                    UNICEF_DOCUMENT_ID,
                    OXFAM_DOCUMENT_ID,
                }
            ),
            expected_document=(
                UNICEF_DOCUMENT_ID
            ),
            expect_sufficient=True,
        )

        # Bidirectional attacks.
        run_case(
            title=(
                "ATTACK A — UNICEF-ONLY "
                "/ OXFAM CLAIM"
            ),
            orchestration=orchestration,
            identity=unicef_identity,
            claim=OXFAM_CLAIM,
            allowed_documents=frozenset(
                {UNICEF_DOCUMENT_ID}
            ),
            expected_document=None,
            expect_sufficient=False,
        )

        run_case(
            title=(
                "ATTACK B — OXFAM-ONLY "
                "/ UNICEF CLAIM"
            ),
            orchestration=orchestration,
            identity=oxfam_identity,
            claim=UNICEF_CLAIM,
            allowed_documents=frozenset(
                {OXFAM_DOCUMENT_ID}
            ),
            expected_document=None,
            expect_sufficient=False,
        )

        print()
        print("=" * 72)
        print(
            "ATTACK C — PARTIAL AUTHORIZATION "
            "/ MULTI-CLAIM"
        )
        print("=" * 72)

        partial_result = orchestration.evaluate(
            identity=unicef_identity,
            claims=(
                UNICEF_CLAIM,
                OXFAM_CLAIM,
            ),
        )

        print(
            "Decision:",
            partial_result.assessment.decision,
        )

        print(
            "is_sufficient:",
            partial_result.is_sufficient,
        )

        print(
            "Unsupported required claims:",
            partial_result.assessment
            .unsupported_required_claim_ids,
        )

        selected_by_claim = {
            item.claim_id: item.span.chunk.document_id
            for item in (
                partial_result.assessment
                .selected_evidence
            )
        }

        print(
            "Selected evidence by claim:",
            selected_by_claim,
        )

        for claim_result in (
            partial_result.claim_results
        ):
            retrieved_ids = {
                chunk.document_id
                for chunk in (
                    claim_result.retrieved_evidence
                )
            }

            nli_ids = {
                span.chunk.document_id
                for span in (
                    claim_result.verification.spans
                )
            }

            print()
            print(
                "Claim:",
                claim_result.claim.claim_id,
            )

            print(
                "Retrieved document IDs:",
                retrieved_ids,
            )

            print(
                "NLI document IDs:",
                nli_ids,
            )

            if not retrieved_ids.issubset(
                {UNICEF_DOCUMENT_ID}
            ):
                raise RuntimeError(
                    "Partial-auth attack leaked "
                    "an unauthorized document "
                    "into retrieval."
                )

            if not nli_ids.issubset(
                {UNICEF_DOCUMENT_ID}
            ):
                raise RuntimeError(
                    "Partial-auth attack leaked "
                    "an unauthorized document "
                    "into NLI."
                )

        if partial_result.is_sufficient:
            raise RuntimeError(
                "Partial authorization incorrectly "
                "produced a sufficient assessment."
            )

        if (
            "oxfam-partnership-policy"
            not in partial_result.assessment
            .unsupported_required_claim_ids
        ):
            raise RuntimeError(
                "Unauthorized Oxfam claim was not "
                "marked unsupported."
            )

        if (
            selected_by_claim.get(
                "unicef-level3-allocation"
            )
            != UNICEF_DOCUMENT_ID
        ):
            raise RuntimeError(
                "Authorized UNICEF claim lost "
                "its valid provenance."
            )

        if (
            "oxfam-partnership-policy"
            in selected_by_claim
        ):
            raise RuntimeError(
                "Unauthorized Oxfam claim "
                "received selected evidence."
            )

        print()
        print(
            "PARTIAL AUTHORIZATION ATTACK: PASS"
        )

        print()
        print("=" * 72)
        print(
            "TWO-DOCUMENT EVIDENCE ATTACK: PASS"
        )
        print("=" * 72)

        print(
            "Unauthorized docs in retrieval: 0"
        )

        print(
            "Unauthorized docs reaching NLI: 0"
        )

        print(
            "Unauthorized docs selected as "
            "evidence: 0"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()