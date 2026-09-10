from uuid import UUID, uuid4

from sqlalchemy import delete

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
    DocumentVersion,
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


VERSION_ID = UUID(
    "86aa0794-a065-4819-a4fe-01694fad6122"
)

QUERY = (
    "Quelle est la responsabilité principale "
    "du Conseil de sécurité ?"
)


def main() -> None:
    db = SessionLocal()

    user_id = uuid4()
    permission_id = uuid4()

    try:
        version = db.get(
            DocumentVersion,
            VERSION_ID,
        )

        if version is None:
            raise RuntimeError(
                "UN document version not found."
            )

        document = db.get(
            Document,
            version.document_id,
        )

        if document is None:
            raise RuntimeError(
                "UN document not found."
            )

        if document.status != "ACTIVE":
            raise RuntimeError(
                "UN document is not ACTIVE."
            )

        username = (
            "stage4-search-"
            + user_id.hex[:12]
        )

        db.add(
            User(
                id=user_id,
                username=username,
                display_name=(
                    "Stage 4 Search Test"
                ),
                email=None,
                department="Security Test",
                language_preference="fr",
                is_active=True,
            )
        )

        db.add(
            DocumentPermission(
                id=permission_id,
                user_id=user_id,
                document_id=document.id,
                action=(
                    PermissionAction.READ.value
                ),
            )
        )

        db.commit()

        identity = CurrentIdentity(
            user_id=user_id,
            username=username,
            display_name=(
                "Stage 4 Search Test"
            ),
            email=None,
            department="Security Test",
            language_preference="fr",
            roles=("EMPLOYEE",),
        )

        provider = (
            SentenceTransformerEmbeddingProvider()
        )

        authorization = AuthorizationService(
            PostgresAuthorizationRepository(
                db
            )
        )

        retrieval = (
            PostgresVectorRetrievalAdapter(
                db
            )
        )

        search = AuthorizedSearchService(
            authorization=authorization,
            embedding_provider=provider,
            retrieval=retrieval,
        )

        print()
        print(
            "=== Real authorized semantic search ==="
        )
        print(
            "Question:",
            QUERY,
        )
        print(
            "Document:",
            document.title,
        )
        print(
            "Model:",
            provider.model_name,
        )
        print(
            "Revision:",
            provider.model_revision,
        )
        print()

        result = search.search(
            identity=identity,
            query=QUERY,
            top_k=5,
        )

        if not result.results:
            raise RuntimeError(
                "Authorized search returned "
                "no evidence."
            )

        if any(
            item.document_id
            != document.id
            for item in result.results
        ):
            raise RuntimeError(
                "Search escaped the authorized "
                "document scope."
            )

        for rank, item in enumerate(
            result.results,
            start=1,
        ):
            snippet = " ".join(
                item.content.split()
            )

            print(
                f"{rank}. "
                f"page={item.page_number} "
                f"score={item.score:.4f}"
            )

            print(
                "   ",
                snippet[:260],
            )

        print()
        print(
            "Authorized retrieval:",
            "PASS",
        )

        # Revoke the only READ permission.
        db.execute(
            delete(
                DocumentPermission
            ).where(
                DocumentPermission.id
                == permission_id
            )
        )

        db.commit()

        after_revocation = search.search(
            identity=identity,
            query=QUERY,
            top_k=5,
        )

        if after_revocation.results:
            raise RuntimeError(
                "Evidence remained retrievable "
                "after permission revocation."
            )

        print(
            "Retrieval after revocation:",
            len(
                after_revocation.results
            ),
        )

        print(
            "Revocation boundary:",
            "PASS",
        )

        print()
        print(
            "Stage 4 real search proof:",
            "PASS",
        )

    finally:
        db.rollback()

        db.execute(
            delete(
                DocumentPermission
            ).where(
                DocumentPermission.id
                == permission_id
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


if __name__ == "__main__":
    main()
