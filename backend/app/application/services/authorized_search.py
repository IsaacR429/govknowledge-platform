from dataclasses import dataclass

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.retrieval import (
    RetrievalPort,
    RetrievedChunk,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.ports.embedding_provider import (
    EmbeddingProvider,
)


@dataclass(frozen=True)
class AuthorizedSearchResult:
    query: str
    results: tuple[
        RetrievedChunk,
        ...,
    ]


class AuthorizedSearchService:
    def __init__(
        self,
        *,
        authorization: AuthorizationService,
        embedding_provider: EmbeddingProvider,
        retrieval: RetrievalPort,
    ) -> None:
        self._authorization = authorization
        self._embedding_provider = (
            embedding_provider
        )
        self._retrieval = retrieval

    def search(
        self,
        *,
        identity: CurrentIdentity,
        query: str,
        top_k: int = 5,
    ) -> AuthorizedSearchResult:
        cleaned = query.strip()

        if not cleaned:
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        # Authorization happens BEFORE
        # embedding-backed retrieval.
        scope = (
            self._authorization
            .retrieval_scope(
                identity
            )
        )

        # No authorization => no model work
        # and absolutely no corpus query.
        if not scope.document_ids:
            return AuthorizedSearchResult(
                query=cleaned,
                results=(),
            )

        query_embedding = (
            self._embedding_provider
            .embed_query(
                cleaned
            )
        )

        results = (
            self._retrieval
            .search_authorized(
                scope=scope,
                query_vector=(
                    query_embedding.values
                ),
                model_name=(
                    query_embedding.model_name
                ),
                model_revision=(
                    query_embedding
                    .model_revision
                ),
                top_k=top_k,
            )
        )

        # Defense-in-depth assertion.
        # The actual security filtering must
        # already have happened in SQL.
        if any(
            not scope.allows(
                result.document_id
            )
            for result in results
        ):
            raise RuntimeError(
                "Retrieval adapter violated "
                "authorization scope."
            )

        return AuthorizedSearchResult(
            query=cleaned,
            results=results,
        )
