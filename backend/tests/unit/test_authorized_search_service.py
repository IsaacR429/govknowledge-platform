from uuid import uuid4

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.application.ports.embedding_provider import (
    EmbeddingResult,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.domain.enums.permission import (
    PermissionAction,
)
from app.application.services.authorized_search import (
    AuthorizedSearchService,
)


def identity():
    return CurrentIdentity(
        user_id=uuid4(),
        username="employee-test",
        display_name="Employee Test",
        email=None,
        department="Test",
        language_preference="fr",
        roles=("EMPLOYEE",),
    )


class FakeAuthorization:
    def __init__(
        self,
        *,
        document_ids,
        events,
    ):
        self.document_ids = frozenset(
            document_ids
        )
        self.events = events

    def retrieval_scope(
        self,
        current_identity,
    ):
        self.events.append(
            "authorization"
        )

        return AuthorizationScope(
            user_id=(
                current_identity.user_id
            ),
            action=PermissionAction.READ,
            document_ids=(
                self.document_ids
            ),
        )


class FakeEmbeddingProvider:
    model_name = "test-model"
    model_revision = "test-revision"
    dimension = 768

    def __init__(
        self,
        events,
    ):
        self.events = events

    def embed_query(
        self,
        text,
    ):
        self.events.append(
            "embedding"
        )

        return EmbeddingResult(
            values=tuple(
                [1.0]
                + [0.0] * 767
            ),
            model_name=self.model_name,
            model_revision=(
                self.model_revision
            ),
            dimension=self.dimension,
        )


class FakeRetrieval:
    def __init__(
        self,
        events,
    ):
        self.events = events
        self.scope = None

    def search_authorized(
        self,
        *,
        scope,
        query_vector,
        model_name,
        model_revision,
        top_k,
    ):
        self.events.append(
            "retrieval"
        )
        self.scope = scope
        return ()


def test_authorization_happens_before_retrieval():
    events = []
    allowed = uuid4()

    authorization = FakeAuthorization(
        document_ids={allowed},
        events=events,
    )

    retrieval = FakeRetrieval(
        events
    )

    service = AuthorizedSearchService(
        authorization=authorization,
        embedding_provider=(
            FakeEmbeddingProvider(
                events
            )
        ),
        retrieval=retrieval,
    )

    service.search(
        identity=identity(),
        query="Quelle est la politique ?",
    )

    assert events == [
        "authorization",
        "embedding",
        "retrieval",
    ]

    assert (
        retrieval.scope.document_ids
        == frozenset(
            {allowed}
        )
    )


def test_empty_scope_never_embeds_or_retrieves():
    events = []

    service = AuthorizedSearchService(
        authorization=FakeAuthorization(
            document_ids=set(),
            events=events,
        ),
        embedding_provider=(
            FakeEmbeddingProvider(
                events
            )
        ),
        retrieval=FakeRetrieval(
            events
        ),
    )

    result = service.search(
        identity=identity(),
        query="Document secret ?",
    )

    assert result.results == ()

    assert events == [
        "authorization",
    ]


def test_empty_query_is_rejected():
    events = []

    service = AuthorizedSearchService(
        authorization=FakeAuthorization(
            document_ids=set(),
            events=events,
        ),
        embedding_provider=(
            FakeEmbeddingProvider(
                events
            )
        ),
        retrieval=FakeRetrieval(
            events
        ),
    )

    try:
        service.search(
            identity=identity(),
            query="   ",
        )
    except ValueError as exc:
        assert (
            "cannot be empty"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )
