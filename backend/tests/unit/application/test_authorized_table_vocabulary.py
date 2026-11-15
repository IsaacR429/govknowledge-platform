from uuid import uuid4

import pytest

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.application.ports.table_evidence_repository import (
    AuthorizedTableVocabulary,
)
from app.application.services.authorized_table_evidence import (
    AuthorizedTableEvidenceError,
    AuthorizedTableEvidenceService,
)
from app.domain.enums.permission import (
    PermissionAction,
)


class FakeAuthorization:
    def __init__(
        self,
        *,
        document_ids,
    ):
        self.scope = (
            AuthorizationScope(
                user_id=uuid4(),
                action=(
                    PermissionAction.READ
                ),
                document_ids=(
                    frozenset(
                        document_ids
                    )
                ),
            )
        )

        self.scope_calls = []
        self.read_calls = []

    def retrieval_scope(
        self,
        identity,
    ):
        self.scope_calls.append(
            identity
        )

        return self.scope

    def require_document_read(
        self,
        identity,
        document_id,
    ):
        self.read_calls.append(
            (
                identity,
                document_id,
            )
        )


class RecordingRepository:
    def __init__(
        self,
        *,
        vocabulary,
    ):
        self.vocabulary = (
            vocabulary
        )

        self.calls = []

    def list_authorized_vocabulary(
        self,
        *,
        scope,
    ):
        self.calls.append(
            scope
        )

        return self.vocabulary


class NeverRepository:
    def __init__(self):
        self.calls = []

    def list_authorized_vocabulary(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        raise AssertionError(
            "Repository must not run "
            "without authorization."
        )


def test_authorized_vocabulary_uses_scope_first():
    document_id = uuid4()

    authorization = (
        FakeAuthorization(
            document_ids=(
                document_id,
            )
        )
    )

    expected = (
        AuthorizedTableVocabulary(
            section_keys=(
                "LOT_1",
                "LOT_2",
            ),
            column_names=(
                "N°",
                "Soumissionnaires",
            ),
        )
    )

    repository = (
        RecordingRepository(
            vocabulary=expected
        )
    )

    identity = object()

    result = (
        AuthorizedTableEvidenceService(
            authorization=(
                authorization
            ),
            repository=repository,
        )
        .vocabulary(
            identity=identity
        )
    )

    assert result == expected

    assert (
        authorization
        .scope_calls
        == [
            identity
        ]
    )

    assert (
        repository.calls
        == [
            authorization.scope
        ]
    )

    # Vocabulary does not disclose table cell
    # values, so exact document reauthorization
    # is not performed by this service.
    #
    # The PostgreSQL repository independently
    # verifies current DocumentPermission.
    assert (
        authorization
        .read_calls
        == []
    )


def test_no_read_scope_skips_repository():
    authorization = (
        FakeAuthorization(
            document_ids=()
        )
    )

    repository = (
        NeverRepository()
    )

    result = (
        AuthorizedTableEvidenceService(
            authorization=(
                authorization
            ),
            repository=repository,
        )
        .vocabulary(
            identity=object()
        )
    )

    assert (
        result
        == (
            AuthorizedTableVocabulary
            .empty()
        )
    )

    assert repository.calls == []


def test_invalid_repository_vocabulary_fails_closed():
    authorization = (
        FakeAuthorization(
            document_ids=(
                uuid4(),
            )
        )
    )

    repository = (
        RecordingRepository(
            vocabulary=object()
        )
    )

    with pytest.raises(
        AuthorizedTableEvidenceError,
        match="invalid vocabulary",
    ):
        (
            AuthorizedTableEvidenceService(
                authorization=(
                    authorization
                ),
                repository=repository,
            )
            .vocabulary(
                identity=object()
            )
        )


def test_vocabulary_model_rejects_duplicates():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        AuthorizedTableVocabulary(
            section_keys=(
                "LOT_2",
                "LOT_2",
            ),
            column_names=(
                "Soumissionnaires",
            ),
        )

def test_generic_vocabulary_preserves_table_row_relationships():
    from app.application.ports.table_evidence_repository import (
        AuthorizedGenericTable,
        AuthorizedLabeledRow,
    )

    first_table_key = (
        "GENERIC/"
        + "a" * 64
    )

    second_table_key = (
        "GENERIC/"
        + "b" * 64
    )

    vocabulary = AuthorizedTableVocabulary(
        section_keys=(),
        column_names=(),
        generic_tables=(
            AuthorizedGenericTable(
                table_key=(
                    first_table_key
                ),
                rows=(
                    AuthorizedLabeledRow(
                        label="Output",
                        logical_key=(
                            first_table_key
                            + "/"
                            + "1" * 64
                        ),
                    ),
                ),
                column_names=(
                    "Feature",
                    "Credit Score System",
                ),
            ),
            AuthorizedGenericTable(
                table_key=(
                    second_table_key
                ),
                rows=(
                    AuthorizedLabeledRow(
                        label="Output",
                        logical_key=(
                            second_table_key
                            + "/"
                            + "2" * 64
                        ),
                    ),
                ),
                column_names=(
                    "Feature",
                    "Revenue",
                ),
            ),
        ),
    )

    assert len(
        vocabulary.generic_tables
    ) == 2

    assert (
        vocabulary
        .generic_tables[0]
        .rows[0]
        .label
        == "Output"
    )

    assert (
        vocabulary
        .generic_tables[1]
        .rows[0]
        .label
        == "Output"
    )

    assert (
        vocabulary
        .generic_tables[0]
        .table_key
        != vocabulary
        .generic_tables[1]
        .table_key
    )


def test_generic_vocabulary_rejects_duplicate_table_keys():
    from app.application.ports.table_evidence_repository import (
        AuthorizedGenericTable,
        AuthorizedLabeledRow,
    )

    table_key = (
        "GENERIC/"
        + "a" * 64
    )

    table = AuthorizedGenericTable(
        table_key=table_key,
        rows=(
            AuthorizedLabeledRow(
                label="Output",
                logical_key=(
                    table_key
                    + "/"
                    + "1" * 64
                ),
            ),
        ),
        column_names=(
            "Feature",
            "Output",
        ),
    )

    with pytest.raises(
        ValueError,
        match="table keys",
    ):
        AuthorizedTableVocabulary(
            section_keys=(),
            column_names=(),
            generic_tables=(
                table,
                table,
            ),
        )


def test_empty_vocabulary_contains_no_generic_tables():
    result = (
        AuthorizedTableVocabulary
        .empty()
    )

    assert result.section_keys == ()
    assert result.column_names == ()
    assert result.generic_tables == ()
