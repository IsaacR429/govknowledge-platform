from dataclasses import replace
from uuid import uuid4

import pytest

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.authorized_table_evidence import (
    AuthorizedTableEvidenceError,
    AuthorizedTableEvidenceService,
)
from app.application.services.table_evidence_atoms import (
    TableCellEvidenceAtom,
    TableEvidenceAtom,
    TableLookupRequest,
)
from app.application.services.table_evidence_indexing import (
    TableEvidenceIndexingError,
    TableEvidenceIndexingService,
)
from app.domain.enums.permission import (
    PermissionAction,
)
from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
)


def identity():
    return CurrentIdentity(
        user_id=uuid4(),
        username="table-user",
        display_name=(
            "Table Evidence User"
        ),
        email=None,
        department="Test",
        language_preference="fr",
        roles=(
            "EMPLOYEE",
        ),
    )


def atom(
    *,
    document_id,
    version_id,
    logical_key="LOT_2/05",
    supplier=(
        "ENTREPRISE ZERBO AWA "
        "ET FRERES (EZAF)"
    ),
    evidence_id=None,
):
    page_id = uuid4()

    names = (
        "N°",
        "Soumissionnaires",
        "Montants lus en FCFA",
        "Montants corrigés en FCFA",
        "Observations",
    )

    values = (
        "05",
        supplier,
        "3 972 300 HTVA",
        "4 417 530 TTC",
        "Conforme 1er",
    )

    cells = tuple(
        TableCellEvidenceAtom(
            column_start=index,
            column_end=index + 1,
            column_name=name,
            text=value,
            value_source=(
                TableCellValueSource
                .SOURCE_TEXT
            ),
            origin_spans=(),
            structure_spans=(),
        )
        for index, (
            name,
            value,
        )
        in enumerate(
            zip(
                names,
                values,
                strict=True,
            )
        )
    )

    return TableEvidenceAtom(
        evidence_id=(
            evidence_id
            or (
                f"table:{version_id}:"
                f"{logical_key}"
            )
        ),
        document_id=document_id,
        document_version_id=(
            version_id
        ),
        page_id=page_id,
        page_number=4,
        source_table_refs=(
            "test-table",
        ),
        row_index=5,
        role=TableRowRole.DATA,
        logical_key=logical_key,
        section_key="LOT_2",
        column_names=names,
        cells=cells,
        language="fr",
    )


class FakeAuthorization:
    def __init__(
        self,
        *,
        document_ids,
        events=None,
    ):
        self.document_ids = (
            frozenset(
                document_ids
            )
        )

        self.events = (
            events
            if events is not None
            else []
        )

        self.scope_calls = []
        self.read_calls = []

    def retrieval_scope(
        self,
        current_identity,
    ):
        self.events.append(
            "authorization"
        )

        self.scope_calls.append(
            current_identity
        )

        return AuthorizationScope(
            user_id=(
                current_identity.user_id
            ),
            action=(
                PermissionAction.READ
            ),
            document_ids=(
                self.document_ids
            ),
        )

    def require_document_read(
        self,
        current_identity,
        document_id,
    ):
        self.events.append(
            "reauthorize"
        )

        self.read_calls.append(
            (
                current_identity,
                document_id,
            )
        )

        if (
            document_id
            not in self.document_ids
        ):
            raise AssertionError(
                "Unauthorized exact READ."
            )


class FakeRepository:
    def __init__(
        self,
        *,
        rows=(),
        events=None,
    ):
        self.rows = tuple(
            rows
        )

        self.events = (
            events
            if events is not None
            else []
        )

        self.lookup_calls = []
        self.replace_calls = []

    def find_authorized_by_logical_key(
        self,
        *,
        scope,
        logical_key,
    ):
        self.events.append(
            "repository"
        )

        self.lookup_calls.append(
            (
                scope,
                logical_key,
            )
        )

        return self.rows

    def replace_version_atoms(
        self,
        *,
        version_id,
        atoms,
    ):
        self.replace_calls.append(
            (
                version_id,
                tuple(
                    atoms
                ),
            )
        )


def request():
    return TableLookupRequest(
        section_key="LOT_2",
        ordinal=5,
        column_name=(
            "Soumissionnaires"
        ),
    )


def test_authorization_happens_before_repository():
    events = []

    document_id = uuid4()
    version_id = uuid4()

    authorization = FakeAuthorization(
        document_ids={
            document_id,
        },
        events=events,
    )

    repository = FakeRepository(
        rows=(
            atom(
                document_id=(
                    document_id
                ),
                version_id=(
                    version_id
                ),
            ),
        ),
        events=events,
    )

    result = (
        AuthorizedTableEvidenceService(
            authorization=authorization,
            repository=repository,
        )
        .lookup(
            identity=identity(),
            request=request(),
        )
    )

    assert result is not None

    assert events == [
        "authorization",
        "repository",
        "reauthorize",
    ]


def test_empty_scope_never_queries_repository():
    events = []

    repository = FakeRepository(
        events=events
    )

    service = (
        AuthorizedTableEvidenceService(
            authorization=(
                FakeAuthorization(
                    document_ids=set(),
                    events=events,
                )
            ),
            repository=repository,
        )
    )

    result = service.lookup(
        identity=identity(),
        request=request(),
    )

    assert result is None

    assert events == [
        "authorization",
    ]

    assert (
        repository.lookup_calls
        == []
    )


def test_authorized_lookup_returns_exact_cell():
    document_id = uuid4()
    version_id = uuid4()

    authorization = (
        FakeAuthorization(
            document_ids={
                document_id,
            }
        )
    )

    result = (
        AuthorizedTableEvidenceService(
            authorization=authorization,
            repository=(
                FakeRepository(
                    rows=(
                        atom(
                            document_id=(
                                document_id
                            ),
                            version_id=(
                                version_id
                            ),
                        ),
                    )
                )
            ),
        )
        .lookup(
            identity=identity(),
            request=request(),
        )
    )

    assert result is not None

    assert (
        result.row.logical_key
        == "LOT_2/05"
    )

    assert (
        "EZAF"
        in result.cell.text
    )

    assert len(
        authorization.read_calls
    ) == 1

    assert (
        authorization
        .read_calls[0][1]
        == document_id
    )


def test_missing_authorized_row_returns_none():
    authorization = (
        FakeAuthorization(
            document_ids={
                uuid4(),
            }
        )
    )

    result = (
        AuthorizedTableEvidenceService(
            authorization=authorization,
            repository=(
                FakeRepository(
                    rows=()
                )
            ),
        )
        .lookup(
            identity=identity(),
            request=request(),
        )
    )

    assert result is None

    # Nothing is disclosed, so no exact
    # document reauthorization is needed.
    assert (
        authorization.read_calls
        == []
    )


def test_repository_scope_leak_fails_closed():
    allowed_document = uuid4()
    leaked_document = uuid4()
    version_id = uuid4()

    service = (
        AuthorizedTableEvidenceService(
            authorization=(
                FakeAuthorization(
                    document_ids={
                        allowed_document,
                    }
                )
            ),
            repository=(
                FakeRepository(
                    rows=(
                        atom(
                            document_id=(
                                leaked_document
                            ),
                            version_id=(
                                version_id
                            ),
                        ),
                    )
                )
            ),
        )
    )

    with pytest.raises(
        AuthorizedTableEvidenceError,
        match="authorization scope",
    ):
        service.lookup(
            identity=identity(),
            request=request(),
        )


def test_repository_wrong_logical_key_fails_closed():
    document_id = uuid4()
    version_id = uuid4()

    service = (
        AuthorizedTableEvidenceService(
            authorization=(
                FakeAuthorization(
                    document_ids={
                        document_id,
                    }
                )
            ),
            repository=(
                FakeRepository(
                    rows=(
                        atom(
                            document_id=(
                                document_id
                            ),
                            version_id=(
                                version_id
                            ),
                            logical_key=(
                                "LOT_2/06"
                            ),
                        ),
                    )
                )
            ),
        )
    )

    with pytest.raises(
        AuthorizedTableEvidenceError,
        match="logical key",
    ):
        service.lookup(
            identity=identity(),
            request=request(),
        )


def test_indexing_rejects_wrong_version():
    expected_version = uuid4()
    other_version = uuid4()

    document_id = uuid4()

    repository = (
        FakeRepository()
    )

    service = (
        TableEvidenceIndexingService(
            repository=repository
        )
    )

    with pytest.raises(
        TableEvidenceIndexingError,
        match="outside",
    ):
        service.index_version(
            version_id=(
                expected_version
            ),
            atoms=(
                atom(
                    document_id=(
                        document_id
                    ),
                    version_id=(
                        other_version
                    ),
                ),
            ),
        )

    assert (
        repository.replace_calls
        == []
    )


def test_indexing_rejects_duplicate_evidence_ids():
    version_id = uuid4()
    document_id = uuid4()

    first = atom(
        document_id=document_id,
        version_id=version_id,
        evidence_id="duplicate",
    )

    second = replace(
        first,
        row_index=999,
    )

    repository = (
        FakeRepository()
    )

    with pytest.raises(
        TableEvidenceIndexingError,
        match="Duplicate",
    ):
        (
            TableEvidenceIndexingService(
                repository=repository
            )
            .index_version(
                version_id=version_id,
                atoms=(
                    first,
                    second,
                ),
            )
        )

    assert (
        repository.replace_calls
        == []
    )


def test_empty_indexing_still_replaces_version():
    version_id = uuid4()

    repository = (
        FakeRepository()
    )

    result = (
        TableEvidenceIndexingService(
            repository=repository
        )
        .index_version(
            version_id=version_id,
            atoms=(),
        )
    )

    assert (
        result.indexed_count
        == 0
    )

    assert (
        repository.replace_calls
        == [
            (
                version_id,
                (),
            )
        ]
    )
