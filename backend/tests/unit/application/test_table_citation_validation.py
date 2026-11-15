from dataclasses import replace
from uuid import uuid4

import pytest

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.table_citation_validation import (
    InvalidTableCitationError,
    TableClaimCitationValidator,
)
from app.application.services.table_claim_provenance import (
    DirectTableClaimBuilder,
    TableAtomicClaim,
)
from app.application.services.table_evidence_atoms import (
    TableCellEvidenceAtom,
    TableEvidenceAtom,
    TableLookupRequest,
    TableLookupSelection,
)
from app.domain.enums.permission import (
    PermissionAction,
)
from app.domain.enums.table import (
    TableCellValueSource,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    TableSourceSpan,
)


SUPPLIER = (
    "ENTREPRISE ZERBO AWA "
    "ET FRERES (EZAF)"
)


def identity():
    return CurrentIdentity(
        user_id=uuid4(),
        username="citation-user",
        display_name=(
            "Citation User"
        ),
        email=None,
        department="QA",
        language_preference="fr",
        roles=(
            "EMPLOYEE",
        ),
    )


def source_span(
    *,
    page=4,
    text=SUPPLIER,
    source_ref="quotidien:p4:supplier",
):
    return TableSourceSpan(
        page_number=page,
        kind=TableSpanKind.TEXT,
        source_ref=source_ref,
        bbox=None,
        text=text,
    )


def structure_span(
    *,
    page=4,
):
    return TableSourceSpan(
        page_number=page,
        kind=TableSpanKind.STRUCTURE,
        source_ref=(
            "quotidien:p4:structure"
        ),
        bbox=None,
        text=None,
    )


def canonical_row():
    document_id = uuid4()
    version_id = uuid4()
    page_id = uuid4()

    supplier_cell = (
        TableCellEvidenceAtom(
            column_start=1,
            column_end=2,
            column_name=(
                "Soumissionnaires"
            ),
            text=SUPPLIER,
            value_source=(
                TableCellValueSource
                .SOURCE_TEXT
            ),
            origin_spans=(
                source_span(),
            ),
            structure_spans=(
                structure_span(),
            ),
        )
    )

    row = TableEvidenceAtom(
        evidence_id=(
            f"table:{version_id}:"
            "LOT_2/05"
        ),
        document_id=document_id,
        document_version_id=(
            version_id
        ),
        page_id=page_id,
        page_number=4,
        source_table_refs=(
            "xref:781",
        ),
        row_index=10,
        role=TableRowRole.DATA,
        logical_key="LOT_2/05",
        section_key="LOT_2",
        column_names=(
            "N°",
            "Soumissionnaires",
        ),
        cells=(
            TableCellEvidenceAtom(
                column_start=0,
                column_end=1,
                column_name="N°",
                text="05",
                value_source=(
                    TableCellValueSource
                    .SOURCE_TEXT
                ),
                origin_spans=(
                    TableSourceSpan(
                        page_number=4,
                        kind=(
                            TableSpanKind.TEXT
                        ),
                        source_ref=(
                            "quotidien:p4:"
                            "ordinal"
                        ),
                        bbox=None,
                        text="05",
                    ),
                ),
                structure_spans=(),
            ),
            supplier_cell,
        ),
        language="fr",
    )

    request = TableLookupRequest(
        section_key="LOT_2",
        ordinal=5,
        column_name=(
            "Soumissionnaires"
        ),
    )

    selection = (
        TableLookupSelection(
            request=request,
            row=row,
            cell=supplier_cell,
        )
    )

    claim = (
        DirectTableClaimBuilder()
        .build(
            request=request,
            selection=selection,
        )
    )

    return row, claim


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

        self.read_calls = []

    def retrieval_scope(
        self,
        current_identity,
    ):
        self.events.append(
            "scope"
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
            "read"
        )

        self.read_calls.append(
            document_id
        )

        if (
            document_id
            not in self.document_ids
        ):
            raise AssertionError(
                "Unauthorized exact read."
            )


class FakeRepository:
    def __init__(
        self,
        *,
        row=None,
        events=None,
    ):
        self.row = row

        self.events = (
            events
            if events is not None
            else []
        )

        self.calls = []

    def find_authorized_by_evidence_id(
        self,
        *,
        scope,
        evidence_id,
    ):
        self.events.append(
            "repository"
        )

        self.calls.append(
            (
                scope,
                evidence_id,
            )
        )

        return self.row


def validator(
    row,
    *,
    authorized=True,
    events=None,
):
    document_ids = (
        {
            row.document_id,
        }
        if authorized
        else set()
    )

    authorization = (
        FakeAuthorization(
            document_ids=(
                document_ids
            ),
            events=events,
        )
    )

    repository = (
        FakeRepository(
            row=row,
            events=events,
        )
    )

    service = (
        TableClaimCitationValidator(
            authorization=(
                authorization
            ),
            repository=repository,
        )
    )

    return (
        service,
        authorization,
        repository,
    )


def test_builds_canonical_table_citation():
    row, claim = canonical_row()

    service, authorization, repository = (
        validator(row)
    )

    result = service.validate(
        identity=identity(),
        claim=claim,
    )

    assert (
        result.evidence_id
        == row.evidence_id
    )

    assert (
        result.document_id
        == row.document_id
    )

    assert (
        result.page_number
        == 4
    )

    assert (
        result.logical_key
        == "LOT_2/05"
    )

    assert (
        result.column_name
        == "Soumissionnaires"
    )

    assert (
        result.cited_text
        == SUPPLIER
    )

    assert (
        result.origin_pages
        == (4,)
    )

    assert (
        result.structure_pages
        == (4,)
    )

    assert (
        result.source_refs
        == (
            "quotidien:p4:supplier",
        )
    )

    assert (
        authorization.read_calls
        == [
            row.document_id,
        ]
    )


def test_security_order_is_scope_repository_read():
    row, claim = canonical_row()

    events = []

    service, _, _ = validator(
        row,
        events=events,
    )

    service.validate(
        identity=identity(),
        claim=claim,
    )

    assert events == [
        "scope",
        "repository",
        "read",
    ]


def test_empty_scope_fails_before_repository():
    row, claim = canonical_row()

    events = []

    service, _, repository = (
        validator(
            row,
            authorized=False,
            events=events,
        )
    )

    with pytest.raises(
        InvalidTableCitationError,
        match="invalid",
    ):
        service.validate(
            identity=identity(),
            claim=claim,
        )

    assert events == [
        "scope",
    ]

    assert repository.calls == []


def test_missing_persisted_evidence_fails_closed():
    row, claim = canonical_row()

    authorization = (
        FakeAuthorization(
            document_ids={
                row.document_id,
            }
        )
    )

    service = (
        TableClaimCitationValidator(
            authorization=(
                authorization
            ),
            repository=(
                FakeRepository(
                    row=None
                )
            ),
        )
    )

    with pytest.raises(
        InvalidTableCitationError,
        match="invalid",
    ):
        service.validate(
            identity=identity(),
            claim=claim,
        )


def test_tampered_claim_text_fails_closed():
    row, claim = canonical_row()

    tampered_origin = replace(
        claim.origin,
        text="TAMPERED SUPPLIER",
    )

    tampered = TableAtomicClaim(
        claim_id=claim.claim_id,
        text="TAMPERED SUPPLIER",
        origin=tampered_origin,
        language=claim.language,
        required=True,
    )

    service, _, _ = validator(
        row
    )

    with pytest.raises(
        InvalidTableCitationError,
        match="invalid",
    ):
        service.validate(
            identity=identity(),
            claim=tampered,
        )


def test_tampered_page_fails_closed():
    row, claim = canonical_row()

    tampered_origin = replace(
        claim.origin,
        page_number=99,
    )

    tampered = TableAtomicClaim(
        claim_id=claim.claim_id,
        text=claim.text,
        origin=tampered_origin,
        language=claim.language,
        required=True,
    )

    service, _, _ = validator(
        row
    )

    with pytest.raises(
        InvalidTableCitationError,
        match="invalid",
    ):
        service.validate(
            identity=identity(),
            claim=tampered,
        )


def test_tampered_origin_spans_fail_closed():
    row, claim = canonical_row()

    changed_span = source_span(
        source_ref=(
            "tampered:source"
        )
    )

    tampered_origin = replace(
        claim.origin,
        origin_spans=(
            changed_span,
        ),
    )

    tampered = TableAtomicClaim(
        claim_id=claim.claim_id,
        text=claim.text,
        origin=tampered_origin,
        language=claim.language,
        required=True,
    )

    service, _, _ = validator(
        row
    )

    with pytest.raises(
        InvalidTableCitationError,
        match="invalid",
    ):
        service.validate(
            identity=identity(),
            claim=tampered,
        )


def test_duplicate_normalized_column_fails_closed():
    row, claim = canonical_row()

    supplier = row.cells[1]

    duplicate = replace(
        supplier,
        column_start=2,
        column_end=3,
        column_name=(
            "Soumissionnaires"
        ),
    )

    malformed_row = replace(
        row,
        cells=(
            *row.cells,
            duplicate,
        ),
    )

    service, _, _ = validator(
        malformed_row
    )

    with pytest.raises(
        InvalidTableCitationError,
        match="invalid",
    ):
        service.validate(
            identity=identity(),
            claim=claim,
        )


def test_inherited_persisted_value_fails_closed():
    row, claim = canonical_row()

    inherited_cell = replace(
        row.cells[1],
        value_source=(
            TableCellValueSource
            .INHERITED
        ),
    )

    inherited_row = replace(
        row,
        cells=(
            row.cells[0],
            inherited_cell,
        ),
    )

    inherited_origin = replace(
        claim.origin,
        value_source=(
            TableCellValueSource
            .INHERITED
        ),
    )

    inherited_claim = (
        TableAtomicClaim(
            claim_id=claim.claim_id,
            text=claim.text,
            origin=inherited_origin,
            language=claim.language,
            required=True,
        )
    )

    service, _, _ = validator(
        inherited_row
    )

    with pytest.raises(
        InvalidTableCitationError,
        match="invalid",
    ):
        service.validate(
            identity=identity(),
            claim=inherited_claim,
        )
