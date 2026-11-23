from dataclasses import replace
from uuid import uuid4

from app.application.services.table_citation_validation import (
    TableCitation,
)
from app.application.services.table_claim_provenance import (
    TableAtomicClaim,
    TableClaimOrigin,
)
from app.application.services.table_grounded_answer_generation import (
    TableGroundedAnswerGenerationService,
)
from app.domain.enums.table import (
    TableCellValueSource,
    TableSpanKind,
)
from app.domain.models.table import (
    TableSourceSpan,
)


SUPPLIER = (
    "ENTREPRISE ZERBO AWA "
    "ET FRERES (EZAF)"
)


def source_span():
    return TableSourceSpan(
        page_number=4,
        kind=TableSpanKind.TEXT,
        source_ref=(
            "quotidien:xref781:"
            "lot2:row05:supplier"
        ),
        bbox=None,
        text=SUPPLIER,
    )


def structure_span():
    return TableSourceSpan(
        page_number=4,
        kind=(
            TableSpanKind
            .STRUCTURE
        ),
        source_ref=(
            "quotidien:xref781:"
            "lot2:row05"
        ),
        bbox=None,
        text=None,
    )


def claim():
    document_id = uuid4()
    version_id = uuid4()
    page_id = uuid4()

    origin = TableClaimOrigin(
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
        logical_key="LOT_2/05",
        section_key="LOT_2",
        column_name=(
            "Soumissionnaires"
        ),
        text=SUPPLIER,
        language="fr",
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

    return TableAtomicClaim(
        claim_id="table-claim-1",
        text=SUPPLIER,
        origin=origin,
        language="fr",
        required=True,
    )


class RecordingCitationValidator:
    def __init__(self):
        self.calls = []

    def validate(
        self,
        *,
        identity,
        claim,
    ):
        self.calls.append(
            (
                identity,
                claim,
            )
        )

        origin = claim.origin

        return TableCitation(
            evidence_id=(
                origin.evidence_id
            ),
            document_id=(
                origin.document_id
            ),
            document_version_id=(
                origin.document_version_id
            ),
            page_id=(
                origin.page_id
            ),
            page_number=(
                origin.page_number
            ),
            logical_key=(
                origin.logical_key
            ),
            section_key=(
                origin.section_key
            ),
            column_name=(
                origin.column_name
            ),
            cited_text=(
                origin.text
            ),
            language=(
                origin.language
            ),
            origin_spans=(
                origin.origin_spans
            ),
            structure_spans=(
                origin.structure_spans
            ),
        )


class NeverCitationValidator:
    def __init__(self):
        self.calls = []

    def validate(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        raise AssertionError(
            "Citation validation must "
            "not run without evidence."
        )


def test_supported_table_answer_is_exact_canonical_cell():
    selected_claim = claim()

    validator = (
        RecordingCitationValidator()
    )

    identity = object()

    correlation_id = uuid4()

    result = (
        TableGroundedAnswerGenerationService(
            citation_validator=(
                validator
            )
        )
        .generate(
            identity=identity,
            question=(
                "Quel soumissionnaire "
                "correspond au numéro "
                "05 du lot 2 ?"
            ),
            claim=selected_claim,
            correlation_id=(
                correlation_id
            ),
            language="fr",
        )
    )

    assert (
        result.correlation_id
        == correlation_id
    )

    assert (
        result.status
        == "SUPPORTED"
    )

    assert (
        result.answer
        == SUPPLIER
    )

    assert (
        result.cited_evidence_ids
        == (
            "table-claim-1",
        )
    )

    assert (
        len(
            result.citations
            .citations
        )
        == 1
    )

    citation = (
        result.citations
        .citations[0]
        .citation
    )

    assert isinstance(
        citation,
        TableCitation,
    )

    assert (
        citation.cited_text
        == SUPPLIER
    )

    assert (
        citation.page_number
        == 4
    )

    assert (
        validator.calls
        == [
            (
                identity,
                selected_claim,
            )
        ]
    )


def test_missing_table_evidence_returns_refusal():
    validator = (
        NeverCitationValidator()
    )

    result = (
        TableGroundedAnswerGenerationService(
            citation_validator=(
                validator
            )
        )
        .generate(
            identity=object(),
            question=(
                "Quel soumissionnaire "
                "correspond au numéro "
                "09 du lot 2 ?"
            ),
            claim=None,
            language="fr",
        )
    )

    assert (
        result.status
        == "INSUFFICIENT_EVIDENCE"
    )

    assert (
        result.citations
        .citations
        == ()
    )

    assert (
        result.cited_evidence_ids
        == ()
    )

    assert validator.calls == []


def test_answer_uses_revalidated_citation_text():
    selected_claim = claim()

    validator = (
        RecordingCitationValidator()
    )

    result = (
        TableGroundedAnswerGenerationService(
            citation_validator=(
                validator
            )
        )
        .generate(
            identity=object(),
            question="Question",
            claim=selected_claim,
        )
    )

    assert (
        result.answer
        == SUPPLIER
    )


def test_answer_citation_uses_same_claim_id():
    selected_claim = claim()

    result = (
        TableGroundedAnswerGenerationService(
            citation_validator=(
                RecordingCitationValidator()
            )
        )
        .generate(
            identity=object(),
            question="Question",
            claim=selected_claim,
        )
    )

    entry = (
        result.citations
        .citations[0]
    )

    assert (
        entry.claim_id
        == selected_claim.claim_id
    )

    assert (
        entry.claim
        == selected_claim.text
    )


def test_comparison_answer_revalidates_and_cites_every_claim():

    first = claim()

    first_origin = replace(
        first.origin,
        logical_key=(
            "GENERIC/"
            + "a" * 64
            + "/"
            + "b" * 64
        ),
        section_key=None,
        column_name="Celluleanimale",
        text="Absents",
    )

    first = replace(
        first,
        claim_id="table-claim-animal",
        text="Absents",
        origin=first_origin,
    )

    second_origin = replace(
        first.origin,
        column_name=(
            "Cellulev\u00e9g\u00e9tale"
        ),
        text=(
            "Pr\u00e9sents "
            "(contiennent la chlorophylle "
            "pour la photosynth\u00e8se)"
        ),
    )

    second = replace(
        first,
        claim_id="table-claim-plant",
        text=(
            "Pr\u00e9sents "
            "(contiennent la chlorophylle "
            "pour la photosynth\u00e8se)"
        ),
        origin=second_origin,
    )

    validator = (
        RecordingCitationValidator()
    )

    identity = object()

    correlation_id = uuid4()

    result = (
        TableGroundedAnswerGenerationService(
            citation_validator=validator
        )
        .generate_comparison(
            identity=identity,
            question=(
                "Compare les chloroplastes "
                "chez la cellule animale "
                "et la cellule v\u00e9g\u00e9tale."
            ),
            claims=(
                first,
                second,
            ),
            correlation_id=(
                correlation_id
            ),
            language="fr",
        )
    )

    assert (
        result.correlation_id
        == correlation_id
    )

    assert result.status == "SUPPORTED"

    assert (
        result.answer
        == (
            "Celluleanimale: Absents; "
            "Cellulev\u00e9g\u00e9tale: "
            "Pr\u00e9sents "
            "(contiennent la chlorophylle "
            "pour la photosynth\u00e8se)"
        )
    )

    assert (
        result.cited_evidence_ids
        == (
            "table-claim-animal",
            "table-claim-plant",
        )
    )

    assert (
        len(
            result.citations.citations
        )
        == 2
    )

    assert (
        tuple(
            entry.claim_id
            for entry
            in result.citations.citations
        )
        == (
            "table-claim-animal",
            "table-claim-plant",
        )
    )

    assert (
        tuple(
            entry.citation.column_name
            for entry
            in result.citations.citations
        )
        == (
            "Celluleanimale",
            "Cellulev\u00e9g\u00e9tale",
        )
    )

    assert (
        tuple(
            entry.citation.cited_text
            for entry
            in result.citations.citations
        )
        == (
            "Absents",
            (
                "Pr\u00e9sents "
                "(contiennent la chlorophylle "
                "pour la photosynth\u00e8se)"
            ),
        )
    )

    assert (
        validator.calls
        == [
            (
                identity,
                first,
            ),
            (
                identity,
                second,
            ),
        ]
    )
