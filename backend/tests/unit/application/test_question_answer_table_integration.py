from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from app.application.services.grounded_answer_generation import (
    GroundedAnswerResult,
)
from app.application.services.question_answer import (
    QuestionAnswerService,
)
from app.application.services.validated_citations import (
    ClaimCitation,
    ValidatedCitationSet,
)
from app.domain.enums.audit import (
    AuditEventType,
)


class RecordingAudit:
    def __init__(self):
        self.events = []

    def record(
        self,
        event,
    ):
        self.events.append(
            event
        )


class NeverSearch:
    def __init__(self):
        self.calls = []

    def search(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        raise AssertionError(
            "Text search must not run."
        )


class EmptySearch:
    def __init__(self):
        self.calls = []

    def search(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        return SimpleNamespace(
            results=()
        )


class NeverPlanner:
    def plan(
        self,
        request,
    ):
        raise AssertionError(
            "Text claim planner must not run."
        )


class EmptyEvidence:
    def __init__(self):
        self.calls = []

    def evaluate(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        return SimpleNamespace(
            correlation_id=(
                kwargs[
                    "correlation_id"
                ]
            ),
            assessment=(
                SimpleNamespace(
                    is_sufficient=False,
                )
            ),
            claim_results=(),
        )


class RecordingTextAnswer:
    def __init__(self):
        self.calls = []

    def generate(
        self,
        *,
        question,
        evidence_result,
        language,
    ):
        self.calls.append(
            {
                "question": question,
                "evidence_result": (
                    evidence_result
                ),
                "language": language,
            }
        )

        return GroundedAnswerResult(
            correlation_id=(
                evidence_result
                .correlation_id
            ),
            status=(
                "INSUFFICIENT_EVIDENCE"
            ),
            answer=(
                "TEXT REFUSAL"
            ),
            citations=(
                ValidatedCitationSet(
                    citations=()
                )
            ),
            cited_evidence_ids=(),
        )


class NeverAuthorization:
    def require_document_read(
        self,
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Text reauthorization "
            "must not run."
        )


class RecordingTableAnswer:
    def __init__(
        self,
        result,
    ):
        self.result = result
        self.calls = []

    def try_answer(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        return self.result


def identity():
    return SimpleNamespace(
        user_id=uuid4()
    )


def table_citation():
    return SimpleNamespace(
        evidence_id=(
            "table:test:LOT_2/05"
        ),
        document_id=uuid4(),
        document_version_id=(
            uuid4()
        ),
        page_id=uuid4(),
        page_number=4,
        logical_key="LOT_2/05",
        column_name=(
            "Soumissionnaires"
        ),
        cited_text=(
            "ENTREPRISE ZERBO AWA "
            "ET FRERES (EZAF)"
        ),
    )


def supported_table_result():
    citation = (
        table_citation()
    )

    return GroundedAnswerResult(
        correlation_id=uuid4(),
        status="SUPPORTED",
        answer=(
            citation.cited_text
        ),
        citations=(
            ValidatedCitationSet(
                citations=(
                    ClaimCitation(
                        claim_id=(
                            "table-claim-1"
                        ),
                        claim=(
                            citation
                            .cited_text
                        ),
                        citation=(
                            citation
                        ),
                    ),
                )
            )
        ),
        cited_evidence_ids=(
            "table-claim-1",
        ),
    )


def make_service(
    *,
    table_result,
    search,
    audit,
    evidence=None,
    text_answer=None,
):
    return QuestionAnswerService(
        search=search,
        planner=NeverPlanner(),
        evidence=(
            evidence
            or EmptyEvidence()
        ),
        answer=(
            text_answer
            or RecordingTextAnswer()
        ),
        authorization=(
            NeverAuthorization()
        ),
        audit=audit,
        table_answer=(
            RecordingTableAnswer(
                table_result
            )
        ),
    )


def test_supported_table_answer_stops_text_rag():
    audit = RecordingAudit()

    search = NeverSearch()

    table_result = (
        supported_table_result()
    )

    service = make_service(
        table_result=(
            table_result
        ),
        search=search,
        audit=audit,
    )

    result = service.answer(
        identity=identity(),
        question=(
            "Quel soumissionnaire "
            "correspond au numero "
            "05 du lot 2 ?"
        ),
        language="fr",
    )

    assert (
        result is table_result
    )

    assert search.calls == []

    event_types = tuple(
        item.event_type
        for item in audit.events
    )

    assert (
        AuditEventType
        .CITATION_VALIDATED
        in event_types
    )

    assert (
        AuditEventType
        .ANSWER_GENERATED
        in event_types
    )

    citation_events = [
        event
        for event in audit.events
        if (
            event.event_type
            == AuditEventType
            .CITATION_VALIDATED
        )
    ]

    assert (
        len(citation_events)
        == 1
    )

    metadata = (
        citation_events[0]
        .metadata
    )

    assert (
        metadata[
            "table_evidence_id"
        ]
        == "table:test:LOT_2/05"
    )

    assert (
        metadata[
            "logical_key"
        ]
        == "LOT_2/05"
    )

    assert (
        metadata[
            "column_name"
        ]
        == "Soumissionnaires"
    )

    assert (
        "chunk_id"
        not in metadata
    )


def test_table_insufficient_evidence_does_not_fall_back():
    audit = RecordingAudit()

    search = NeverSearch()

    table_result = (
        GroundedAnswerResult(
            correlation_id=uuid4(),
            status=(
                "INSUFFICIENT_EVIDENCE"
            ),
            answer=(
                "TABLE REFUSAL"
            ),
            citations=(
                ValidatedCitationSet(
                    citations=()
                )
            ),
            cited_evidence_ids=(),
        )
    )

    service = make_service(
        table_result=(
            table_result
        ),
        search=search,
        audit=audit,
    )

    result = service.answer(
        identity=identity(),
        question=(
            "Quel soumissionnaire "
            "correspond au numero "
            "09 du lot 2 ?"
        ),
        language="fr",
    )

    assert (
        result is table_result
    )

    assert search.calls == []

    event_types = tuple(
        item.event_type
        for item in audit.events
    )

    assert (
        AuditEventType
        .ANSWER_REFUSED
        in event_types
    )


def test_table_decline_preserves_existing_text_path():
    audit = RecordingAudit()

    search = EmptySearch()

    evidence = EmptyEvidence()

    text_answer = (
        RecordingTextAnswer()
    )

    service = make_service(
        table_result=None,
        search=search,
        audit=audit,
        evidence=evidence,
        text_answer=(
            text_answer
        ),
    )

    result = service.answer(
        identity=identity(),
        question=(
            "Resume les obligations "
            "principales du document."
        ),
        language="fr",
    )

    assert (
        result.status
        == "INSUFFICIENT_EVIDENCE"
    )

    assert (
        result.answer
        == "TEXT REFUSAL"
    )

    assert (
        len(search.calls)
        == 1
    )

    assert (
        len(evidence.calls)
        == 1
    )

    assert (
        len(text_answer.calls)
        == 1
    )


def test_table_route_receives_same_normalized_question_context():
    audit = RecordingAudit()

    table = (
        RecordingTableAnswer(
            None
        )
    )

    search = EmptySearch()

    service = QuestionAnswerService(
        search=search,
        planner=NeverPlanner(),
        evidence=(
            EmptyEvidence()
        ),
        answer=(
            RecordingTextAnswer()
        ),
        authorization=(
            NeverAuthorization()
        ),
        audit=audit,
        table_answer=table,
    )

    user = identity()

    service.answer(
        identity=user,
        question="  Question  ",
        language=" FR ",
    )

    assert (
        len(table.calls)
        == 1
    )

    call = table.calls[0]

    assert (
        call["identity"]
        is user
    )

    assert (
        call["question"]
        == "Question"
    )

    assert (
        call["language"]
        == "fr"
    )

    assert (
        call["correlation_id"]
        is not None
    )
