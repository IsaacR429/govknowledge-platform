from uuid import uuid4

from app.application.services.grounded_answer_generation import (
    GroundedAnswerResult,
)
from app.application.services.table_claim_provenance import (
    TableAtomicClaim,
)
from app.application.services.table_evidence_atoms import (
    TableLookupRequest,
)
from app.application.services.table_question_orchestration import (
    TableQuestionOrchestrationService,
)
from app.application.services.validated_citations import (
    ValidatedCitationSet,
)


class RecordingEvidence:
    def __init__(
        self,
        selection,
        events,
    ):
        self.selection = selection
        self.events = events
        self.calls = []

    def lookup(
        self,
        *,
        identity,
        request,
    ):
        self.events.append(
            "authorized_lookup"
        )

        self.calls.append(
            (
                identity,
                request,
            )
        )

        return self.selection


class RecordingClaimBuilder:
    def __init__(
        self,
        claim,
        events,
    ):
        self.claim = claim
        self.events = events
        self.calls = []

    def build(
        self,
        *,
        request,
        selection,
    ):
        self.events.append(
            "claim_builder"
        )

        self.calls.append(
            (
                request,
                selection,
            )
        )

        return self.claim


class NeverClaimBuilder:
    def __init__(
        self,
        events,
    ):
        self.events = events
        self.calls = []

    def build(
        self,
        **kwargs,
    ):
        self.events.append(
            "claim_builder"
        )

        self.calls.append(
            kwargs
        )

        raise AssertionError(
            "Claim builder must not run "
            "without table evidence."
        )


class RecordingAnswer:
    def __init__(
        self,
        events,
    ):
        self.events = events
        self.calls = []

    def generate(
        self,
        *,
        identity,
        question,
        claim,
        correlation_id,
        language,
    ):
        self.events.append(
            "answer"
        )

        self.calls.append(
            {
                "identity": identity,
                "question": question,
                "claim": claim,
                "correlation_id": (
                    correlation_id
                ),
                "language": language,
            }
        )

        if claim is None:
            return GroundedAnswerResult(
                correlation_id=(
                    correlation_id
                ),
                status=(
                    "INSUFFICIENT_EVIDENCE"
                ),
                answer="REFUSAL",
                citations=(
                    ValidatedCitationSet(
                        citations=()
                    )
                ),
                cited_evidence_ids=(),
            )

        return GroundedAnswerResult(
            correlation_id=(
                correlation_id
            ),
            status="SUPPORTED",
            answer=claim.text,
            citations=(
                ValidatedCitationSet(
                    citations=()
                )
            ),
            cited_evidence_ids=(
                claim.claim_id,
            ),
        )


def lookup_request(
    *,
    ordinal=5,
):
    return TableLookupRequest(
        section_key="LOT_2",
        ordinal=ordinal,
        column_name=(
            "Soumissionnaires"
        ),
    )


def test_orchestrates_authorized_lookup_before_claim_and_answer():
    events = []

    selection = object()

    claim = type(
        "TestClaim",
        (),
        {
            "text": (
                "ENTREPRISE ZERBO AWA "
                "ET FRERES (EZAF)"
            ),
            "claim_id": (
                "table-claim-1"
            ),
        },
    )()

    evidence = RecordingEvidence(
        selection=selection,
        events=events,
    )

    builder = RecordingClaimBuilder(
        claim=claim,
        events=events,
    )

    answer = RecordingAnswer(
        events=events,
    )

    identity = object()

    request = lookup_request()

    correlation_id = uuid4()

    result = (
        TableQuestionOrchestrationService(
            evidence=evidence,
            claim_builder=builder,
            answer=answer,
        )
        .answer_lookup(
            identity=identity,
            question=(
                "Quel soumissionnaire "
                "correspond au numero "
                "05 du lot 2 ?"
            ),
            request=request,
            language="fr",
            correlation_id=(
                correlation_id
            ),
        )
    )

    assert events == [
        "authorized_lookup",
        "claim_builder",
        "answer",
    ]

    assert (
        evidence.calls
        == [
            (
                identity,
                request,
            )
        ]
    )

    assert (
        builder.calls
        == [
            (
                request,
                selection,
            )
        ]
    )

    assert (
        answer.calls[0][
            "claim"
        ]
        is claim
    )

    assert (
        result.correlation_id
        == correlation_id
    )


def test_missing_table_evidence_skips_claim_builder():
    events = []

    evidence = RecordingEvidence(
        selection=None,
        events=events,
    )

    builder = NeverClaimBuilder(
        events=events,
    )

    answer = RecordingAnswer(
        events=events,
    )

    correlation_id = uuid4()

    result = (
        TableQuestionOrchestrationService(
            evidence=evidence,
            claim_builder=builder,
            answer=answer,
        )
        .answer_lookup(
            identity=object(),
            question=(
                "Quel soumissionnaire "
                "correspond au numero "
                "09 du lot 2 ?"
            ),
            request=lookup_request(
                ordinal=9
            ),
            correlation_id=(
                correlation_id
            ),
        )
    )

    assert events == [
        "authorized_lookup",
        "answer",
    ]

    assert builder.calls == []

    assert (
        answer.calls[0][
            "claim"
        ]
        is None
    )

    assert (
        result.status
        == "INSUFFICIENT_EVIDENCE"
    )

    assert (
        result.correlation_id
        == correlation_id
    )


def test_normalizes_language_before_answer():
    events = []

    evidence = RecordingEvidence(
        selection=None,
        events=events,
    )

    answer = RecordingAnswer(
        events=events,
    )

    (
        TableQuestionOrchestrationService(
            evidence=evidence,
            claim_builder=(
                NeverClaimBuilder(
                    events=events
                )
            ),
            answer=answer,
        )
        .answer_lookup(
            identity=object(),
            question="Question",
            request=lookup_request(),
            language=" FR ",
        )
    )

    assert (
        answer.calls[0][
            "language"
        ]
        == "fr"
    )


def test_preserves_one_correlation_id_through_pipeline():
    events = []

    correlation_id = uuid4()

    answer = RecordingAnswer(
        events=events,
    )

    result = (
        TableQuestionOrchestrationService(
            evidence=RecordingEvidence(
                selection=None,
                events=events,
            ),
            claim_builder=(
                NeverClaimBuilder(
                    events=events
                )
            ),
            answer=answer,
        )
        .answer_lookup(
            identity=object(),
            question="Question",
            request=lookup_request(),
            correlation_id=(
                correlation_id
            ),
        )
    )

    assert (
        answer.calls[0][
            "correlation_id"
        ]
        == correlation_id
    )

    assert (
        result.correlation_id
        == correlation_id
    )
