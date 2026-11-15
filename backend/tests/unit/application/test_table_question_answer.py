from uuid import uuid4

from app.application.ports.table_evidence_repository import (
    AuthorizedTableVocabulary,
)
from app.application.services.grounded_answer_generation import (
    GroundedAnswerResult,
)
from app.application.services.table_evidence_atoms import (
    TableLookupRequest,
)
from app.application.services.table_question_answer import (
    TableQuestionAnswerService,
)
from app.application.services.validated_citations import (
    ValidatedCitationSet,
)


class RecordingEvidence:
    def __init__(
        self,
        *,
        vocabulary,
        events,
    ):
        self._vocabulary = (
            vocabulary
        )
        self.events = events
        self.calls = []

    def vocabulary(
        self,
        *,
        identity,
    ):
        self.events.append(
            "vocabulary"
        )

        self.calls.append(
            identity
        )

        return self._vocabulary


class RecordingInterpreter:
    def __init__(
        self,
        *,
        request,
        events,
    ):
        self.request = request
        self.events = events
        self.calls = []

    def interpret(
        self,
        **kwargs,
    ):
        self.events.append(
            "interpret"
        )

        self.calls.append(
            kwargs
        )

        return self.request


class NeverInterpreter:
    def __init__(
        self,
        *,
        events,
    ):
        self.events = events
        self.calls = []

    def interpret(
        self,
        **kwargs,
    ):
        self.events.append(
            "interpret"
        )

        self.calls.append(
            kwargs
        )

        raise AssertionError(
            "Interpreter must not run "
            "without authorized table vocabulary."
        )


class RecordingOrchestration:
    def __init__(
        self,
        *,
        events,
    ):
        self.events = events
        self.calls = []

    def answer_lookup(
        self,
        *,
        identity,
        question,
        request,
        language,
        correlation_id,
    ):
        self.events.append(
            "orchestration"
        )

        self.calls.append(
            {
                "identity": identity,
                "question": question,
                "request": request,
                "language": language,
                "correlation_id": (
                    correlation_id
                ),
            }
        )

        return GroundedAnswerResult(
            correlation_id=(
                correlation_id
            ),
            status="SUPPORTED",
            answer=(
                "ENTREPRISE ZERBO AWA "
                "ET FRERES (EZAF)"
            ),
            citations=(
                ValidatedCitationSet(
                    citations=()
                )
            ),
            cited_evidence_ids=(),
        )


class NeverOrchestration:
    def __init__(
        self,
        *,
        events,
    ):
        self.events = events
        self.calls = []

    def answer_lookup(
        self,
        **kwargs,
    ):
        self.events.append(
            "orchestration"
        )

        self.calls.append(
            kwargs
        )

        raise AssertionError(
            "Orchestration must not run."
        )


def vocabulary():
    return AuthorizedTableVocabulary(
        section_keys=(
            "LOT_1",
            "LOT_2",
        ),
        column_names=(
            "N°",
            "Soumissionnaires",
            "Observations",
        ),
    )


def request():
    return TableLookupRequest(
        section_key="LOT_2",
        ordinal=5,
        column_name=(
            "Soumissionnaires"
        ),
    )


def test_complete_table_question_route():
    events = []

    evidence = RecordingEvidence(
        vocabulary=vocabulary(),
        events=events,
    )

    interpreter = (
        RecordingInterpreter(
            request=request(),
            events=events,
        )
    )

    orchestration = (
        RecordingOrchestration(
            events=events
        )
    )

    identity = object()
    correlation_id = uuid4()

    result = (
        TableQuestionAnswerService(
            evidence=evidence,
            interpreter=interpreter,
            orchestration=(
                orchestration
            ),
        )
        .try_answer(
            identity=identity,
            question=(
                "Quel soumissionnaire "
                "correspond au numero "
                "05 du lot 2 ?"
            ),
            language="fr",
            correlation_id=(
                correlation_id
            ),
        )
    )

    assert events == [
        "vocabulary",
        "interpret",
        "orchestration",
    ]

    assert result is not None

    assert (
        result.status
        == "SUPPORTED"
    )

    assert (
        result.correlation_id
        == correlation_id
    )

    interpreter_call = (
        interpreter.calls[0]
    )

    assert (
        interpreter_call[
            "section_keys"
        ]
        == (
            "LOT_1",
            "LOT_2",
        )
    )

    assert (
        interpreter_call[
            "column_names"
        ]
        == (
            "N°",
            "Soumissionnaires",
            "Observations",
        )
    )

    assert (
        orchestration.calls[0][
            "request"
        ]
        == request()
    )


def test_no_authorized_table_vocabulary_returns_none():
    events = []

    result = (
        TableQuestionAnswerService(
            evidence=RecordingEvidence(
                vocabulary=(
                    AuthorizedTableVocabulary
                    .empty()
                ),
                events=events,
            ),
            interpreter=(
                NeverInterpreter(
                    events=events
                )
            ),
            orchestration=(
                NeverOrchestration(
                    events=events
                )
            ),
        )
        .try_answer(
            identity=object(),
            question="Question",
            language="fr",
            correlation_id=uuid4(),
        )
    )

    assert result is None

    assert events == [
        "vocabulary",
    ]


def test_planner_declines_table_question_and_allows_fallback():
    events = []

    interpreter = (
        RecordingInterpreter(
            request=None,
            events=events,
        )
    )

    result = (
        TableQuestionAnswerService(
            evidence=RecordingEvidence(
                vocabulary=vocabulary(),
                events=events,
            ),
            interpreter=interpreter,
            orchestration=(
                NeverOrchestration(
                    events=events
                )
            ),
        )
        .try_answer(
            identity=object(),
            question=(
                "Resume les principales "
                "obligations du document."
            ),
            language="fr",
            correlation_id=uuid4(),
        )
    )

    assert result is None

    assert events == [
        "vocabulary",
        "interpret",
    ]


def test_valid_table_request_owns_insufficient_result():
    events = []

    class InsufficientOrchestration:
        def __init__(self):
            self.calls = []

        def answer_lookup(
            self,
            *,
            identity,
            question,
            request,
            language,
            correlation_id,
        ):
            events.append(
                "orchestration"
            )

            self.calls.append(
                request
            )

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

    orchestration = (
        InsufficientOrchestration()
    )

    result = (
        TableQuestionAnswerService(
            evidence=RecordingEvidence(
                vocabulary=vocabulary(),
                events=events,
            ),
            interpreter=(
                RecordingInterpreter(
                    request=(
                        TableLookupRequest(
                            section_key="LOT_2",
                            ordinal=9,
                            column_name=(
                                "Soumissionnaires"
                            ),
                        )
                    ),
                    events=events,
                )
            ),
            orchestration=(
                orchestration
            ),
        )
        .try_answer(
            identity=object(),
            question=(
                "Quel soumissionnaire "
                "correspond au numero "
                "09 du lot 2 ?"
            ),
            language="fr",
            correlation_id=uuid4(),
        )
    )

    assert result is not None

    assert (
        result.status
        == "INSUFFICIENT_EVIDENCE"
    )

    # Important:
    # once a valid structured lookup was selected,
    # we do NOT fall back to text retrieval.
    assert events == [
        "vocabulary",
        "interpret",
        "orchestration",
    ]


def test_language_is_normalized_before_interpretation():
    events = []

    interpreter = (
        RecordingInterpreter(
            request=None,
            events=events,
        )
    )

    (
        TableQuestionAnswerService(
            evidence=RecordingEvidence(
                vocabulary=vocabulary(),
                events=events,
            ),
            interpreter=interpreter,
            orchestration=(
                NeverOrchestration(
                    events=events
                )
            ),
        )
        .try_answer(
            identity=object(),
            question="Question",
            language=" FR ",
            correlation_id=uuid4(),
        )
    )

    assert (
        interpreter.calls[0][
            "language"
        ]
        == "fr"
    )
