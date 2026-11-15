from __future__ import annotations

from uuid import UUID

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.authorized_table_evidence import (
    AuthorizedTableEvidenceService,
)
from app.application.services.grounded_answer_generation import (
    GroundedAnswerResult,
)
from app.application.services.table_question_interpretation import (
    ValidatedTableQuestionInterpreter,
)
from app.application.services.table_question_orchestration import (
    TableQuestionOrchestrationService,
)


class TableQuestionAnswerService:
    """
    Application bridge between:

        authorized table vocabulary
            ↓
        natural-language interpretation
            ↓
        canonical table lookup pipeline

    Returning None means:

        "This question was not interpreted as a
         structured table lookup."

    The caller may then continue through the normal
    text-RAG path.

    A GroundedAnswerResult means that a table lookup
    WAS selected and therefore owns the answer,
    including an insufficient-evidence result.
    """

    def __init__(
        self,
        *,
        evidence: (
            AuthorizedTableEvidenceService
        ),
        interpreter: (
            ValidatedTableQuestionInterpreter
        ),
        orchestration: (
            TableQuestionOrchestrationService
        ),
    ) -> None:

        self._evidence = evidence
        self._interpreter = (
            interpreter
        )
        self._orchestration = (
            orchestration
        )

    def try_answer(
        self,
        *,
        identity: CurrentIdentity,
        question: str,
        language: str,
        correlation_id: UUID,
    ) -> GroundedAnswerResult | None:

        clean_question = (
            question.strip()
        )

        clean_language = (
            language
            .strip()
            .lower()
        )

        if not clean_question:
            raise ValueError(
                "question cannot be empty."
            )

        if not clean_language:
            raise ValueError(
                "language cannot be empty."
            )

        # --------------------------------------------
        # Authorization-first structural discovery.
        # --------------------------------------------

        vocabulary = (
            self._evidence.vocabulary(
                identity=identity
            )
        )

        # --------------------------------------------
        # No visible table structure means there is
        # no table interpretation path.
        #
        # Caller remains free to use ordinary text RAG.
        # --------------------------------------------

        has_procurement = bool(
            vocabulary.section_keys
            and vocabulary.column_names
        )

        has_generic = bool(
            vocabulary.generic_tables
        )

        if (
            not has_procurement
            and not has_generic
        ):
            return None

        # --------------------------------------------
        # Untrusted planner output is validated against
        # the authorized structural vocabulary.
        # --------------------------------------------

        request = (
            self._interpreter.interpret(
                question=(
                    clean_question
                ),
                language=(
                    clean_language
                ),
                section_keys=(
                    vocabulary.section_keys
                ),
                column_names=(
                    vocabulary.column_names
                ),
                generic_tables=(
                    vocabulary.generic_tables
                ),
            )
        )

        # Planner explicitly declined table routing.
        if request is None:
            return None

        # --------------------------------------------
        # Once a valid table lookup request exists,
        # the typed-table path owns the response.
        #
        # Missing row/cell becomes insufficient
        # evidence rather than falling back to a
        # different corpus path and guessing.
        # --------------------------------------------

        return (
            self._orchestration
            .answer_lookup(
                identity=identity,
                question=(
                    clean_question
                ),
                request=request,
                language=(
                    clean_language
                ),
                correlation_id=(
                    correlation_id
                ),
            )
        )
