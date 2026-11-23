from __future__ import annotations

from uuid import (
    UUID,
    uuid4,
)

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.authorized_table_evidence import (
    AuthorizedTableEvidenceService,
)
from app.application.services.grounded_answer_generation import (
    GroundedAnswerResult,
)
from app.application.services.table_claim_provenance import (
    DirectTableClaimBuilder,
)
from app.application.services.table_evidence_atoms import (
    LabeledRowComparisonRequest,
    LabeledRowLookupRequest,
    TableLookupRequest,
)
from app.application.services.table_grounded_answer_generation import (
    TableGroundedAnswerGenerationService,
)


class TableQuestionOrchestrationService:
    """
    Reusable P0 application pipeline for an already
    interpreted structured-table lookup.

    Natural-language interpretation deliberately does
    NOT live here.

    Security/provenance order:

        identity
          ↓
        authorized typed lookup
          ↓
        validated table selection
          ↓
        typed claim origin
          ↓
        persisted citation revalidation
          ↓
        deterministic grounded answer
    """

    def __init__(
        self,
        *,
        evidence: (
            AuthorizedTableEvidenceService
        ),
        claim_builder: (
            DirectTableClaimBuilder
        ),
        answer: (
            TableGroundedAnswerGenerationService
        ),
    ) -> None:

        self._evidence = evidence

        self._claim_builder = (
            claim_builder
        )

        self._answer = answer

    def answer_lookup(
        self,
        *,
        identity: CurrentIdentity,
        question: str,
        request: (
            TableLookupRequest
            | LabeledRowLookupRequest
            | LabeledRowComparisonRequest
        ),
        language: str = "fr",
        correlation_id: UUID | None = None,
    ) -> GroundedAnswerResult:

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

        operation_id = (
            correlation_id
            if correlation_id is not None
            else uuid4()
        )

        # --------------------------------------------
        # Authorization-first typed evidence lookup.
        # --------------------------------------------

        selection = (
            self._evidence.lookup(
                identity=identity,
                request=request,
            )
        )

        # --------------------------------------------
        # No canonical row/cell is a normal
        # insufficient-evidence outcome.
        # --------------------------------------------

        if selection is None:
            return (
                self._answer.generate(
                    identity=identity,
                    question=clean_question,
                    claim=None,
                    correlation_id=(
                        operation_id
                    ),
                    language=(
                        clean_language
                    ),
                )
            )

        # --------------------------------------------
        # Multi-column generic comparison.
        #
        # Retrieval already guarantees that all cells
        # belong to one resolved logical row.
        # Each cell is provenance-validated separately
        # before disclosure.
        # --------------------------------------------

        if isinstance(
            request,
            LabeledRowComparisonRequest,
        ):

            claims = (
                self._claim_builder
                .build_comparison(
                    request=request,
                    selection=selection,
                )
            )

            return (
                self._answer
                .generate_comparison(
                    identity=identity,
                    question=clean_question,
                    claims=claims,
                    correlation_id=(
                        operation_id
                    ),
                    language=(
                        clean_language
                    ),
                )
            )

        # --------------------------------------------
        # Build provenance from the selected
        # canonical typed cell.
        # --------------------------------------------

        claim = (
            self._claim_builder.build(
                request=request,
                selection=selection,
            )
        )

        # --------------------------------------------
        # Answer service revalidates persistence +
        # ACL again immediately before disclosure.
        # --------------------------------------------

        return self._answer.generate(
            identity=identity,
            question=clean_question,
            claim=claim,
            correlation_id=(
                operation_id
            ),
            language=clean_language,
        )
