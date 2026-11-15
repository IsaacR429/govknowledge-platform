from __future__ import annotations

from uuid import uuid4

from app.application.ports.audit import (
    AuditEvent,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.question_claim_planner import (
    ClaimPlanningEvidence,
    QuestionClaimPlanner,
    QuestionClaimPlanningRequest,
)
from app.application.services.atomic_evidence_orchestration import (
    AtomicClaim,
    AtomicEvidenceOrchestrationService,
)
from app.application.services.claim_provenance import (
    ClaimOrigin,
)
from app.application.services.audit import (
    AuditPersistenceError,
    AuditService,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.authorized_search import (
    AuthorizedSearchService,
)
from app.application.services.citation_validation import (
    InvalidCitationError,
)
from app.application.services.grounded_answer_generation import (
    GroundedAnswerGenerationService,
    GroundedAnswerResult,
    InvalidGeneratedCitationError,
    MissingGeneratedCitationError,
)
from app.application.services.table_question_answer import (
    TableQuestionAnswerService,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)


MAX_PLANNED_CLAIMS = 5


class InvalidPlannedClaimError(
    RuntimeError
):
    pass


class QuestionAnswerService:
    def __init__(
        self,
        *,
        search: AuthorizedSearchService,
        planner: QuestionClaimPlanner,
        evidence: AtomicEvidenceOrchestrationService,
        answer: GroundedAnswerGenerationService,
        authorization: AuthorizationService,
        audit: AuditService,
        discovery_top_k: int = 10,
        table_answer: TableQuestionAnswerService | None = None,
    ) -> None:
        if discovery_top_k < 1:
            raise ValueError(
                "discovery_top_k must be positive."
            )

        self._search = search
        self._planner = planner
        self._evidence = evidence
        self._answer = answer
        self._authorization = authorization
        self._audit = audit
        self._discovery_top_k = discovery_top_k
        self._table_answer = table_answer

    def _record_answer_outcome(
        self,
        *,
        identity: CurrentIdentity,
        result: GroundedAnswerResult,
    ) -> None:
        if result.status == "SUPPORTED":
            event_type = (
                AuditEventType.ANSWER_GENERATED
            )
        elif (
            result.status
            == "INSUFFICIENT_EVIDENCE"
        ):
            event_type = (
                AuditEventType.ANSWER_REFUSED
            )
        else:
            raise RuntimeError(
                "Unsupported grounded answer status."
            )

        self._audit.record(
            AuditEvent(
                event_type=event_type,
                outcome=AuditOutcome.SUCCEEDED,
                actor_user_id=identity.user_id,
                resource_type="qa_answer",
                correlation_id=(
                    result.correlation_id
                ),
                metadata={
                    "status": result.status,
                },
            )
        )

    def _record_validated_citations(
        self,
        *,
        identity: CurrentIdentity,
        result: GroundedAnswerResult,
    ) -> None:
        for item in result.citations.citations:
            citation = item.citation

            if hasattr(
                citation,
                "chunk_id",
            ):
                # Preserve the established text-citation
                # audit metadata exactly.
                metadata = {
                    "claim_id": item.claim_id,
                    "chunk_id": str(
                        citation.chunk_id
                    ),
                    "document_version_id": str(
                        citation.document_version_id
                    ),
                    "page_number": (
                        citation.page_number
                    ),
                }

            else:
                # Typed-table citation.
                #
                # No fake chunk_id or character offsets
                # are introduced.
                metadata = {
                    "claim_id": item.claim_id,
                    "table_evidence_id": str(
                        citation.evidence_id
                    ),
                    "logical_key": (
                        citation.logical_key
                    ),
                    "column_name": (
                        citation.column_name
                    ),
                    "document_version_id": str(
                        citation.document_version_id
                    ),
                    "page_number": (
                        citation.page_number
                    ),
                }

            self._audit.record(
                AuditEvent(
                    event_type=(
                        AuditEventType.CITATION_VALIDATED
                    ),
                    outcome=AuditOutcome.SUCCEEDED,
                    actor_user_id=identity.user_id,
                    resource_type="citation",
                    resource_id=citation.page_id,
                    document_id=citation.document_id,
                    correlation_id=(
                        result.correlation_id
                    ),
                    metadata=metadata,
                )
            )

    def _record_citation_rejection(
        self,
        *,
        identity: CurrentIdentity,
        correlation_id,
        error_code: str,
        stage: str,
    ) -> None:
        try:
            self._audit.record(
                AuditEvent(
                    event_type=(
                        AuditEventType.CITATION_REJECTED
                    ),
                    outcome=AuditOutcome.FAILED,
                    actor_user_id=identity.user_id,
                    resource_type="qa_citation",
                    correlation_id=correlation_id,
                    error_code=error_code,
                    metadata={
                        "stage": stage,
                    },
                )
            )
        except AuditPersistenceError:
            # The citation was already rejected.
            # Audit failure must never turn a
            # rejected citation into a disclosure.
            pass

    def answer(
        self,
        *,
        identity: CurrentIdentity,
        question: str,
        language: str = "fr",
    ) -> GroundedAnswerResult:
        clean_question = question.strip()
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

        correlation_id = uuid4()

        # --------------------------------------------------
        # STRUCTURED TABLE ROUTE
        #
        # Table routing is attempted before ordinary text
        # discovery.
        #
        # None:
        #   planner declined the table path, so the existing
        #   text-RAG pipeline continues unchanged.
        #
        # GroundedAnswerResult:
        #   the table path owns the answer. This includes
        #   INSUFFICIENT_EVIDENCE and therefore MUST NOT
        #   fall through to vector retrieval.
        # --------------------------------------------------

        if self._table_answer is not None:

            try:
                table_result = (
                    self._table_answer
                    .try_answer(
                        identity=identity,
                        question=clean_question,
                        language=clean_language,
                        correlation_id=(
                            correlation_id
                        ),
                    )
                )

            except Exception:
                self._audit.record(
                    AuditEvent(
                        event_type=(
                            AuditEventType.EVIDENCE_FAILED
                        ),
                        outcome=AuditOutcome.FAILED,
                        actor_user_id=identity.user_id,
                        resource_type="qa_evidence",
                        correlation_id=(
                            correlation_id
                        ),
                        error_code=(
                            "TABLE_ROUTE_FAILED"
                        ),
                        metadata={
                            "stage": (
                                "table_question_route"
                            ),
                        },
                    )
                )

                raise

            if table_result is not None:

                self._record_validated_citations(
                    identity=identity,
                    result=table_result,
                )

                self._record_answer_outcome(
                    identity=identity,
                    result=table_result,
                )

                return table_result

        # Authorization happens inside
        # AuthorizedSearchService before retrieval.
        try:
            discovery = self._search.search(
                identity=identity,
                query=clean_question,
                top_k=self._discovery_top_k,
            )
        except Exception:
            self._audit.record(
                AuditEvent(
                    event_type=(
                        AuditEventType.EVIDENCE_FAILED
                    ),
                    outcome=AuditOutcome.FAILED,
                    actor_user_id=identity.user_id,
                    resource_type="qa_evidence",
                    correlation_id=correlation_id,
                    error_code=(
                        "EVIDENCE_DISCOVERY_FAILED"
                    ),
                    metadata={
                        "stage": "authorized_discovery",
                    },
                )
            )

            raise

        # No authorized evidence means there is
        # nothing that may be planned or answered.
        if not discovery.results:
            evidence_result = (
                self._evidence.evaluate(
                    identity=identity,
                    claims=(),
                    correlation_id=correlation_id,
                )
            )

            result = self._answer.generate(
                question=clean_question,
                evidence_result=evidence_result,
                language=clean_language,
            )

            self._record_answer_outcome(
                identity=identity,
                result=result,
            )

            return result

        # P0 rule:
        # the highest-ranked authorized result
        # determines the source language used for
        # claim planning and later NLI.
        source_language = (
            discovery.results[0]
            .language
            .strip()
            .lower()
        )

        if not source_language:
            raise RuntimeError(
                "Retrieved evidence has no language."
            )

        source_items = tuple(
            item
            for item in discovery.results
            if (
                item.language
                .strip()
                .lower()
                == source_language
            )
        )

        planning_pairs = tuple(
            (
                f"evidence-{index}",
                item,
            )
            for index, item in enumerate(
                source_items,
                start=1,
            )
        )

        planning_evidence = tuple(
            ClaimPlanningEvidence(
                evidence_id=evidence_id,
                chunk_id=item.chunk_id,
                language=item.language,
                content=item.content,
            )
            for evidence_id, item
            in planning_pairs
        )

        planning_sources = {
            evidence_id: item
            for evidence_id, item
            in planning_pairs
        }

        planned = self._planner.plan(
            QuestionClaimPlanningRequest(
                question=clean_question,
                question_language=clean_language,
                source_language=source_language,
                evidence=planning_evidence,
            )
        )

        if len(planned) > MAX_PLANNED_CLAIMS:
            raise InvalidPlannedClaimError(
                "Planner returned too many claims."
            )

        atomic_claims = []

        seen = set()

        for index, item in enumerate(
            planned,
            start=1,
        ):
            evidence_id = item.evidence_id.strip()
            claim_text = item.text.strip()

            if not claim_text:
                raise InvalidPlannedClaimError(
                    "Planner returned an empty claim."
                )

            if claim_text in seen:
                raise InvalidPlannedClaimError(
                    "Planner returned duplicate claims."
                )

            seen.add(claim_text)

            source = planning_sources.get(
                evidence_id
            )

            if source is None:
                raise InvalidPlannedClaimError(
                    "Planner referenced unknown evidence."
                )

            start_char = source.content.find(
                claim_text
            )

            if start_char < 0:
                raise InvalidPlannedClaimError(
                    "Planner quote is not present in "
                    "the selected authorized evidence."
                )

            second_match = source.content.find(
                claim_text,
                start_char + 1,
            )

            if second_match >= 0:
                raise InvalidPlannedClaimError(
                    "Planner quote is ambiguous within "
                    "the selected authorized evidence."
                )

            end_char = (
                start_char
                + len(claim_text)
            )

            origin = ClaimOrigin(
                evidence_id=evidence_id,
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                document_version_id=(
                    source.document_version_id
                ),
                page_id=source.page_id,
                page_number=source.page_number,
                start_char=start_char,
                end_char=end_char,
                text=claim_text,
                language=source.language,
            )

            atomic_claims.append(
                AtomicClaim(
                    claim_id=(
                        f"claim-{index}"
                    ),
                    text=claim_text,
                    origin=origin,
                    language=source_language,
                    required=True,
                )
            )

        evidence_result = (
            self._evidence.evaluate(
                identity=identity,
                claims=tuple(
                    atomic_claims
                ),
                correlation_id=correlation_id,
            )
        )

        if evidence_result.assessment.is_sufficient:
            claims_by_id = {
                item.claim.claim_id: item.claim
                for item in evidence_result.claim_results
            }

            for selected in (
                evidence_result
                .assessment
                .selected_evidence
            ):
                claim = claims_by_id.get(
                    selected.claim_id
                )

                if claim is None:
                    raise RuntimeError(
                        "Selected evidence has no "
                        "matching atomic claim."
                    )

                self._authorization.require_document_read(
                    identity,
                    claim.origin.document_id,
                )

        try:
            result = self._answer.generate(
                question=clean_question,
                evidence_result=evidence_result,
                language=clean_language,
            )

        except InvalidCitationError:
            self._record_citation_rejection(
                identity=identity,
                correlation_id=correlation_id,
                error_code=(
                    "INVALID_CANONICAL_CITATION"
                ),
                stage="canonical_validation",
            )
            raise

        except MissingGeneratedCitationError:
            self._record_citation_rejection(
                identity=identity,
                correlation_id=correlation_id,
                error_code=(
                    "MISSING_GENERATED_CITATION"
                ),
                stage="generated_answer_validation",
            )
            raise

        except InvalidGeneratedCitationError:
            self._record_citation_rejection(
                identity=identity,
                correlation_id=correlation_id,
                error_code=(
                    "INVALID_GENERATED_CITATION"
                ),
                stage="generated_answer_validation",
            )
            raise

        self._record_validated_citations(
            identity=identity,
            result=result,
        )

        self._record_answer_outcome(
            identity=identity,
            result=result,
        )

        return result
