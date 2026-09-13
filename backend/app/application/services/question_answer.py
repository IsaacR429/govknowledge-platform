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
from app.application.services.audit import (
    AuditService,
)
from app.application.services.authorized_search import (
    AuthorizedSearchService,
)
from app.application.services.grounded_answer_generation import (
    GroundedAnswerGenerationService,
    GroundedAnswerResult,
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
        audit: AuditService,
        discovery_top_k: int = 10,
    ) -> None:
        if discovery_top_k < 1:
            raise ValueError(
                "discovery_top_k must be positive."
            )

        self._search = search
        self._planner = planner
        self._evidence = evidence
        self._answer = answer
        self._audit = audit
        self._discovery_top_k = discovery_top_k

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

            return self._answer.generate(
                question=clean_question,
                evidence_result=evidence_result,
                language=clean_language,
            )

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

        planning_evidence = tuple(
            ClaimPlanningEvidence(
                chunk_id=item.chunk_id,
                language=item.language,
                content=item.content,
            )
            for item in discovery.results
            if (
                item.language
                .strip()
                .lower()
                == source_language
            )
        )

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

            # Hard grounding rule:
            # planner output must literally occur in
            # authorized source-language evidence.
            if not any(
                claim_text in evidence_item.content
                for evidence_item
                in planning_evidence
            ):
                raise InvalidPlannedClaimError(
                    "Planner returned a claim that "
                    "is not present in authorized "
                    "source evidence."
                )

            atomic_claims.append(
                AtomicClaim(
                    claim_id=(
                        f"claim-{index}"
                    ),
                    text=claim_text,
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

        return self._answer.generate(
            question=clean_question,
            evidence_result=evidence_result,
            language=clean_language,
        )
