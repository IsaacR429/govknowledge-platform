from __future__ import annotations

from uuid import uuid4

import pytest

from app.application.ports.entailment_verifier import (
    EntailmentScore,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.retrieval import (
    RetrievedChunk,
)
from app.application.services.atomic_evidence_orchestration import (
    AtomicClaim,
    AtomicEvidenceOrchestrationService,
)
from app.application.services.atomic_evidence_policy import (
    AtomicEvidencePolicy,
)
from app.application.services.authorized_search import (
    AuthorizedSearchResult,
)
from app.application.services.sentence_evidence_verification import (
    SentenceEvidenceVerificationService,
)
from app.domain.enums.evidence import (
    EntailmentLabel,
    EvidenceDecision,
)


def chunk(
    *,
    text: str,
    index: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=index + 1,
        chunk_index=index,
        content=text,
        section_title=None,
        language="fr",
        score=0.85,
    )


def score(
    label: EntailmentLabel,
) -> EntailmentScore:
    if (
        label
        == EntailmentLabel.ENTAILMENT
    ):
        values = (
            0.99,
            0.009,
            0.001,
        )

    elif (
        label
        == EntailmentLabel.NEUTRAL
    ):
        values = (
            0.01,
            0.98,
            0.01,
        )

    else:
        values = (
            0.001,
            0.009,
            0.99,
        )

    return EntailmentScore(
        label=label,
        entailment_probability=(
            values[0]
        ),
        neutral_probability=(
            values[1]
        ),
        contradiction_probability=(
            values[2]
        ),
    )


class FakeAuthorizedSearch:
    def __init__(
        self,
        *,
        results_by_query=None,
        fail_on_query=None,
    ):
        self.results_by_query = (
            results_by_query
            or {}
        )

        self.fail_on_query = (
            fail_on_query
        )

        self.calls = []

    def search(
        self,
        *,
        identity,
        query,
        top_k=5,
    ):
        self.calls.append(
            (
                identity,
                query,
                top_k,
            )
        )

        if query == self.fail_on_query:
            raise RuntimeError(
                "Authorization boundary "
                "rejected retrieval."
            )

        return AuthorizedSearchResult(
            query=query,
            results=tuple(
                self.results_by_query.get(
                    query,
                    (),
                )
            ),
        )


class FakeEntailmentVerifier:
    model_name = "fake-nli"
    model_revision = "test"

    def __init__(
        self,
        *,
        labels_by_claim,
    ):
        self.labels_by_claim = (
            labels_by_claim
        )

        self.calls = []

    def score(
        self,
        *,
        claim,
        premises,
    ):
        premises = tuple(
            premises
        )

        self.calls.append(
            (
                claim,
                premises,
            )
        )

        labels = tuple(
            self.labels_by_claim[
                claim
            ]
        )

        if len(labels) != len(
            premises
        ):
            raise AssertionError(
                "Test fixture label count "
                "does not match premises."
            )

        return tuple(
            score(label)
            for label in labels
        )



class FakeAuditService:
    def __init__(
        self,
        *,
        fail_on_call=None,
    ):
        self.calls = []
        self.fail_on_call = fail_on_call

    def record(
        self,
        event,
    ):
        self.calls.append(event)

        if (
            self.fail_on_call is not None
            and len(self.calls)
            == self.fail_on_call
        ):
            from app.application.services.audit import (
                AuditPersistenceError,
            )

            raise AuditPersistenceError(
                "Audit unavailable."
            )

        return uuid4()


def current_identity():
    return CurrentIdentity(
        user_id=uuid4(),
        username="employee",
        display_name="Employee",
        email=None,
        department="Finance",
        language_preference="fr",
        roles=("EMPLOYEE",),
    )



def service(
    *,
    search,
    verifier,
    audit=None,
    top_k=5,
):
    return (
        AtomicEvidenceOrchestrationService(
            search=search,
            verification=(
                SentenceEvidenceVerificationService(
                    verifier=verifier
                )
            ),
            policy=AtomicEvidencePolicy(),
            audit=(
                audit
                if audit is not None
                else FakeAuditService()
            ),
            top_k=top_k,
        )
    )


def test_retrieves_each_atomic_claim_independently():
    claim_a = "Claim A"
    claim_b = "Claim B"

    evidence_a = chunk(
        text="Evidence for A.",
        index=0,
    )

    evidence_b = chunk(
        text="Evidence for B.",
        index=1,
    )

    search = FakeAuthorizedSearch(
        results_by_query={
            claim_a: (evidence_a,),
            claim_b: (evidence_b,),
        }
    )

    verifier = FakeEntailmentVerifier(
        labels_by_claim={
            claim_a: (
                EntailmentLabel
                .ENTAILMENT,
            ),
            claim_b: (
                EntailmentLabel
                .ENTAILMENT,
            ),
        }
    )

    identity = current_identity()

    result = service(
        search=search,
        verifier=verifier,
        top_k=3,
    ).evaluate(
        identity=identity,
        claims=(
            AtomicClaim(
                claim_id="A",
                text=claim_a,
            ),
            AtomicClaim(
                claim_id="B",
                text=claim_b,
            ),
        ),
    )

    assert result.is_sufficient

    assert (
        result.assessment.decision
        == EvidenceDecision.SUFFICIENT
    )

    assert [
        call[1:]
        for call in search.calls
    ] == [
        (
            claim_a,
            3,
        ),
        (
            claim_b,
            3,
        ),
    ]

    assert [
        item.claim.claim_id
        for item in result.claim_results
    ] == [
        "A",
        "B",
    ]

    assert (
        result.claim_results[0]
        .retrieved_evidence
        == (evidence_a,)
    )

    assert (
        result.claim_results[1]
        .retrieved_evidence
        == (evidence_b,)
    )


def test_weakest_required_claim_blocks_answer():
    claim_a = "Supported claim"
    claim_b = "Unsupported claim"

    search = FakeAuthorizedSearch(
        results_by_query={
            claim_a: (
                chunk(
                    text="Supports A.",
                    index=0,
                ),
            ),
            claim_b: (
                chunk(
                    text="Does not support B.",
                    index=1,
                ),
            ),
        }
    )

    verifier = FakeEntailmentVerifier(
        labels_by_claim={
            claim_a: (
                EntailmentLabel
                .ENTAILMENT,
            ),
            claim_b: (
                EntailmentLabel
                .NEUTRAL,
            ),
        }
    )

    result = service(
        search=search,
        verifier=verifier,
    ).evaluate(
        identity=current_identity(),
        claims=(
            AtomicClaim(
                claim_id="A",
                text=claim_a,
            ),
            AtomicClaim(
                claim_id="B",
                text=claim_b,
            ),
        ),
    )

    assert not result.is_sufficient

    assert (
        result.assessment.decision
        == EvidenceDecision
        .INSUFFICIENT_SUPPORT
    )

    assert (
        result.assessment
        .unsupported_required_claim_ids
        == ("B",)
    )


def test_missing_evidence_for_one_claim_blocks_answer():
    claim_a = "Supported"
    claim_b = "Missing"

    search = FakeAuthorizedSearch(
        results_by_query={
            claim_a: (
                chunk(
                    text="Supports claim.",
                    index=0,
                ),
            ),
            claim_b: (),
        }
    )

    verifier = FakeEntailmentVerifier(
        labels_by_claim={
            claim_a: (
                EntailmentLabel
                .ENTAILMENT,
            ),
        }
    )

    result = service(
        search=search,
        verifier=verifier,
    ).evaluate(
        identity=current_identity(),
        claims=(
            AtomicClaim(
                claim_id="A",
                text=claim_a,
            ),
            AtomicClaim(
                claim_id="B",
                text=claim_b,
            ),
        ),
    )

    assert (
        result.assessment.decision
        == EvidenceDecision
        .INSUFFICIENT_SUPPORT
    )

    assert (
        result.assessment
        .unsupported_required_claim_ids
        == ("B",)
    )

    # No retrieved sentence means the
    # verifier is never called for B.
    assert [
        call[0]
        for call in verifier.calls
    ] == [
        claim_a,
    ]


def test_no_evidence_for_any_claim_returns_no_evidence():
    search = FakeAuthorizedSearch()

    verifier = FakeEntailmentVerifier(
        labels_by_claim={}
    )

    result = service(
        search=search,
        verifier=verifier,
    ).evaluate(
        identity=current_identity(),
        claims=(
            AtomicClaim(
                claim_id="A",
                text="A",
            ),
            AtomicClaim(
                claim_id="B",
                text="B",
            ),
        ),
    )

    assert (
        result.assessment.decision
        == EvidenceDecision.NO_EVIDENCE
    )

    assert verifier.calls == []


def test_only_authorized_search_results_reach_verifier():
    claim = "Claim"

    authorized = chunk(
        text="Authorized evidence.",
        index=0,
    )

    search = FakeAuthorizedSearch(
        results_by_query={
            claim: (
                authorized,
            ),
        }
    )

    verifier = FakeEntailmentVerifier(
        labels_by_claim={
            claim: (
                EntailmentLabel
                .ENTAILMENT,
            ),
        }
    )

    service(
        search=search,
        verifier=verifier,
    ).evaluate(
        identity=current_identity(),
        claims=(
            AtomicClaim(
                claim_id="A",
                text=claim,
            ),
        ),
    )

    assert len(
        verifier.calls
    ) == 1

    assert (
        verifier.calls[0][1]
        == (
            "Authorized evidence.",
        )
    )


def test_authorization_failure_is_not_swallowed():
    claim = "Restricted claim"

    search = FakeAuthorizedSearch(
        fail_on_query=claim
    )

    verifier = FakeEntailmentVerifier(
        labels_by_claim={}
    )

    with pytest.raises(
        RuntimeError,
        match="Authorization boundary",
    ):
        service(
            search=search,
            verifier=verifier,
        ).evaluate(
            identity=current_identity(),
            claims=(
                AtomicClaim(
                    claim_id="restricted",
                    text=claim,
                ),
            ),
        )

    # Fail closed: no NLI after
    # authorization/search failure.
    assert verifier.calls == []


def test_multi_claim_authorization_failure_aborts_entire_request():
    authorized_claim = "Authorized claim"
    restricted_claim = "Restricted claim"

    search = FakeAuthorizedSearch(
        results_by_query={
            authorized_claim: (
                chunk(
                    text="Authorized evidence.",
                    index=0,
                ),
            ),
        },
        fail_on_query=restricted_claim,
    )

    verifier = FakeEntailmentVerifier(
        labels_by_claim={
            authorized_claim: (
                EntailmentLabel.ENTAILMENT,
            ),
        }
    )

    with pytest.raises(
        RuntimeError,
        match="Authorization boundary",
    ):
        service(
            search=search,
            verifier=verifier,
        ).evaluate(
            identity=current_identity(),
            claims=(
                AtomicClaim(
                    claim_id="authorized",
                    text=authorized_claim,
                ),
                AtomicClaim(
                    claim_id="restricted",
                    text=restricted_claim,
                ),
            ),
        )

    # Claim 1 completed normally before
    # claim 2 hit the authorization boundary.
    assert [
        call[1]
        for call in search.calls
    ] == [
        authorized_claim,
        restricted_claim,
    ]

    # Only claim 1 was allowed to reach NLI.
    # Claim 2 must never be converted into
    # "no evidence" or otherwise verified.
    assert verifier.calls == [
        (
            authorized_claim,
            (
                "Authorized evidence.",
            ),
        ),
    ]


def test_duplicate_claim_ids_fail_before_search():
    search = FakeAuthorizedSearch()

    verifier = FakeEntailmentVerifier(
        labels_by_claim={}
    )

    orchestrator = service(
        search=search,
        verifier=verifier,
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        orchestrator.evaluate(
            identity=current_identity(),
            claims=(
                AtomicClaim(
                    claim_id="same",
                    text="A",
                ),
                AtomicClaim(
                    claim_id="same",
                    text="B",
                ),
            ),
        )

    assert search.calls == []
    assert verifier.calls == []


def test_empty_claim_text_is_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        AtomicClaim(
            claim_id="A",
            text="   ",
        )


def test_invalid_top_k_is_rejected():
    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        service(
            search=FakeAuthorizedSearch(),
            verifier=(
                FakeEntailmentVerifier(
                    labels_by_claim={}
                )
            ),
            top_k=0,
        )



def test_orchestration_audits_single_correlated_operation():
    from app.domain.enums.audit import (
        AuditEventType,
    )

    claim_text = "Supported claim"

    search = FakeAuthorizedSearch(
        results_by_query={
            claim_text: (
                chunk(
                    text="Supporting evidence.",
                    index=0,
                ),
            ),
        }
    )

    verifier = FakeEntailmentVerifier(
        labels_by_claim={
            claim_text: (
                EntailmentLabel.ENTAILMENT,
            ),
        }
    )

    audit = FakeAuditService()
    identity = current_identity()
    correlation_id = uuid4()

    result = service(
        search=search,
        verifier=verifier,
        audit=audit,
    ).evaluate(
        identity=identity,
        claims=(
            AtomicClaim(
                claim_id="A",
                text=claim_text,
            ),
        ),
        correlation_id=correlation_id,
    )

    assert result.correlation_id == correlation_id
    assert len(audit.calls) == 2

    submitted, decision = audit.calls

    assert (
        submitted.event_type
        == AuditEventType.QUERY_SUBMITTED
    )

    assert (
        decision.event_type
        == AuditEventType.EVIDENCE_SUFFICIENT
    )

    assert (
        submitted.correlation_id
        == correlation_id
    )
    assert (
        decision.correlation_id
        == correlation_id
    )

    assert (
        submitted.actor_user_id
        == identity.user_id
    )

    # Raw claim/query text must never be
    # copied into audit metadata.
    assert claim_text not in str(
        submitted.metadata
    )
    assert claim_text not in str(
        decision.metadata
    )

    assert decision.metadata[
        "retrieved_result_count"
    ] == 1


def test_insufficient_evidence_is_audited_once():
    from app.domain.enums.audit import (
        AuditEventType,
    )

    claim_text = "Unsupported claim"

    search = FakeAuthorizedSearch(
        results_by_query={
            claim_text: (
                chunk(
                    text="Unrelated evidence.",
                    index=0,
                ),
            ),
        }
    )

    verifier = FakeEntailmentVerifier(
        labels_by_claim={
            claim_text: (
                EntailmentLabel.NEUTRAL,
            ),
        }
    )

    audit = FakeAuditService()

    result = service(
        search=search,
        verifier=verifier,
        audit=audit,
    ).evaluate(
        identity=current_identity(),
        claims=(
            AtomicClaim(
                claim_id="A",
                text=claim_text,
            ),
        ),
    )

    assert not result.is_sufficient
    assert len(audit.calls) == 2

    assert (
        audit.calls[1].event_type
        == AuditEventType
        .EVIDENCE_INSUFFICIENT
    )

    assert (
        audit.calls[1].metadata[
            "unsupported_required_claim_count"
        ]
        == 1
    )


def test_initial_audit_failure_blocks_retrieval():
    from app.application.services.audit import (
        AuditPersistenceError,
    )

    search = FakeAuthorizedSearch()

    verifier = FakeEntailmentVerifier(
        labels_by_claim={}
    )

    with pytest.raises(
        AuditPersistenceError
    ):
        service(
            search=search,
            verifier=verifier,
            audit=FakeAuditService(
                fail_on_call=1
            ),
        ).evaluate(
            identity=current_identity(),
            claims=(
                AtomicClaim(
                    claim_id="A",
                    text="Claim",
                ),
            ),
        )

    assert search.calls == []
    assert verifier.calls == []


def test_final_audit_failure_blocks_result_return():
    from app.application.services.audit import (
        AuditPersistenceError,
    )

    claim_text = "Supported"

    search = FakeAuthorizedSearch(
        results_by_query={
            claim_text: (
                chunk(
                    text="Supported.",
                    index=0,
                ),
            ),
        }
    )

    verifier = FakeEntailmentVerifier(
        labels_by_claim={
            claim_text: (
                EntailmentLabel.ENTAILMENT,
            ),
        }
    )

    with pytest.raises(
        AuditPersistenceError
    ):
        service(
            search=search,
            verifier=verifier,
            audit=FakeAuditService(
                fail_on_call=2
            ),
        ).evaluate(
            identity=current_identity(),
            claims=(
                AtomicClaim(
                    claim_id="A",
                    text=claim_text,
                ),
            ),
        )

    assert len(search.calls) == 1
    assert len(verifier.calls) == 1
