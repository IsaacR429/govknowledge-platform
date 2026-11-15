from __future__ import annotations

import hashlib
from pathlib import Path


PDF_PATH = Path(
    r"C:\Users\isaac\Downloads\table example.pdf"
)

EXPECTED_SHA256 = (
    "9e7cb2a7512879b6c1d3c4c00ddb880e"
    "5393d7001cab4e53714daa27528dc432"
)


def test_conventional_fixture_identity():
    assert PDF_PATH.is_file(), (
        f"Conventional-table PDF not found: {PDF_PATH}"
    )

    pdf_bytes = PDF_PATH.read_bytes()

    assert pdf_bytes.startswith(b"%PDF-"), (
        "The fixture is not a PDF."
    )

    actual_sha256 = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    assert actual_sha256 == EXPECTED_SHA256, (
        "PDF differs from the reviewed conventional-table fixture."
    )

    print("CONVENTIONAL QA FIXTURE VERIFIED")


def test_conventional_table_parser_output(monkeypatch):
    from app.api.dependencies.document_processing import (
        build_routed_parser_executor,
    )

    # Confirm the exact PDF before invoking the parser.
    test_conventional_fixture_identity()

    # Do not enable the graphical-matrix experiment.
    monkeypatch.delenv(
        "GOVKNOWLEDGE_EXPERIMENTAL_VECTOR_TABLES",
        raising=False,
    )
    monkeypatch.delenv(
        "GOVKNOWLEDGE_EXPERIMENTAL_TABLE_EVIDENCE",
        raising=False,
    )

    executor = build_routed_parser_executor()
    result = executor.execute(PDF_PATH, language="en")

    print("CONVENTIONAL QA PARSER STATUS:", result.parsing_status.value)
    print("CONVENTIONAL QA PARSER ROUTING:",
          result.lineage.get("parser_routing"))
    print("CONVENTIONAL QA FAILURE CLASS:", result.failure_class)

    assert result.parsed_document is not None, (
        "GovKnowledge did not return a parsed document."
    )

    parsed = result.parsed_document
    print("CONVENTIONAL QA PAGES:", len(parsed.pages))
    print("CONVENTIONAL QA CHUNKS:", len(parsed.chunks))

    page_one = next(
        page for page in parsed.pages
        if page.page_number == 1
    )

    print("CONVENTIONAL QA PAGE 1 TEXT:")
    print(page_one.text[:2500])

    print("CONVENTIONAL QA CHUNKS CONTAINING T-004 OR T-005:")
    for chunk in parsed.chunks:
        if "T-004" in chunk.content or "T-005" in chunk.content:
            print(
                f"PAGE {chunk.page_number} "
                f"| CHUNK {chunk.chunk_index}"
            )
            print(chunk.content[:1500])

    from app.api.dependencies.qa import (
        get_qa_entailment_verifier,
    )

    t004_row = "|**T-004**|State Machine|Unit|"
    t005_row = "|**T-005**|State Machine|Negative|"

    # Verify the actual parser output; do not manufacture evidence.
    assert t004_row in page_one.text
    assert t005_row in page_one.text

    table_chunk = next(
        chunk
        for chunk in parsed.chunks
        if t004_row in chunk.content
        and t005_row in chunk.content
    )

    verifier = get_qa_entailment_verifier()

    cases = (
        (
            "T004_CORRECT",
            "The test type for T-004 is Unit.",
        ),
        (
            "T004_FALSE",
            "The test type for T-004 is Negative.",
        ),
    )

    premises = (
        ("T004_ROW", t004_row),
        ("T005_ROW", t005_row),
        ("FULL_TABLE_CHUNK", table_chunk.content),
    )

    for case_name, claim in cases:
        scores = verifier.score(
            claim=claim,
            premises=tuple(text for _, text in premises),
        )

        assert len(scores) == len(premises)

        print(f"\nCONVENTIONAL NLI CASE: {case_name}")
        print(f"CLAIM: {claim}")

        for (premise_name, _), score in zip(
            premises,
            scores,
            strict=True,
        ):
            print(
                f"{premise_name}: "
                f"label={score.label.value}; "
                f"entailment_probability="
                f"{score.entailment_probability:.4f}"
            )

    from uuid import uuid4

    from app.application.ports.retrieval import RetrievedChunk
    from app.application.services.atomic_evidence_policy import (
        AtomicClaimEvidence,
        AtomicEvidencePolicy,
    )
    from app.application.services.sentence_evidence_verification import (
        SentenceEvidenceVerificationService,
    )

    # Model the real parsed chunk in memory. These UUIDs are test
    # placeholders: no document is created in PostgreSQL.
    evidence_chunk = RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=table_chunk.page_number,
        chunk_index=table_chunk.chunk_index,
        content=table_chunk.content,
        section_title=table_chunk.section_title,
        language=table_chunk.language,
        score=1.0,
        page_end=table_chunk.page_number,
    )

    verification_service = SentenceEvidenceVerificationService(
        verifier=verifier,
        contextual_fallback=False,
    )
    policy = AtomicEvidencePolicy()

    assessments = {}

    for case_name, claim in cases:
        verification = verification_service.verify(
            claim=claim,
            evidence=(evidence_chunk,),
        )
        assessment = policy.assess(
            (
                AtomicClaimEvidence(
                    claim_id=case_name,
                    verification=verification,
                ),
            )
        )
        assessments[case_name] = assessment

        print(f"\nCONVENTIONAL POLICY CASE: {case_name}")
        print("DECISION:", assessment.decision.value)

        for span in verification.spans:
            print(
                "SPAN:",
                repr(span.text[:180]),
                "| LABEL:",
                span.verification.label.value,
                "| ENTAILMENT:",
                round(span.verification.entailment_probability, 4),
            )

    # Security/correctness regression: a neighboring row must not
    # establish a false fact about T-004.
    assert not assessments["T004_FALSE"].is_sufficient, (
        "FALSE SUPPORT: GovKnowledge accepted 'T-004 is Negative' "
        "from a table where T-004 is Unit."
    )

    assert assessments["T004_CORRECT"].is_sufficient, (
        "MISSED SUPPORT: GovKnowledge failed to support "
        "the correct fact 'T-004 is Unit'."
    )


def test_markdown_table_claim_uses_only_its_source_row():
    from types import SimpleNamespace
    from uuid import uuid4

    from app.application.ports.retrieval import RetrievedChunk
    from app.application.services.sentence_evidence_verification import (
        SentenceEvidenceVerificationService,
    )
    from app.domain.enums.evidence import EntailmentLabel

    # Exact Markdown format observed in the parser output.
    t004 = "|**T-004**|State Machine|Unit|"
    t005 = "|**T-005**|State Machine|Negative|"
    table = (
        "|**Test ID**|**Feature**|**Test Type**|\n"
        "|---|---|---|\n"
        f"{t004}\n"
        f"{t005}"
    )

    class RecordingVerifier:
        def __init__(self):
            self.calls = []

        def score(self, *, claim, premises):
            self.calls.append((claim, premises))
            return tuple(
                SimpleNamespace(
                    label=EntailmentLabel.NEUTRAL,
                    entailment_probability=0.0,
                )
                for _ in premises
            )

    chunk = RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=1,
        chunk_index=0,
        content=table,
        section_title=None,
        language="en",
        score=1.0,
        page_end=1,
    )

    verifier = RecordingVerifier()
    service = SentenceEvidenceVerificationService(
        verifier=verifier,
        contextual_fallback=False,
    )

    claim = "The test type for T-004 is Unit."
    result = service.verify(claim=claim, evidence=(chunk,))

    # NLI must receive the matching row, not the entire table.
    assert verifier.calls == [(claim, (t004,))]
    assert len(result.spans) == 1

    span = result.spans[0]
    assert span.text == t004

    # Citation offsets must refer to the unchanged source text.
    assert chunk.content[span.start_char:span.end_char] == t004

    # A nonexistent ID must not borrow another row's evidence.
    missing = service.verify(
        claim="The test type for T-999 is Unit.",
        evidence=(chunk,),
    )
    assert missing.spans == ()
    assert len(verifier.calls) == 1


def test_graphical_matrix_dll_keeps_table_identity():
    from types import SimpleNamespace
    from uuid import uuid4

    from app.application.ports.retrieval import RetrievedChunk
    from app.application.services.sentence_evidence_verification import (
        SentenceEvidenceVerificationService,
    )
    from app.domain.enums.evidence import EntailmentLabel

    left = (
        "PDF page 19, table 1. Organization: DLL. "
        "Marked activities: Creation, Diffusion."
    )
    right = (
        "PDF page 19, table 2. Organization: DLL. "
        "Marked activities: Diffusion, Network."
    )

    def make_chunk(content, index):
        return RetrievedChunk(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_version_id=uuid4(),
            page_id=uuid4(),
            page_number=19,
            chunk_index=index,
            content=content,
            section_title=None,
            language="en",
            score=1.0,
            page_end=19,
        )

    class RecordingVerifier:
        def __init__(self):
            self.calls = []

        def score(self, *, claim, premises):
            self.calls.append((claim, premises))
            return tuple(
                SimpleNamespace(
                    label=EntailmentLabel.NEUTRAL,
                    entailment_probability=0.0,
                )
                for _ in premises
            )

    left_chunk = make_chunk(left, 0)
    right_chunk = make_chunk(right, 1)
    verifier = RecordingVerifier()
    service = SentenceEvidenceVerificationService(
        verifier=verifier,
        contextual_fallback=False,
    )

    claim = "On PDF page 19, table 1, DLL has Network marked."
    result = service.verify(
        claim=claim,
        evidence=(left_chunk, right_chunk),
    )

    # The claim names table 1: table 2 cannot provide its evidence.
    assert verifier.calls == [(claim, (left,))]
    assert len(result.spans) == 1

    span = result.spans[0]
    assert span.chunk.chunk_id == left_chunk.chunk_id
    assert span.text == left
    assert (
        left_chunk.content[span.start_char:span.end_char]
        == left
    )


def test_p0_retrieval_excludes_vector_marker_evidence():
    from types import SimpleNamespace
    from uuid import uuid4

    from sqlalchemy.dialects import postgresql

    from app.domain.enums.permission import PermissionAction
    from app.infrastructure.database.models.embeddings import (
        EMBEDDING_DIMENSION,
    )
    from app.infrastructure.retrieval.postgres_vector_retrieval import (
        PostgresVectorRetrievalAdapter,
    )

    class RecordingSession:
        statement = None

        def execute(self, statement):
            self.statement = statement
            return SimpleNamespace(all=lambda: [])

    db = RecordingSession()
    retrieval = PostgresVectorRetrievalAdapter(db)

    result = retrieval.search_authorized(
        scope=SimpleNamespace(
            action=PermissionAction.READ,
            document_ids=frozenset({uuid4()}),
        ),
        query_vector=(1.0,) + (0.0,) * (EMBEDDING_DIMENSION - 1),
        model_name="test-model",
        model_revision="test-revision",
        top_k=5,
    )

    assert result == ()
    assert db.statement is not None

    import re

    compiled = db.statement.compile(dialect=postgresql.dialect())
    sql_text = str(compiled)
    where_sql = re.split(r"\s+WHERE\s+", sql_text, flags=re.IGNORECASE)[1].split(
        "ORDER BY", 1
    )[0]

    assert "->>" in where_sql
    assert "evidence_type" in compiled.params.values()
    assert "vector_marker_table" in compiled.params.values()
