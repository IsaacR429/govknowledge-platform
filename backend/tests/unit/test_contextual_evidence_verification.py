from __future__ import annotations

from uuid import uuid4

import pytest

from app.application.ports.citation_source_repository import CitationSource
from app.application.ports.entailment_verifier import EntailmentScore
from app.application.ports.retrieval import RetrievedChunk
from app.application.services.citation_mapping import EvidenceSpanCitationMapper
from app.application.services.citation_validation import CitationValidator
from app.application.services.sentence_evidence_verification import (
    SentenceEvidenceVerificationService,
)
from app.domain.enums.evidence import EntailmentLabel


R01 = (
    "## **Article 7**\n\n1. "
    "The Security Council is one of the principal organs "
    "of the United Nations."
)
R03 = (
    "## Five principles of partnership\n\n"
    "### **1. Complementary purpose and added value**\n\n"
    "**Any partnership between Oxfam GB and other organisations "
    "is based on shared objectives."
)
R05 = (
    "STATUTE OF THE INTERNATIONAL COURT OF JUSTICE\n\n"
    "Article 1\n\n"
    "The Court is established under the Charter.\n\n"
    "Article 2\n\n"
    "The Court is composed of independent judges.\n\n"
    "Article 3\n\n1. "
    "The Court shall consist of fifteen members, "
    "no two of whom may be nationals of the same state."
)


class ExactPremiseVerifier:
    model_name = "fake-nli"
    model_revision = "test"

    def __init__(self, entailing_premise: str):
        self.entailing_premise = entailing_premise
        self.calls = []

    def score(self, *, claim, premises):
        self.calls.append((claim, tuple(premises)))
        result = []
        for premise in premises:
            supports = premise == self.entailing_premise
            result.append(
                EntailmentScore(
                    label=(
                        EntailmentLabel.ENTAILMENT
                        if supports else EntailmentLabel.NEUTRAL
                    ),
                    entailment_probability=0.99 if supports else 0.01,
                    neutral_probability=0.009 if supports else 0.98,
                    contradiction_probability=0.001 if supports else 0.01,
                )
            )
        return tuple(result)


def chunk(
    content: str, *, page_end: int | None = 7
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        page_id=uuid4(),
        page_number=7,
        page_end=page_end,
        chunk_index=2,
        content=content,
        section_title=None,
        language="en",
        score=0.84,
    )


class SourceRepository:
    def __init__(self, source):
        self.source = source

    def get_by_chunk_id(self, chunk_id):
        if chunk_id == self.source.chunk_id:
            return self.source
        return None


@pytest.mark.parametrize(
    "case_id, body, prefix, suffix",
    (
        ("R01", R01, "Previous provision.\n\n", "\n\nArticle 8\n\nOther text."),
        ("R03", R03, "", "\n\nFurther partnership guidance."),
        ("R05", R05, "", "\n\nArticle 4\n\nFurther provisions."),
    ),
)
def test_contextual_support_retains_one_exact_citable_slice(
    case_id, body, prefix, suffix
):
    evidence = chunk(prefix + body + suffix)
    verifier = ExactPremiseVerifier(body)

    result = SentenceEvidenceVerificationService(
        verifier=verifier,
        contextual_fallback=True,
    ).verify(
        claim=f"Context-dependent claim {case_id}",
        evidence=(evidence,),
    )

    assert result.is_supported
    assert len(result.supporting_spans) == 1

    span = result.supporting_spans[0]
    assert span.chunk.chunk_id == evidence.chunk_id
    assert span.chunk.document_version_id == evidence.document_version_id
    assert span.chunk.page_id == evidence.page_id
    assert span.start_char == len(prefix)
    assert span.end_char == len(prefix) + len(body)
    assert span.text == body
    assert evidence.content[span.start_char:span.end_char] == span.text

    citation = EvidenceSpanCitationMapper().map(span)
    source = CitationSource(
        chunk_id=evidence.chunk_id,
        document_id=evidence.document_id,
        document_version_id=evidence.document_version_id,
        page_id=evidence.page_id,
        page_number=evidence.page_number,
        content=evidence.content,
    )
    assert CitationValidator(
        repository=SourceRepository(source)
    ).validate(citation) == citation


def test_context_cannot_cross_chunk_boundaries():
    heading = chunk(
        "STATUTE OF THE INTERNATIONAL COURT OF JUSTICE\n\n"
        "Article 3\n\n1."
    )
    fact = chunk("The Court shall consist of fifteen members.")
    combined = heading.content + "\n" + fact.content
    verifier = ExactPremiseVerifier(combined)

    result = SentenceEvidenceVerificationService(
        verifier=verifier,
        contextual_fallback=True,
    ).verify(claim="The Court has fifteen members.", evidence=(heading, fact))

    assert not result.is_supported
    assert all(
        combined not in premises
        for _, premises in verifier.calls
    )


def test_context_cannot_exceed_1000_characters():
    content = (
        "STATUTE OF THE INTERNATIONAL COURT OF JUSTICE\n\n"
        "Article 1\n\n"
        + ("An unrelated sentence. " * 60)
        + "\n\nArticle 3\n\n1. "
        + "The Court shall consist of fifteen members."
    )
    verifier = ExactPremiseVerifier(content)

    result = SentenceEvidenceVerificationService(
        verifier=verifier,
        contextual_fallback=True,
    ).verify(claim="Article 3 concerns the Court.", evidence=(chunk(content),))

    assert not result.is_supported
    assert all(
        len(premise) <= 1000
        for _, premises in verifier.calls[1:]
        for premise in premises
    )


def test_unheaded_sentences_are_not_arbitrarily_merged():
    content = "First statement. Second statement."
    verifier = ExactPremiseVerifier(content)

    result = SentenceEvidenceVerificationService(
        verifier=verifier,
        contextual_fallback=True,
    ).verify(claim="Combined claim", evidence=(chunk(content),))

    assert not result.is_supported
    assert all(
        content not in premises
        for _, premises in verifier.calls
    )


def test_existing_sentence_support_does_not_trigger_context_fallback():
    fact = "A sufficient standalone sentence."
    verifier = ExactPremiseVerifier(fact)

    result = SentenceEvidenceVerificationService(
        verifier=verifier,
        contextual_fallback=True,
    ).verify(
        claim="A sufficient claim",
        evidence=(chunk("Article 3\n\n1. " + fact),),
    )

    assert result.is_supported
    assert len(verifier.calls) == 1
    assert result.supporting_spans[0].text == fact




@pytest.mark.parametrize("page_end", (None, 8))
def test_context_refuses_missing_or_multipage_provenance(page_end):
    verifier = ExactPremiseVerifier(R01)
    evidence = chunk(R01, page_end=page_end)

    result = SentenceEvidenceVerificationService(
        verifier=verifier,
        contextual_fallback=True,
    ).verify(
        claim="The Security Council is a principal organ under Article 7.",
        evidence=(evidence,),
    )

    assert not result.is_supported
    assert len(verifier.calls) == 1
    assert all(
        R01 not in premises
        for _, premises in verifier.calls
    )


def test_context_fallback_is_disabled_by_default():
    verifier = ExactPremiseVerifier(R05)

    result = SentenceEvidenceVerificationService(
        verifier=verifier,
    ).verify(claim="The Court has fifteen members.", evidence=(chunk(R05),))

    assert not result.is_supported
    assert len(verifier.calls) == 1


@pytest.mark.parametrize(
    "prose",
    (
        "Article 3 applies to parking.",
        "Article 3 is mentioned in this paragraph.",
    ),
)
def test_article_reference_in_prose_is_not_a_heading(prose):
    content = prose + " The Court shall consist of fifteen members."
    verifier = ExactPremiseVerifier(content)

    result = SentenceEvidenceVerificationService(
        verifier=verifier,
        contextual_fallback=True,
    ).verify(
        claim="The Court consists of fifteen members under Article 3.",
        evidence=(chunk(content),),
    )

    assert not result.is_supported
    assert all(
        content not in premises
        for _, premises in verifier.calls
    )
