from pathlib import Path
from uuid import UUID, uuid5, NAMESPACE_URL

from app.application.ports.retrieval import RetrievedChunk
from app.application.services.sentence_evidence_verification import (
    SentenceEvidenceVerificationService,
)
from app.infrastructure.parsing.docling_parser import (
    DoclingDocumentParser,
)
from app.infrastructure.verification.mdeberta_entailment_verifier import (
    MDebertaEntailmentVerifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PDF_PATH = (
    PROJECT_ROOT
    / "documents"
    / "qualification"
    / "unicef-emergency-procedures-fr.pdf"
)

DOCUMENT_ID = UUID(
    "ba4f8e8c-f165-4940-9148-611fb488b653"
)

VERSION_ID = UUID(
    "7326d7cb-2fad-400c-84aa-526073914849"
)

PAGE_ID = uuid5(
    NAMESPACE_URL,
    "govknowledge-unicef-page-12",
)

CLAIM = (
    "750 000 dollars de cette allocation "
    "sont affectés à la PEAS."
)


def main() -> None:
    parser = DoclingDocumentParser()

    parsed = parser.parse(
        PDF_PATH,
        language="fr",
    )

    page12 = tuple(
        chunk
        for chunk in parsed.chunks
        if chunk.page_number == 12
    )

    interesting = tuple(
        chunk
        for chunk in page12
        if (
            "750 000" in chunk.content
            or "500 000" in chunk.content
        )
    )

    print("=" * 72)
    print("PAGE 12 — FIXED CHUNK OUTPUT")
    print("=" * 72)

    print(
        "Relevant chunks:",
        len(interesting),
    )

    evidence = []

    for chunk in interesting:
        print()
        print("-" * 72)
        print(
            f"chunk_index={chunk.chunk_index} "
            f"length={len(chunk.content)}"
        )
        print("-" * 72)

        print(chunk.content)

        evidence.append(
            RetrievedChunk(
                chunk_id=uuid5(
                    NAMESPACE_URL,
                    (
                        "govknowledge-unicef-page12-"
                        f"chunk-{chunk.chunk_index}"
                    ),
                ),
                document_id=DOCUMENT_ID,
                document_version_id=VERSION_ID,
                page_id=PAGE_ID,
                page_number=12,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                section_title=chunk.section_title,
                language=chunk.language,
                score=1.0,
            )
        )

    print()
    print("=" * 72)
    print("CLAIM")
    print("=" * 72)
    print(CLAIM)

    verifier = MDebertaEntailmentVerifier(
        local_files_only=True,
    )

    service = SentenceEvidenceVerificationService(
        verifier=verifier,
    )

    result = service.verify(
        claim=CLAIM,
        evidence=tuple(evidence),
    )

    print()
    print("=" * 72)
    print("SENTENCE-LEVEL VERIFICATION")
    print("=" * 72)

    for span in result.spans:
        verification = span.verification

        print()
        print("-" * 72)
        print(
            "chunk_index:",
            span.chunk.chunk_index,
        )
        print(
            "sentence_index:",
            span.sentence_index,
        )
        print(
            "text:",
            repr(span.text),
        )
        print(
            "label:",
            verification.label,
        )
        print(
            "entailment:",
            verification.entailment_probability,
        )
        print(
            "neutral:",
            verification.neutral_probability,
        )
        print(
            "contradiction:",
            verification.contradiction_probability,
        )

    print()
    print("=" * 72)
    print("DIAGNOSTIC")
    print("=" * 72)

    entailed = [
        span
        for span in result.spans
        if (
            span.verification.label.value
            == "entailment"
        )
    ]

    if entailed:
        best = max(
            entailed,
            key=lambda span: (
                span.verification
                .entailment_probability
            ),
        )

        print("RESULT: ENTAILMENT FOUND")
        print(
            "Best probability:",
            best.verification
            .entailment_probability,
        )
        print(
            "Best sentence:",
            best.text,
        )
    else:
        print("RESULT: NO ENTAILMENT")
        print(
            "The chunk-boundary corruption is fixed, "
            "but sentence-level verification still "
            "cannot connect the level-3 amount to PEAS."
        )


if __name__ == "__main__":
    main()
