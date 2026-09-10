from pathlib import Path
import sys

from app.infrastructure.parsing.docling_parser import (
    DoclingDocumentParser,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m scripts.smoke_docling_parser <pdf-path>"
        )

    pdf_path = Path(
        sys.argv[1]
    ).resolve()

    if not pdf_path.is_file():
        raise SystemExit(
            f"PDF not found: {pdf_path}"
        )

    parser = DoclingDocumentParser(
        max_chunk_chars=1500
    )

    parsed = parser.parse(
        pdf_path,
        language="fr",
    )

    print()
    print("=== GovKnowledge Docling Smoke Test ===")
    print("File:", pdf_path.name)
    print("Pages:", len(parsed.pages))
    print("Chunks:", len(parsed.chunks))
    print()

    page_numbers = [
        page.page_number
        for page in parsed.pages
    ]

    print(
        "Page numbers:",
        page_numbers,
    )

    assert page_numbers
    assert page_numbers[0] == 1
    assert page_numbers == list(
        range(
            1,
            len(page_numbers) + 1,
        )
    )

    for page in parsed.pages:
        page_chunks = [
            chunk
            for chunk in parsed.chunks
            if (
                chunk.page_number
                == page.page_number
            )
        ]

        preview = (
            page.text[:200]
            .replace("\n", " ")
        )

        print()
        print(
            f"--- Page {page.page_number} ---"
        )
        print(
            "Characters:",
            len(page.text),
        )
        print(
            "Chunks:",
            len(page_chunks),
        )
        print(
            "Preview:",
            preview,
        )

        for chunk in page_chunks:
            assert (
                chunk.page_number
                == page.page_number
            )

            assert (
                len(chunk.content)
                <= 1500
            )

            assert (
                chunk.metadata[
                    "page_number"
                ]
                == page.page_number
            )

    chunk_indexes = [
        chunk.chunk_index
        for chunk in parsed.chunks
    ]

    assert chunk_indexes == list(
        range(len(parsed.chunks))
    )

    print()
    print(
        "Real Docling page provenance: PASS"
    )


if __name__ == "__main__":
    main()
