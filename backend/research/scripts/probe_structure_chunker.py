from __future__ import annotations

import statistics
from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
)

from research.parsing.experimental_structure_chunker import (
    ExperimentalStructureChunker,
)


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "documents"
    / "sample"
    / "test-policy.pdf"
)

TARGET_ARTICLES = {
    "Article 6",
    "Article 24",
    "Article 97",
}


def preview(
    text: str,
    limit: int = 650,
) -> str:
    value = " ".join(
        text.split()
    )

    if len(value) <= limit:
        return value

    return value[:limit] + "..."


def main() -> None:
    print()
    print(
        "=== Experimental structural chunking ==="
    )

    print(
        "Source:",
        SOURCE,
    )

    result = (
        DocumentConverter()
        .convert(
            SOURCE,
            raises_on_error=True,
        )
    )

    chunker = (
        ExperimentalStructureChunker(
            max_chars=2200,
        )
    )

    chunks = chunker.chunk(
        result.document
    )

    lengths = [
        len(chunk.content)
        for chunk in chunks
    ]

    with_article = [
        chunk
        for chunk in chunks
        if chunk.article is not None
    ]

    print()
    print(
        "Experimental chunks:",
        len(chunks),
    )

    print(
        "Article-labelled chunks:",
        len(with_article),
    )

    print(
        "Min chars:",
        min(lengths),
    )

    print(
        "Median chars:",
        int(
            statistics.median(
                lengths
            )
        ),
    )

    print(
        "Max chars:",
        max(lengths),
    )

    print()
    print(
        "=== Target evidence units ==="
    )

    for target in TARGET_ARTICLES:
        matches = [
            chunk
            for chunk in chunks
            if (
                chunk.article
                and chunk.article.lower()
                == target.lower()
            )
        ]

        print()
        print(
            target,
            "chunks:",
            len(matches),
        )

        for item in matches:
            print(
                f"page={item.page_number} "
                f"chars={len(item.content)}"
            )

            print(
                preview(
                    item.content
                )
            )

            print()

    print(
        "=== Safety checks ==="
    )

    print(
        "All chunks page-local:",
        all(
            chunk.page_number >= 1
            for chunk in chunks
        ),
    )

    print(
        "No empty chunks:",
        all(
            bool(
                chunk.content.strip()
            )
            for chunk in chunks
        ),
    )


if __name__ == "__main__":
    main()
