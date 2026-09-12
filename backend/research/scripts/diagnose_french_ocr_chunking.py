from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

from docling.document_converter import DocumentConverter

from research.parsing.experimental_structure_chunker import (
    ExperimentalStructureChunker,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE = (
    REPO_ROOT
    / "documents"
    / "sample"
    / "world-bank-procurement-fr.pdf"
)


TESTS = {
    31: {
        "name": "FR03 electronic procurement",
        "terms": [
            "accessibilite",
            "securite",
            "integrite",
            "confidentialite",
        ],
    },
    40: {
        "name": "FR05 late offers",
        "terms": [
            "offres",
            "propositions",
            "recues",
            "hors",
            "delai",
            "rejetees",
        ],
    },
}


def normalize(value: str) -> str:
    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    value = value.replace(
        "\u00ad",
        "",
    )

    value = re.sub(
        r"(?<=\w)-\s*(?=\w)",
        "",
        value,
    )

    value = re.sub(
        r"[^a-zA-Z0-9]+",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip().lower()


def all_terms_present(
    text: str,
    terms: list[str],
) -> bool:
    value = normalize(text)

    return all(
        term in value
        for term in terms
    )


def hit_count(
    text: str,
    terms: list[str],
) -> int:
    value = normalize(text)

    return sum(
        term in value
        for term in terms
    )


def main() -> None:
    print()
    print(
        "=== Targeted French OCR/chunk diagnostic ==="
    )

    converter = DocumentConverter()

    for page_number, test in TESTS.items():
        print()
        print(
            "=" * 100
        )

        print(
            "Page:",
            page_number,
        )

        print(
            "Case:",
            test["name"],
        )

        started = time.perf_counter()

        result = converter.convert(
            SOURCE,
            page_range=(
                page_number,
                page_number,
            ),
            raises_on_error=True,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            "Parse time:",
            f"{elapsed:.1f}s",
        )

        markdown = (
            result.document
            .export_to_markdown()
        )

        document_has_all = (
            all_terms_present(
                markdown,
                test["terms"],
            )
        )

        print()
        print(
            "DOCLEVEL"
        )

        print(
            "All target terms present:",
            document_has_all,
        )

        print(
            "Target hits:",
            hit_count(
                markdown,
                test["terms"],
            ),
            "/",
            len(test["terms"]),
        )

        chunks = (
            ExperimentalStructureChunker(
                max_chars=2200,
            )
            .chunk(
                result.document
            )
        )

        print()
        print(
            "CHUNKLEVEL"
        )

        print(
            "Chunks:",
            len(chunks),
        )

        chunk_has_all = any(
            all_terms_present(
                chunk.content,
                test["terms"],
            )
            for chunk in chunks
        )

        print(
            "One chunk contains all terms:",
            chunk_has_all,
        )

        ranked = sorted(
            (
                (
                    hit_count(
                        chunk.content,
                        test["terms"],
                    ),
                    chunk,
                )
                for chunk in chunks
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        print()
        print(
            "Best matching chunks:"
        )

        for hits, chunk in ranked[:5]:
            snippet = " ".join(
                chunk.content.split()
            )[:900]

            print()
            print(
                f"hits={hits}/{len(test['terms'])} "
                f"page={chunk.page_number} "
                f"structure={chunk.structural_label}"
            )

            print(
                snippet
            )

        print()
        print(
            "RAW DOCLING MARKDOWN:"
        )

        print(
            markdown[:5000]
        )

    print()
    print(
        "=" * 100
    )

    print(
        "Interpretation:"
    )

    print(
        "DOCLEVEL false => OCR/parsing loss."
    )

    print(
        "DOCLEVEL true + CHUNKLEVEL false "
        "=> chunker loss."
    )

    print(
        "Both true => retrieval/benchmark issue."
    )

    print()
    print(
        "No embeddings were generated."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
