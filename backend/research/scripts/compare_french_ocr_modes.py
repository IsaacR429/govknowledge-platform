from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

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

PAGE = 40

TARGET_TERMS = [
    "offres",
    "propositions",
    "recues",
    "hors",
    "delai",
    "rejetees",
]

TARGET_PHRASES = [
    "offres propositions recues hors delai sont rejetees",
    "recues hors delai sont rejetees",
]


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

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip().lower()


def term_hits(text: str) -> int:
    normalized = normalize(text)

    return sum(
        term in normalized
        for term in TARGET_TERMS
    )


def phrase_found(text: str) -> bool:
    normalized = normalize(text)

    return any(
        phrase in normalized
        for phrase in TARGET_PHRASES
    )


def parse_default():
    converter = DocumentConverter()

    return converter.convert(
        SOURCE,
        page_range=(PAGE, PAGE),
        raises_on_error=True,
    )


def parse_full_page():
    pipeline_options = (
        PdfPipelineOptions()
    )

    pipeline_options.do_ocr = True

    pipeline_options.ocr_options = (
        RapidOcrOptions(
            backend="torch",
            force_full_page_ocr=True,
        )
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: (
                PdfFormatOption(
                    pipeline_options=(
                        pipeline_options
                    ),
                )
            )
        }
    )

    return converter.convert(
        SOURCE,
        page_range=(PAGE, PAGE),
        raises_on_error=True,
    )


def evaluate(
    *,
    name: str,
    result,
    elapsed: float,
):
    print()
    print(
        "=" * 105
    )

    print(
        name
    )

    print(
        "Parse time:",
        f"{elapsed:.1f}s",
    )

    markdown = (
        result.document
        .export_to_markdown()
    )

    print()
    print(
        "DOCLEVEL"
    )

    print(
        "Target terms:",
        f"{term_hits(markdown)}/"
        f"{len(TARGET_TERMS)}",
    )

    print(
        "Exact late-offer rule recovered:",
        phrase_found(markdown),
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

    best = sorted(
        (
            (
                term_hits(
                    chunk.content
                ),
                phrase_found(
                    chunk.content
                ),
                chunk,
            )
            for chunk in chunks
        ),
        key=lambda row: (
            row[1],
            row[0],
        ),
        reverse=True,
    )

    print(
        "Rule present in a chunk:",
        any(
            row[1]
            for row in best
        ),
    )

    print()
    print(
        "Best chunks:"
    )

    for hits, exact, chunk in best[:5]:
        print()
        print(
            f"hits={hits}/"
            f"{len(TARGET_TERMS)} "
            f"exact={exact} "
            f"structure="
            f"{chunk.structural_label}"
        )

        print(
            " ".join(
                chunk.content.split()
            )[:1200]
        )

    print()
    print(
        "MARKDOWN"
    )

    print(
        markdown[:7000]
    )


def main() -> None:
    print()
    print(
        "=== Page 40 OCR mode A/B ==="
    )

    print(
        "Control: current/default OCR"
    )

    started = time.perf_counter()

    default_result = (
        parse_default()
    )

    default_elapsed = (
        time.perf_counter()
        - started
    )

    evaluate(
        name="DEFAULT OCR",
        result=default_result,
        elapsed=default_elapsed,
    )

    print()
    print(
        "Experiment: RapidOCR FULL_PAGE"
    )

    started = time.perf_counter()

    full_result = (
        parse_full_page()
    )

    full_elapsed = (
        time.perf_counter()
        - started
    )

    evaluate(
        name="FULL_PAGE OCR",
        result=full_result,
        elapsed=full_elapsed,
    )

    print()
    print(
        "=" * 105
    )

    default_markdown = (
        default_result.document
        .export_to_markdown()
    )

    full_markdown = (
        full_result.document
        .export_to_markdown()
    )

    print(
        "COMPARISON"
    )

    print(
        "Default terms:",
        f"{term_hits(default_markdown)}/6",
    )

    print(
        "Full-page terms:",
        f"{term_hits(full_markdown)}/6",
    )

    print(
        "Default exact rule:",
        phrase_found(
            default_markdown
        ),
    )

    print(
        "Full-page exact rule:",
        phrase_found(
            full_markdown
        ),
    )

    print()
    print(
        "No embedding model was loaded."
    )

    print(
        "No OCR language was changed."
    )

    print(
        "No OCR engine was changed."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
