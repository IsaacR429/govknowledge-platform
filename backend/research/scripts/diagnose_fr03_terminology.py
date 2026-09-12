from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import numpy as np
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

from research.parsing.experimental_structure_chunker import (
    ExperimentalStructureChunker,
)


MODEL_NAME = "intfloat/multilingual-e5-base"

MODEL_REVISION = (
    "d128750597153bb5987e10b1c3493a34e5a4502a"
)

REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE = (
    REPO_ROOT
    / "documents"
    / "sample"
    / "world-bank-procurement-fr.pdf"
)

# Ground-truth page + pages that previously outranked it.
PAGES = (
    31,
    50,
    30,
    35,
    96,
    32,
)

QUERIES = {
    "original": (
        "Quelles conditions doivent respecter les systemes "
        "electroniques utilises pour la passation des marches ?"
    ),
    "source_term": (
        "Quelles conditions doit respecter un systeme "
        "de dematerialisation de la passation des marches ?"
    ),
    "combined": (
        "Quelles conditions doivent respecter les systemes "
        "electroniques ou systemes de dematerialisation "
        "utilises pour la passation des marches ?"
    ),
    "oracle_conditions": (
        "Quelles exigences d'accessibilite, de securite, "
        "d'integrite et de confidentialite doit respecter "
        "un systeme de dematerialisation de la passation "
        "des marches ?"
    ),
}

TARGET_TERMS = (
    "accessibilite",
    "securite",
    "integrite",
    "confidentialite",
)


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


def is_ground_truth(content: str) -> bool:
    value = normalize(content)

    return all(
        term in value
        for term in TARGET_TERMS
    )


def main() -> None:
    print()
    print(
        "=== FR03 terminology alignment diagnostic ==="
    )

    converter = DocumentConverter()

    chunks = []

    for page in PAGES:
        print(
            "Parsing page",
            page,
            "..."
        )

        started = time.perf_counter()

        result = converter.convert(
            SOURCE,
            page_range=(
                page,
                page,
            ),
            raises_on_error=True,
        )

        page_chunks = (
            ExperimentalStructureChunker(
                max_chars=2200,
            )
            .chunk(
                result.document
            )
        )

        chunks.extend(
            page_chunks
        )

        print(
            f"  chunks={len(page_chunks)} "
            f"time={time.perf_counter() - started:.1f}s"
        )

    print()
    print(
        "Candidate chunks:",
        len(chunks),
    )

    ground_truth_indices = [
        index
        for index, chunk in enumerate(
            chunks
        )
        if is_ground_truth(
            chunk.content
        )
    ]

    print(
        "Ground-truth chunks:",
        len(
            ground_truth_indices
        ),
    )

    if not ground_truth_indices:
        raise RuntimeError(
            "Ground-truth FR03 chunk "
            "was not recovered."
        )

    print()
    print(
        "Loading pinned E5..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    vectors = model.encode(
        [
            "passage: " + chunk.content
            for chunk in chunks
        ],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    for name, query in (
        QUERIES.items()
    ):
        query_vector = model.encode(
            [
                "query: " + query
            ],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        scores = (
            vectors
            @ query_vector
        )

        order = np.argsort(
            -scores
        )

        ground_truth_rank = next(
            rank
            for rank, index in enumerate(
                order,
                start=1,
            )
            if int(index)
            in ground_truth_indices
        )

        best_gt_index = max(
            ground_truth_indices,
            key=lambda index: (
                scores[index]
            ),
        )

        print()
        print(
            "=" * 100
        )

        print(
            name.upper()
        )

        print(
            "Query:",
            query,
        )

        print(
            "Ground-truth rank:",
            ground_truth_rank,
        )

        print(
            "Ground-truth score:",
            f"{scores[best_gt_index]:.4f}",
        )

        print()
        print(
            "Top 5:"
        )

        for rank, index in enumerate(
            order[:5],
            start=1,
        ):
            index = int(index)

            chunk = chunks[index]

            marker = (
                " <-- GROUND TRUTH"
                if index
                in ground_truth_indices
                else ""
            )

            print()
            print(
                f"#{rank} "
                f"score={scores[index]:.4f} "
                f"page={chunk.page_number} "
                f"structure="
                f"{chunk.structural_label}"
                f"{marker}"
            )

            print(
                " ".join(
                    chunk.content.split()
                )[:600]
            )

    print()
    print(
        "=" * 100
    )

    print(
        "Interpretation:"
    )

    print(
        "original poor + source_term strong "
        "=> terminology mismatch."
    )

    print(
        "combined strong "
        "=> query expansion is viable."
    )

    print(
        "only oracle_conditions strong "
        "=> deeper semantic reformulation "
        "may be required."
    )

    print()
    print(
        "No database rows were modified."
    )

    print(
        "No retrieval model was changed."
    )

    print(
        "No production query rewriter "
        "was introduced."
    )


if __name__ == "__main__":
    main()
