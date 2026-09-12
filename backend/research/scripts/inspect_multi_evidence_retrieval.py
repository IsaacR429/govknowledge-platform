from __future__ import annotations

import json
import re
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
    / "test-policy.pdf"
)

DATASET = Path(__file__).resolve().parents[1] / "fixtures" / "evidence_support_benchmark_v2.json"

GROUND_TRUTH = {
    "M01": {
        "admission": (
            "admission of any such state to membership "
            "in the united nations"
        ),
    },
    "M02": {
        "secretary_general": (
            "shall be appointed by the general assembly "
            "upon the recommendation of the security council"
        ),
    },
    "M03": {
        "admission": (
            "admission of any such state to membership "
            "in the united nations"
        ),
        "expulsion": (
            "may be expelled from the organization "
            "by the general assembly upon the recommendation "
            "of the security council"
        ),
    },
    "M04": {
        "security_council": (
            "primary responsibility for the maintenance "
            "of international peace and security"
        ),
        "self_defence": (
            "inherent right of individual or collective"
        ),
    },
}


def normalize(value: str) -> str:
    value = unicodedata.normalize(
        "NFKC",
        value,
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
        r"[^\w]+",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip().lower()


def contains(
    content: str,
    anchor: str,
) -> bool:
    return (
        normalize(anchor)
        in normalize(content)
    )


def main() -> None:
    cases = {
        item["id"]: item
        for item in json.loads(
            DATASET.read_text(
                encoding="utf-8"
            )
        )
    }

    print()
    print(
        "=== Multi-evidence retrieval coverage ==="
    )

    parsed = (
        DocumentConverter()
        .convert(
            SOURCE,
            raises_on_error=True,
        )
    )

    chunks = (
        ExperimentalStructureChunker(
            max_chars=2200,
        )
        .chunk(
            parsed.document
        )
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

    for case_id in (
        "M01",
        "M02",
        "M03",
        "M04",
    ):
        question = cases[
            case_id
        ]["question"]

        query_vector = model.encode(
            [
                "query: " + question
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

        print()
        print(
            "=" * 88
        )

        print(
            case_id,
            question,
        )

        anchors = GROUND_TRUTH[
            case_id
        ]

        for name, anchor in anchors.items():
            rank = None
            page = None

            for position, index in enumerate(
                order,
                start=1,
            ):
                chunk = chunks[
                    int(index)
                ]

                if contains(
                    chunk.content,
                    anchor,
                ):
                    rank = position
                    page = (
                        chunk.page_number
                    )
                    break

            print(
                f"required={name:<20} "
                f"rank={rank} "
                f"page={page}"
            )

        print()
        print(
            "Top 5 retrieved chunks:"
        )

        for rank, index in enumerate(
            order[:5],
            start=1,
        ):
            chunk = chunks[
                int(index)
            ]

            snippet = " ".join(
                chunk.content.split()
            )[:400]

            print()
            print(
                f"#{rank} "
                f"score={scores[int(index)]:.4f} "
                f"page={chunk.page_number} "
                f"article={chunk.article}"
            )

            print(
                snippet
            )


if __name__ == "__main__":
    main()
