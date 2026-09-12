from __future__ import annotations

import json
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

from app.application.ports.retrieval import RetrievedChunk
from research.parsing.experimental_structure_chunker import (
    ExperimentalStructureChunker,
)
from research.retrieval.experimental_bge_evidence_support import (
    ExperimentalBgeEvidenceSupportVerifier,
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

DATASET = Path(__file__).resolve().parents[1] / "fixtures" / "atomic_entailment_benchmark_v1.json"

TOP_K = 3

THRESHOLDS = (
    0.025,
    0.050,
    0.100,
    0.200,
    0.300,
    0.400,
    0.500,
)


SUPPORTED_ANCHORS = {
    "E01": (
        "primary responsibility for the maintenance "
        "of international peace and security"
    ),
    "E04": (
        "shall be appointed by the general assembly "
        "upon the recommendation of the security council"
    ),
    "E07": (
        "may be expelled from the organization "
        "by the general assembly upon the recommendation "
        "of the security council"
    ),
    "E09": (
        "principal judicial organ of the united nations"
    ),
    "E11": (
        "decisions of the general assembly on important "
        "questions shall be made by a two-thirds majority"
    ),
    "E13": (
        "inherent right of individual or collective"
    ),
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


def make_chunk(
    *,
    index: int,
    chunk,
    score: float,
) -> RetrievedChunk:
    base = f"entailment-{index}"

    return RetrievedChunk(
        chunk_id=uuid5(
            NAMESPACE_URL,
            base + "-chunk",
        ),
        document_id=uuid5(
            NAMESPACE_URL,
            "entailment-document",
        ),
        document_version_id=uuid5(
            NAMESPACE_URL,
            "entailment-version",
        ),
        page_id=uuid5(
            NAMESPACE_URL,
            base + "-page",
        ),
        page_number=chunk.page_number,
        chunk_index=index,
        content=chunk.content,
        section_title=(
            chunk.structural_label or None
        ),
        language="en",
        score=float(score),
    )


def confusion(
    rows,
    threshold: float,
):
    tp = fp = tn = fn = 0
    errors = []

    for row in rows:
        actual_supported = (
            row["label"]
            == "SUPPORTED"
        )

        predicted_supported = (
            row["max_support"]
            >= threshold
        )

        if (
            actual_supported
            and predicted_supported
        ):
            tp += 1

        elif (
            not actual_supported
            and predicted_supported
        ):
            fp += 1
            errors.append(
                row["id"]
                + ":"
                + row["label"]
            )

        elif (
            not actual_supported
            and not predicted_supported
        ):
            tn += 1

        else:
            fn += 1
            errors.append(
                row["id"]
                + ":SUPPORTED"
            )

    return (
        tp,
        fp,
        tn,
        fn,
        errors,
    )


def describe(values) -> str:
    return (
        f"min={min(values):.4f} "
        f"med={statistics.median(values):.4f} "
        f"max={max(values):.4f}"
    )


def main() -> None:
    cases = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    print()
    print(
        "=== Atomic entailment stress benchmark ==="
    )

    print(
        "Cases:",
        len(cases),
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

    print(
        "Structure chunks:",
        len(chunks),
    )

    print(
        "Loading pinned E5..."
    )

    embedder = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    print(
        "Embedding corpus..."
    )

    vectors = embedder.encode(
        [
            "passage: " + chunk.content
            for chunk in chunks
        ],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print(
        "Loading BGE verifier..."
    )

    verifier = (
        ExperimentalBgeEvidenceSupportVerifier(
            batch_size=3,
            max_length=1024,
        )
    )

    print(
        "Device:",
        verifier.device,
    )

    rows = []
    latencies = []

    print()
    print(
        "=" * 105
    )
    print(
        "PER-CLAIM RESULTS"
    )

    for case in cases:
        claim = case["claim"]

        query_vector = embedder.encode(
            [
                "query: " + claim
            ],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        similarities = (
            vectors
            @ query_vector
        )

        order = np.argsort(
            -similarities
        )[:TOP_K]

        evidence = tuple(
            make_chunk(
                index=int(index),
                chunk=chunks[
                    int(index)
                ],
                score=float(
                    similarities[
                        int(index)
                    ]
                ),
            )
            for index in order
        )

        started = (
            time.perf_counter()
        )

        support = verifier.score(
            query=claim,
            evidence=evidence,
        )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        latencies.append(
            elapsed_ms
        )

        scores = [
            item.support_score
            for item in support
        ]

        max_support = max(
            scores,
            default=0.0,
        )

        best_position = int(
            np.argmax(scores)
        )

        best_chunk = evidence[
            best_position
        ]

        anchor_rank = None

        if case["id"] in SUPPORTED_ANCHORS:
            anchor = SUPPORTED_ANCHORS[
                case["id"]
            ]

            for rank, index in enumerate(
                order,
                start=1,
            ):
                if contains(
                    chunks[
                        int(index)
                    ].content,
                    anchor,
                ):
                    anchor_rank = rank
                    break

        rows.append(
            {
                "id": case["id"],
                "label": case["label"],
                "max_support": max_support,
                "anchor_rank": anchor_rank,
            }
        )

        print()
        print(
            f"{case['id']} "
            f"{case['label']}"
        )

        print(
            "  claim:",
            claim,
        )

        print(
            "  E5 top3:",
            ", ".join(
                f"{similarities[int(index)]:.4f}"
                for index in order
            ),
        )

        print(
            "  BGE top3:",
            ", ".join(
                f"{score:.4f}"
                for score in scores
            ),
        )

        print(
            "  max support:",
            f"{max_support:.4f}",
        )

        if (
            case["label"]
            == "SUPPORTED"
        ):
            print(
                "  expected evidence rank:",
                anchor_rank,
            )

        snippet = " ".join(
            best_chunk.content.split()
        )[:300]

        print(
            "  highest-scoring evidence:"
        )

        print(
            "   ",
            snippet,
        )

        print(
            "  latency:",
            f"{elapsed_ms:.0f} ms",
        )

    print()
    print(
        "=" * 105
    )

    print(
        "CLASS DISTRIBUTIONS"
    )

    grouped = defaultdict(
        list
    )

    for row in rows:
        grouped[
            row["label"]
        ].append(
            row[
                "max_support"
            ]
        )

    for label in (
        "SUPPORTED",
        "CONTRADICTED",
        "OVERCLAIM",
        "UNSUPPORTED",
    ):
        values = grouped[
            label
        ]

        print(
            f"{label:<14}",
            describe(
                values
            ),
        )

    supported = grouped[
        "SUPPORTED"
    ]

    negatives = [
        row["max_support"]
        for row in rows
        if row["label"]
        != "SUPPORTED"
    ]

    print()
    print(
        "Supported minimum:",
        f"{min(supported):.4f}",
    )

    print(
        "Non-supported maximum:",
        f"{max(negatives):.4f}",
    )

    margin = (
        min(supported)
        - max(negatives)
    )

    print(
        "Observed margin:",
        f"{margin:.4f}",
    )

    print()
    print(
        "=" * 105
    )

    print(
        "SUPPORTED VS EVERYTHING-ELSE"
    )

    print(
        "threshold   TP   FP   TN   FN   errors"
    )

    for threshold in THRESHOLDS:
        (
            tp,
            fp,
            tn,
            fn,
            errors,
        ) = confusion(
            rows,
            threshold,
        )

        print(
            f"{threshold:>8.3f}"
            f"{tp:>5}"
            f"{fp:>5}"
            f"{tn:>5}"
            f"{fn:>5}   "
            f"{','.join(errors) if errors else '-'}"
        )

    print()
    print(
        "=" * 105
    )

    print(
        "HIGH-SCORING NON-SUPPORTED CLAIMS"
    )

    negatives_sorted = sorted(
        (
            row
            for row in rows
            if row["label"]
            != "SUPPORTED"
        ),
        key=lambda row: (
            row["max_support"]
        ),
        reverse=True,
    )

    for row in negatives_sorted:
        print(
            f"{row['id']} "
            f"{row['label']:<13} "
            f"{row['max_support']:.4f}"
        )

    print()
    print(
        "=" * 105
    )

    print(
        "SUPPORTED RETRIEVAL CHECK"
    )

    for row in rows:
        if row["label"] == "SUPPORTED":
            print(
                f"{row['id']} "
                f"ground_truth_rank="
                f"{row['anchor_rank']}"
            )

    print()
    print(
        "=" * 105
    )

    print(
        "LATENCY"
    )

    print(
        "Median:",
        f"{statistics.median(latencies):.0f} ms",
    )

    print(
        "Max:",
        f"{max(latencies):.0f} ms",
    )

    print()
    print(
        "This benchmark tests BGE as an "
        "entailment proxy; BGE is not an NLI model."
    )

    print(
        "No threshold was selected."
    )

    print(
        "No production verifier was changed."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
