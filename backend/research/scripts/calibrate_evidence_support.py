from __future__ import annotations

import json
import statistics
import time
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

DATASET = Path(
    "tests/fixtures/evidence_calibration_v1.json"
)

TOP_K = 3

THRESHOLDS = (
    0.001,
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.200,
    0.300,
    0.400,
    0.500,
)


def make_chunk(
    *,
    index: int,
    chunk,
    score: float,
) -> RetrievedChunk:
    base = f"support-benchmark-{index}"

    return RetrievedChunk(
        chunk_id=uuid5(
            NAMESPACE_URL,
            base + "-chunk",
        ),
        document_id=uuid5(
            NAMESPACE_URL,
            "support-benchmark-document",
        ),
        document_version_id=uuid5(
            NAMESPACE_URL,
            "support-benchmark-version",
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


def evaluate_threshold(
    rows,
    threshold: float,
) -> tuple[int, int, int, int]:
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for row in rows:
        predicted_answerable = (
            row["max_support"]
            >= threshold
        )

        actual_answerable = (
            row["label"]
            == "ANSWERABLE"
        )

        if predicted_answerable and actual_answerable:
            tp += 1
        elif predicted_answerable and not actual_answerable:
            fp += 1
        elif not predicted_answerable and not actual_answerable:
            tn += 1
        else:
            fn += 1

    return tp, fp, tn, fn


def main() -> None:
    cases = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    print()
    print(
        "=== Evidence-support calibration ==="
    )

    print(
        "Parsing structure-aware corpus..."
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
        .chunk(parsed.document)
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
        "Loading BGE support verifier..."
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
        "=" * 88
    )
    print(
        "PER-QUESTION SUPPORT"
    )

    for case in cases:
        question = case["question"]

        query_vector = embedder.encode(
            [
                "query: " + question
            ],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        similarities = (
            vectors @ query_vector
        )

        order = np.argsort(
            -similarities
        )[:TOP_K]

        evidence = tuple(
            make_chunk(
                index=int(index),
                chunk=chunks[int(index)],
                score=float(
                    similarities[int(index)]
                ),
            )
            for index in order
        )

        started = time.perf_counter()

        support = verifier.score(
            query=question,
            evidence=evidence,
        )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        latencies.append(
            elapsed_ms
        )

        support_scores = [
            item.support_score
            for item in support
        ]

        max_support = max(
            support_scores,
            default=0.0,
        )

        rows.append(
            {
                "id": case["id"],
                "label": case["label"],
                "max_support": max_support,
                "scores": support_scores,
                "latency_ms": elapsed_ms,
            }
        )

        score_text = ", ".join(
            f"{score:.4f}"
            for score in support_scores
        )

        print(
            f"{case['id']} "
            f"{case['label']:<12} "
            f"E5_top1="
            f"{similarities[int(order[0])]:.4f} "
            f"BGE=[{score_text}] "
            f"max={max_support:.4f} "
            f"latency={elapsed_ms:.0f}ms"
        )

    answerable = [
        row["max_support"]
        for row in rows
        if row["label"] == "ANSWERABLE"
    ]

    unsupported = [
        row["max_support"]
        for row in rows
        if row["label"] != "ANSWERABLE"
    ]

    print()
    print(
        "=" * 88
    )
    print(
        "DISTRIBUTIONS"
    )

    print(
        "Answerable min:   ",
        f"{min(answerable):.4f}",
    )

    print(
        "Answerable median:",
        f"{statistics.median(answerable):.4f}",
    )

    print(
        "Answerable max:   ",
        f"{max(answerable):.4f}",
    )

    print(
        "Unsupported min:  ",
        f"{min(unsupported):.4f}",
    )

    print(
        "Unsupported median:",
        f"{statistics.median(unsupported):.4f}",
    )

    print(
        "Unsupported max:  ",
        f"{max(unsupported):.4f}",
    )

    observed_margin = (
        min(answerable)
        - max(unsupported)
    )

    print(
        "Observed margin:  ",
        f"{observed_margin:.4f}",
    )

    print()
    print(
        "=" * 88
    )
    print(
        "THRESHOLD DIAGNOSTIC"
    )

    print(
        "threshold   TP   FP   TN   FN"
    )

    for threshold in THRESHOLDS:
        tp, fp, tn, fn = (
            evaluate_threshold(
                rows,
                threshold,
            )
        )

        print(
            f"{threshold:>8.3f}"
            f"{tp:>5}"
            f"{fp:>5}"
            f"{tn:>5}"
            f"{fn:>5}"
        )

    print()
    print(
        "=" * 88
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
        "IMPORTANT:"
    )

    print(
        "These thresholds are diagnostics only."
    )

    print(
        "No EvidencePolicy configuration "
        "was changed."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
