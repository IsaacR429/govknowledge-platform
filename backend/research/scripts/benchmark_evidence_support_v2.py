from __future__ import annotations

import json
import statistics
import time
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

DATASET = Path(__file__).resolve().parents[1] / "fixtures" / "evidence_support_benchmark_v2.json"

TOP_K = 3

THRESHOLDS = (
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.200,
    0.300,
    0.400,
)


def make_chunk(
    *,
    index: int,
    chunk,
    score: float,
) -> RetrievedChunk:
    base = f"support-v2-{index}"

    return RetrievedChunk(
        chunk_id=uuid5(
            NAMESPACE_URL,
            base + "-chunk",
        ),
        document_id=uuid5(
            NAMESPACE_URL,
            "support-v2-document",
        ),
        document_version_id=uuid5(
            NAMESPACE_URL,
            "support-v2-version",
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


def make_evidence_bundle(
    evidence: tuple[RetrievedChunk, ...],
) -> RetrievedChunk:
    content = "\n\n--- EVIDENCE ---\n\n".join(
        item.content
        for item in evidence
    )

    first = evidence[0]

    return RetrievedChunk(
        chunk_id=uuid5(
            NAMESPACE_URL,
            "support-v2-bundle-" + str(first.chunk_id),
        ),
        document_id=first.document_id,
        document_version_id=first.document_version_id,
        page_id=first.page_id,
        page_number=first.page_number,
        chunk_index=-1,
        content=content,
        section_title="AUTHORIZED TOP-3 EVIDENCE SET",
        language="en",
        score=max(
            item.score
            for item in evidence
        ),
    )


def classify(
    value: float,
    threshold: float,
) -> bool:
    return value >= threshold


def confusion(
    rows,
    *,
    field: str,
    threshold: float,
):
    tp = fp = tn = fn = 0
    false_positive_ids = []
    false_negative_ids = []

    for row in rows:
        actual = (
            row["label"]
            == "ANSWERABLE"
        )

        predicted = classify(
            row[field],
            threshold,
        )

        if actual and predicted:
            tp += 1

        elif (
            not actual
            and predicted
        ):
            fp += 1
            false_positive_ids.append(
                row["id"]
            )

        elif (
            not actual
            and not predicted
        ):
            tn += 1

        else:
            fn += 1
            false_negative_ids.append(
                row["id"]
            )

    return (
        tp,
        fp,
        tn,
        fn,
        false_positive_ids,
        false_negative_ids,
    )


def summary(
    values,
) -> str:
    if not values:
        return "-"

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
        "=== Evidence Support Benchmark v2 ==="
    )

    print(
        "Cases:",
        len(cases),
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
            "passage: " + item.content
            for item in chunks
        ],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print(
        "Loading BGE evidence verifier..."
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
        "=" * 100
    )

    print(
        "PER-QUESTION RESULTS"
    )

    for case in cases:
        question = case[
            "question"
        ]

        query_vector = embedder.encode(
            [
                "query: " + question
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

        individual = verifier.score(
            query=question,
            evidence=evidence,
        )

        bundle = (
            make_evidence_bundle(
                evidence
            )
        )

        set_result = verifier.score(
            query=question,
            evidence=(bundle,),
        )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        latencies.append(
            elapsed_ms
        )

        individual_scores = [
            item.support_score
            for item in individual
        ]

        max_individual = max(
            individual_scores,
            default=0.0,
        )

        set_support = (
            set_result[0].support_score
            if set_result
            else 0.0
        )

        row = {
            "id": case["id"],
            "label": case["label"],
            "category": case[
                "category"
            ],
            "max_individual": (
                max_individual
            ),
            "set_support": (
                set_support
            ),
            "latency_ms": (
                elapsed_ms
            ),
        }

        rows.append(
            row
        )

        score_text = ", ".join(
            f"{value:.4f}"
            for value
            in individual_scores
        )

        print(
            f"{case['id']:<4} "
            f"{case['category']:<20} "
            f"{case['label']:<12} "
            f"individual=[{score_text}] "
            f"max={max_individual:.4f} "
            f"set={set_support:.4f} "
            f"{elapsed_ms:.0f}ms"
        )

    print()
    print(
        "=" * 100
    )

    print(
        "CATEGORY DISTRIBUTIONS"
    )

    grouped = defaultdict(
        list
    )

    for row in rows:
        grouped[
            row["category"]
        ].append(
            row
        )

    for category in sorted(
        grouped
    ):
        group = grouped[
            category
        ]

        individual_values = [
            row[
                "max_individual"
            ]
            for row in group
        ]

        set_values = [
            row[
                "set_support"
            ]
            for row in group
        ]

        print()
        print(
            category,
            f"(n={len(group)})"
        )

        print(
            "  individual:",
            summary(
                individual_values
            ),
        )

        print(
            "  evidence set:",
            summary(
                set_values
            ),
        )

    print()
    print(
        "=" * 100
    )

    print(
        "GLOBAL ANSWERABLE / UNSUPPORTED"
    )

    for field in (
        "max_individual",
        "set_support",
    ):
        answerable = [
            row[field]
            for row in rows
            if (
                row["label"]
                == "ANSWERABLE"
            )
        ]

        unsupported = [
            row[field]
            for row in rows
            if (
                row["label"]
                != "ANSWERABLE"
            )
        ]

        margin = (
            min(answerable)
            - max(unsupported)
        )

        print()
        print(
            field
        )

        print(
            "  Answerable:",
            summary(
                answerable
            ),
        )

        print(
            "  Unsupported:",
            summary(
                unsupported
            ),
        )

        print(
            "  observed margin:",
            f"{margin:.4f}",
        )

    print()
    print(
        "=" * 100
    )

    print(
        "THRESHOLD DIAGNOSTIC"
    )

    for field in (
        "max_individual",
        "set_support",
    ):
        print()
        print(
            "Signal:",
            field,
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
                fp_ids,
                fn_ids,
            ) = confusion(
                rows,
                field=field,
                threshold=threshold,
            )

            errors = (
                fp_ids
                + fn_ids
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
        "=" * 100
    )

    print(
        "PARTIAL-EVIDENCE + ADVERSARIAL DETAIL"
    )

    for row in rows:
        if row[
            "category"
        ] in {
            "partial_evidence",
            "adversarial",
            "ambiguous",
        }:
            print(
                f"{row['id']:<4} "
                f"{row['category']:<18} "
                f"max={row['max_individual']:.4f} "
                f"set={row['set_support']:.4f}"
            )

    print()
    print(
        "=" * 100
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
        "No threshold was selected."
    )

    print(
        "No EvidencePolicy configuration changed."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
