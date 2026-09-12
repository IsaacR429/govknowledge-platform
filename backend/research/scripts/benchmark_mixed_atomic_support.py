from __future__ import annotations

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

TOP_K = 3

THRESHOLDS = (
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
)


CASES = [
    {
        "id": "SS01",
        "expected": "ANSWERABLE",
        "claims": [
            {
                "name": "admission",
                "expected_support": True,
                "query": (
                    "Comment un Etat est-il admis comme "
                    "membre des Nations Unies ?"
                ),
            },
            {
                "name": "expulsion",
                "expected_support": True,
                "query": (
                    "Comment un Etat membre peut-il etre "
                    "expulse des Nations Unies ?"
                ),
            },
        ],
    },
    {
        "id": "SS02",
        "expected": "ANSWERABLE",
        "claims": [
            {
                "name": "security_council_responsibility",
                "expected_support": True,
                "query": (
                    "Quelle est la responsabilite principale "
                    "du Conseil de securite ?"
                ),
            },
            {
                "name": "self_defence",
                "expected_support": True,
                "query": (
                    "Quel droit de defense individuelle ou "
                    "collective la Charte preserve-t-elle ?"
                ),
            },
        ],
    },
    {
        "id": "SU01",
        "expected": "UNANSWERABLE",
        "claims": [
            {
                "name": "secretary_general_appointment",
                "expected_support": True,
                "query": (
                    "Comment le Secretaire general "
                    "est-il nomme ?"
                ),
            },
            {
                "name": "secretary_general_ten_year_term",
                "expected_support": False,
                "query": (
                    "La Charte fixe-t-elle exactement a dix ans "
                    "le mandat du Secretaire general ?"
                ),
            },
        ],
    },
    {
        "id": "SU02",
        "expected": "UNANSWERABLE",
        "claims": [
            {
                "name": "security_council_responsibility",
                "expected_support": True,
                "query": (
                    "Quelle est la responsabilite principale "
                    "du Conseil de securite ?"
                ),
            },
            {
                "name": "world_tax",
                "expected_support": False,
                "query": (
                    "L'Article 24 autorise-t-il explicitement "
                    "le Conseil de securite a imposer "
                    "une taxe mondiale ?"
                ),
            },
        ],
    },
    {
        "id": "SU03",
        "expected": "UNANSWERABLE",
        "claims": [
            {
                "name": "principal_judicial_organ",
                "expected_support": True,
                "query": (
                    "Quel est l'organe judiciaire principal "
                    "des Nations Unies ?"
                ),
            },
            {
                "name": "cancel_national_laws",
                "expected_support": False,
                "query": (
                    "La Cour internationale de Justice "
                    "peut-elle annuler toute loi nationale ?"
                ),
            },
        ],
    },
    {
        "id": "SU04",
        "expected": "UNANSWERABLE",
        "claims": [
            {
                "name": "self_defence",
                "expected_support": True,
                "query": (
                    "Quel droit de defense individuelle ou "
                    "collective la Charte preserve-t-elle ?"
                ),
            },
            {
                "name": "preventive_war",
                "expected_support": False,
                "query": (
                    "L'Article 51 autorise-t-il automatiquement "
                    "toute intervention militaire preventive ?"
                ),
            },
        ],
    },
    {
        "id": "UU01",
        "expected": "UNANSWERABLE",
        "claims": [
            {
                "name": "security_council_budget",
                "expected_support": False,
                "query": (
                    "Quel est le budget annuel actuel du "
                    "Conseil de securite en dollars americains ?"
                ),
            },
            {
                "name": "security_council_armies",
                "expected_support": False,
                "query": (
                    "Le Conseil de securite controle-t-il "
                    "directement toutes les armees nationales ?"
                ),
            },
        ],
    },
    {
        "id": "UU02",
        "expected": "UNANSWERABLE",
        "claims": [
            {
                "name": "judge_salary_amount",
                "expected_support": False,
                "query": (
                    "Combien gagne aujourd'hui chaque juge "
                    "de la Cour internationale de Justice "
                    "en francs suisses ?"
                ),
            },
            {
                "name": "president_home_address",
                "expected_support": False,
                "query": (
                    "Quelle est l'adresse actuelle du domicile "
                    "officiel du President de la Cour "
                    "internationale de Justice ?"
                ),
            },
        ],
    },
]


def make_chunk(
    *,
    index: int,
    chunk,
    score: float,
) -> RetrievedChunk:
    base = f"mixed-support-{index}"

    return RetrievedChunk(
        chunk_id=uuid5(
            NAMESPACE_URL,
            base + "-chunk",
        ),
        document_id=uuid5(
            NAMESPACE_URL,
            "mixed-support-document",
        ),
        document_version_id=uuid5(
            NAMESPACE_URL,
            "mixed-support-version",
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
    *,
    threshold: float,
):
    tp = fp = tn = fn = 0
    errors = []

    for row in rows:
        predicted_answerable = all(
            claim["max_support"] >= threshold
            for claim in row["claims"]
        )

        actual_answerable = (
            row["expected"]
            == "ANSWERABLE"
        )

        if (
            predicted_answerable
            and actual_answerable
        ):
            tp += 1

        elif (
            predicted_answerable
            and not actual_answerable
        ):
            fp += 1
            errors.append(
                row["id"] + ":FP"
            )

        elif (
            not predicted_answerable
            and not actual_answerable
        ):
            tn += 1

        else:
            fn += 1
            errors.append(
                row["id"] + ":FN"
            )

    return tp, fp, tn, fn, errors


def main() -> None:
    print()
    print(
        "=== Mixed-support atomic benchmark ==="
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

    for case in CASES:
        print()
        print(
            "=" * 100
        )

        print(
            case["id"],
            "expected=",
            case["expected"],
        )

        claim_rows = []

        for claim in case["claims"]:
            query = claim["query"]

            query_vector = embedder.encode(
                [
                    "query: " + query
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
                query=query,
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

            claim_rows.append(
                {
                    "name": claim["name"],
                    "expected_support": (
                        claim[
                            "expected_support"
                        ]
                    ),
                    "max_support": (
                        max_support
                    ),
                }
            )

            print()
            print(
                f"claim={claim['name']}"
            )

            print(
                "  expected support:",
                claim[
                    "expected_support"
                ],
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

            print(
                "  latency:",
                f"{elapsed_ms:.0f} ms",
            )

        weakest = min(
            row["max_support"]
            for row in claim_rows
        )

        strongest = max(
            row["max_support"]
            for row in claim_rows
        )

        print()
        print(
            "CASE SUMMARY"
        )

        print(
            "  strongest-claim:",
            f"{strongest:.4f}",
        )

        print(
            "  weakest-claim:",
            f"{weakest:.4f}",
        )

        rows.append(
            {
                "id": case["id"],
                "expected": (
                    case["expected"]
                ),
                "claims": claim_rows,
                "strongest": strongest,
                "weakest": weakest,
            }
        )

    print()
    print(
        "=" * 100
    )

    print(
        "SUPPORTED VS UNSUPPORTED CLAIMS"
    )

    supported = [
        claim["max_support"]
        for row in rows
        for claim in row["claims"]
        if claim["expected_support"]
    ]

    unsupported = [
        claim["max_support"]
        for row in rows
        for claim in row["claims"]
        if not claim["expected_support"]
    ]

    print(
        "Supported:",
        f"min={min(supported):.4f}",
        f"median={statistics.median(supported):.4f}",
        f"max={max(supported):.4f}",
    )

    print(
        "Unsupported:",
        f"min={min(unsupported):.4f}",
        f"median={statistics.median(unsupported):.4f}",
        f"max={max(unsupported):.4f}",
    )

    observed_margin = (
        min(supported)
        - max(unsupported)
    )

    print(
        "Observed claim margin:",
        f"{observed_margin:.4f}",
    )

    print()
    print(
        "=" * 100
    )

    print(
        "COMPOSITE ALL-CLAIMS DIAGNOSTIC"
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
            threshold=threshold,
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
        "WHY MAX-CLAIM AGGREGATION IS DANGEROUS"
    )

    for row in rows:
        if (
            row["expected"]
            == "UNANSWERABLE"
            and row["strongest"] >= 0.40
        ):
            print(
                f"{row['id']} "
                f"strongest={row['strongest']:.4f} "
                f"weakest={row['weakest']:.4f}"
            )

    print()
    print(
        "=" * 100
    )

    print(
        "LATENCY"
    )

    print(
        "Median per atomic claim:",
        f"{statistics.median(latencies):.0f} ms",
    )

    print(
        "Max per atomic claim:",
        f"{max(latencies):.0f} ms",
    )

    print()
    print(
        "No threshold was selected."
    )

    print(
        "No production decomposer was introduced."
    )

    print(
        "No EvidencePolicy configuration changed."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
