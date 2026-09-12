from __future__ import annotations

import re
import statistics
import time
import unicodedata
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


CASES = {
    "M01": [
        {
            "name": "admission_recommendation",
            "query": (
                "Quel organe recommande l'admission "
                "d'un Etat aux Nations Unies ?"
            ),
            "anchor": (
                "admission of any such state to membership "
                "in the united nations"
            ),
        },
        {
            "name": "admission_decision",
            "query": (
                "Quel organe prend la decision finale "
                "d'admission d'un Etat aux Nations Unies ?"
            ),
            "anchor": (
                "admission of any such state to membership "
                "in the united nations"
            ),
        },
    ],
    "M02": [
        {
            "name": "general_assembly_role",
            "query": (
                "Quel role joue l'Assemblee generale "
                "dans la nomination du Secretaire general ?"
            ),
            "anchor": (
                "shall be appointed by the general assembly "
                "upon the recommendation of the security council"
            ),
        },
        {
            "name": "security_council_role",
            "query": (
                "Quel role joue le Conseil de securite "
                "dans la nomination du Secretaire general ?"
            ),
            "anchor": (
                "shall be appointed by the general assembly "
                "upon the recommendation of the security council"
            ),
        },
    ],
    "M03": [
        {
            "name": "admission",
            "query": (
                "Comment un Etat est-il admis comme "
                "membre des Nations Unies ?"
            ),
            "anchor": (
                "admission of any such state to membership "
                "in the united nations"
            ),
        },
        {
            "name": "expulsion",
            "query": (
                "Comment un Etat membre peut-il etre "
                "expulse des Nations Unies ?"
            ),
            "anchor": (
                "may be expelled from the organization "
                "by the general assembly upon the recommendation "
                "of the security council"
            ),
        },
    ],
    "M04": [
        {
            "name": "security_council_responsibility",
            "query": (
                "Quelle est la responsabilite principale "
                "du Conseil de securite ?"
            ),
            "anchor": (
                "primary responsibility for the maintenance "
                "of international peace and security"
            ),
        },
        {
            "name": "self_defence",
            "query": (
                "Quel droit de defense individuelle ou "
                "collective la Charte preserve-t-elle ?"
            ),
            "anchor": (
                "inherent right of individual or collective"
            ),
        },
    ],
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
    base = f"atomic-support-{index}"

    return RetrievedChunk(
        chunk_id=uuid5(
            NAMESPACE_URL,
            base + "-chunk",
        ),
        document_id=uuid5(
            NAMESPACE_URL,
            "atomic-support-document",
        ),
        document_version_id=uuid5(
            NAMESPACE_URL,
            "atomic-support-version",
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


def main() -> None:
    print()
    print(
        "=== Oracle atomic-query evidence support ==="
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

    all_claim_scores = []
    all_latencies = []

    for case_id, claims in CASES.items():
        print()
        print(
            "=" * 100
        )

        print(
            case_id
        )

        case_scores = []
        case_coverage = []

        for claim in claims:
            query = claim[
                "query"
            ]

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

            all_latencies.append(
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

            case_scores.append(
                max_support
            )

            all_claim_scores.append(
                max_support
            )

            anchor_rank = None
            anchor_support = None

            for rank, (
                index,
                scored,
            ) in enumerate(
                zip(
                    order,
                    support,
                    strict=True,
                ),
                start=1,
            ):
                source_chunk = chunks[
                    int(index)
                ]

                if contains(
                    source_chunk.content,
                    claim["anchor"],
                ):
                    anchor_rank = rank
                    anchor_support = (
                        scored.support_score
                    )
                    break

            covered = (
                anchor_rank is not None
            )

            case_coverage.append(
                covered
            )

            score_text = ", ".join(
                f"{value:.4f}"
                for value in scores
            )

            print()
            print(
                f"claim={claim['name']}"
            )

            print(
                f"  query: {query}"
            )

            print(
                f"  E5 ground-truth rank: "
                f"{anchor_rank}"
            )

            print(
                f"  BGE top3: [{score_text}]"
            )

            print(
                f"  max support: "
                f"{max_support:.4f}"
            )

            if anchor_support is not None:
                print(
                    f"  answer-bearing support: "
                    f"{anchor_support:.4f}"
                )

            print(
                f"  latency: "
                f"{elapsed_ms:.0f} ms"
            )

        weakest_claim = min(
            case_scores,
            default=0.0,
        )

        complete_retrieval = all(
            case_coverage
        )

        print()
        print(
            "CASE SUMMARY"
        )

        print(
            "  all required evidence "
            f"in top-{TOP_K}: "
            f"{complete_retrieval}"
        )

        print(
            "  per-claim maximums:",
            ", ".join(
                f"{value:.4f}"
                for value in case_scores
            ),
        )

        print(
            "  weakest-claim support:",
            f"{weakest_claim:.4f}",
        )

    print()
    print(
        "=" * 100
    )

    print(
        "GLOBAL ATOMIC SUPPORT"
    )

    print(
        "Minimum claim support:",
        f"{min(all_claim_scores):.4f}",
    )

    print(
        "Median claim support:",
        f"{statistics.median(all_claim_scores):.4f}",
    )

    print(
        "Maximum claim support:",
        f"{max(all_claim_scores):.4f}",
    )

    print()
    print(
        "BGE VERIFICATION LATENCY"
    )

    print(
        "Median per atomic query:",
        f"{statistics.median(all_latencies):.0f} ms",
    )

    print(
        "Max per atomic query:",
        f"{max(all_latencies):.0f} ms",
    )

    print()
    print(
        "This is still oracle decomposition."
    )

    print(
        "No production QueryDecomposer was introduced."
    )

    print(
        "No EvidencePolicy threshold was selected."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
