from __future__ import annotations

import json
import re
import statistics
import time
import unicodedata
from pathlib import Path
from uuid import (
    NAMESPACE_URL,
    uuid5,
)

import numpy as np
from docling.document_converter import (
    DocumentConverter,
)
from sentence_transformers import (
    SentenceTransformer,
)

from app.application.ports.retrieval import (
    RetrievedChunk,
)
from research.parsing.experimental_structure_chunker import (
    ExperimentalStructureChunker,
)
from research.retrieval.bge_reranker import (
    BgeRerankerAdapter,
)


MODEL_NAME = (
    "intfloat/multilingual-e5-base"
)

MODEL_REVISION = (
    "d128750597153bb5987e10b1c3493a34e5a4502a"
)

REPO_ROOT = (
    Path(__file__).resolve().parents[3]
)

SOURCE = (
    REPO_ROOT
    / "documents"
    / "sample"
    / "test-policy.pdf"
)

DATASET = Path(
    "tests/fixtures/"
    "evidence_calibration_v1.json"
)

GROUND_TRUTH = {
    "A01": (
        "primary responsibility for the maintenance "
        "of international peace and security"
    ),
    "A02": (
        "the purposes of the united nations are"
    ),
    "A03": (
        "admission of any such state to membership "
        "in the united nations"
    ),
    "A04": (
        "decisions of the general assembly on important "
        "questions shall be made by a two-thirds majority"
    ),
    "A05": (
        "shall be appointed by the general assembly "
        "upon the recommendation of the security council"
    ),
    "A06": (
        "may be expelled from the organization "
        "by the general assembly upon the recommendation "
        "of the security council"
    ),
    "A07": (
        "inherent right of individual or collective"
    ),
    "A08": (
        "principal judicial organ of the united nations"
    ),
}

CANDIDATE_K = 5


def normalize(
    value: str,
) -> str:
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


def contains_anchor(
    content: str,
    anchor: str,
) -> bool:
    return (
        normalize(anchor)
        in normalize(content)
    )


def recall_at(
    ranks,
    k: int,
) -> float:
    return sum(
        1
        for rank in ranks
        if (
            rank is not None
            and rank <= k
        )
    ) / len(ranks)


def mrr(
    ranks,
) -> float:
    return sum(
        (
            1.0 / rank
            if rank is not None
            else 0.0
        )
        for rank in ranks
    ) / len(ranks)


def rank_text(
    rank,
) -> str:
    return (
        str(rank)
        if rank is not None
        else "MISS"
    )


def make_retrieved_chunk(
    *,
    index: int,
    chunk,
    score: float,
) -> RetrievedChunk:
    base = (
        f"experimental-structure-{index}"
    )

    return RetrievedChunk(
        chunk_id=uuid5(
            NAMESPACE_URL,
            base + "-chunk",
        ),
        document_id=uuid5(
            NAMESPACE_URL,
            "experimental-structure-document",
        ),
        document_version_id=uuid5(
            NAMESPACE_URL,
            "experimental-structure-version",
        ),
        page_id=uuid5(
            NAMESPACE_URL,
            base + "-page",
        ),
        page_number=chunk.page_number,
        chunk_index=index,
        content=chunk.content,
        section_title=(
            chunk.structural_label
            or None
        ),
        language="en",
        score=float(score),
    )


def main() -> None:
    cases = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    print()
    print(
        "=== Structure-aware E5 + BGE top-5 A/B ==="
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

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    print(
        "Embedding structure corpus..."
    )

    vectors = model.encode(
        [
            "passage: "
            + item.content
            for item in chunks
        ],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print()
    print(
        "Loading BGE reranker..."
    )

    reranker = BgeRerankerAdapter(
        batch_size=5,
        max_length=1024,
    )

    print(
        "Device:",
        reranker.device,
    )

    semantic_ranks = []
    reranked_ranks = []
    latencies = []

    answerable_bge_scores = []
    unanswerable_bge_scores = []

    print()
    print(
        "=" * 80
    )

    print(
        "PER-QUESTION RESULTS"
    )

    for case in cases:
        question = case[
            "question"
        ]

        query_vector = model.encode(
            [
                "query: "
                + question
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

        top_indices = order[
            :CANDIDATE_K
        ]

        candidates = tuple(
            make_retrieved_chunk(
                index=int(index),
                chunk=chunks[
                    int(index)
                ],
                score=float(
                    scores[
                        int(index)
                    ]
                ),
            )
            for index in top_indices
        )

        start = (
            time.perf_counter()
        )

        reranked = reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=CANDIDATE_K,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        latencies.append(
            elapsed_ms
        )

        if case["label"] == "ANSWERABLE":
            anchor = GROUND_TRUTH[
                case["id"]
            ]

            semantic_rank = None

            for rank, candidate in enumerate(
                candidates,
                start=1,
            ):
                if contains_anchor(
                    candidate.content,
                    anchor,
                ):
                    semantic_rank = rank
                    break

            bge_rank = None
            bge_score = None

            for rank, item in enumerate(
                reranked,
                start=1,
            ):
                if contains_anchor(
                    item.chunk.content,
                    anchor,
                ):
                    bge_rank = rank
                    bge_score = (
                        item.rerank_score
                    )
                    break

            semantic_ranks.append(
                semantic_rank
            )

            reranked_ranks.append(
                bge_rank
            )

            if bge_score is not None:
                answerable_bge_scores.append(
                    bge_score
                )

            print(
                f"{case['id']} "
                f"E5={rank_text(semantic_rank)} "
                f"BGE={rank_text(bge_rank)} "
                f"answer_score="
                f"{bge_score:.4f} "
                f"latency={elapsed_ms:.0f}ms"
            )

        else:
            top_bge_score = (
                reranked[0].rerank_score
                if reranked
                else None
            )

            if top_bge_score is not None:
                unanswerable_bge_scores.append(
                    top_bge_score
                )

            top_chunk = (
                reranked[0].chunk
                if reranked
                else None
            )

            print(
                f"{case['id']} "
                f"UNANSWERABLE "
                f"top_bge="
                f"{top_bge_score:.4f} "
                f"latency={elapsed_ms:.0f}ms"
            )

            if top_chunk is not None:
                snippet = " ".join(
                    top_chunk.content.split()
                )[:180]

                print(
                    "   ",
                    snippet,
                )

    print()
    print(
        "=" * 80
    )

    print(
        "RETRIEVAL QUALITY"
    )

    print()

    for k in (
        1,
        3,
        5,
    ):
        print(
            f"Recall@{k}: "
            f"E5="
            f"{recall_at(semantic_ranks, k):.3f} "
            f"BGE="
            f"{recall_at(reranked_ranks, k):.3f}"
        )

    print(
        "MRR:      "
        f"E5="
        f"{mrr(semantic_ranks):.3f} "
        f"BGE="
        f"{mrr(reranked_ranks):.3f}"
    )

    print()
    print(
        "BGE SCORE DIAGNOSTIC"
    )

    if (
        answerable_bge_scores
        and unanswerable_bge_scores
    ):
        print(
            "Answer-bearing min:",
            f"{min(answerable_bge_scores):.4f}",
        )

        print(
            "Answer-bearing median:",
            f"{statistics.median(answerable_bge_scores):.4f}",
        )

        print(
            "Unsupported top max:",
            f"{max(unanswerable_bge_scores):.4f}",
        )

        observed_margin = (
            min(answerable_bge_scores)
            - max(unanswerable_bge_scores)
        )

        print(
            "Observed margin:",
            f"{observed_margin:.4f}",
        )

    print()
    print(
        "BGE LATENCY"
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
        "No canonical chunks, embeddings, "
        "or database rows were modified."
    )


if __name__ == "__main__":
    main()
