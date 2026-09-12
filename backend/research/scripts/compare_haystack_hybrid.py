from __future__ import annotations

import json
import re
import statistics
import time
import unicodedata
from pathlib import Path

import numpy as np
from docling.document_converter import (
    DocumentConverter,
)
from sentence_transformers import (
    SentenceTransformer,
)

from research.parsing.experimental_structure_chunker import (
    ExperimentalStructureChunker,
)
from research.retrieval.experimental_haystack_hybrid import (
    ExperimentalHaystackHybridConnector,
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


def semantic_rank(
    order,
    chunks,
    anchor: str,
) -> int | None:
    for rank, index in enumerate(
        order,
        start=1,
    ):
        if contains_anchor(
            chunks[int(index)].content,
            anchor,
        ):
            return rank

    return None


def hybrid_rank(
    results,
    anchor: str,
) -> int | None:
    for rank, item in enumerate(
        results,
        start=1,
    ):
        if contains_anchor(
            item.content,
            anchor,
        ):
            return rank

    return None


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


def main() -> None:
    cases = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    print()
    print(
        "=== Haystack hybrid A/B ==="
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
        "Authorized experimental chunks:",
        len(chunks),
    )

    print(
        "Loading pinned E5..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    passages = [
        "passage: " + item.content
        for item in chunks
    ]

    print(
        "Embedding corpus..."
    )

    vectors = model.encode(
        passages,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print(
        "Creating Haystack connector..."
    )

    connector = (
        ExperimentalHaystackHybridConnector(
            chunks=chunks,
            embeddings=vectors,
            semantic_weight=0.8,
            lexical_weight=0.2,
            candidate_k=20,
        )
    )

    semantic_ranks = []
    hybrid_ranks = []
    hybrid_latencies = []

    print()
    print(
        "=" * 80
    )
    print(
        "ANSWERABLE CASES"
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

        start = time.perf_counter()

        hybrid = connector.search(
            query=question,
            query_embedding=(
                query_vector.tolist()
            ),
            top_k=20,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        hybrid_latencies.append(
            elapsed_ms
        )

        if (
            case["label"]
            != "ANSWERABLE"
        ):
            continue

        anchor = GROUND_TRUTH[
            case["id"]
        ]

        s_rank = semantic_rank(
            order,
            chunks,
            anchor,
        )

        h_rank = hybrid_rank(
            hybrid,
            anchor,
        )

        semantic_ranks.append(
            s_rank
        )

        hybrid_ranks.append(
            h_rank
        )

        gt_index = None

        for index, chunk in enumerate(
            chunks
        ):
            if contains_anchor(
                chunk.content,
                anchor,
            ):
                gt_index = index
                break

        lexical_rank = None

        if gt_index is not None:
            for result in hybrid:
                if (
                    result.chunk_index
                    == gt_index
                ):
                    lexical_rank = (
                        result.lexical_rank
                    )
                    break

        print(
            f"{case['id']} "
            f"semantic={rank_text(s_rank)} "
            f"hybrid={rank_text(h_rank)} "
            f"lexical_rank="
            f"{rank_text(lexical_rank)} "
            f"latency={elapsed_ms:.1f}ms"
        )

    print()
    print(
        "=" * 80
    )
    print(
        "RETRIEVAL METRICS"
    )
    print()

    for k in (
        1,
        3,
        5,
    ):
        print(
            f"Recall@{k}: "
            f"semantic="
            f"{recall_at(semantic_ranks, k):.3f} "
            f"hybrid="
            f"{recall_at(hybrid_ranks, k):.3f}"
        )

    print(
        "MRR:      "
        f"semantic="
        f"{mrr(semantic_ranks):.3f} "
        f"hybrid="
        f"{mrr(hybrid_ranks):.3f}"
    )

    print()
    print(
        "HYBRID LATENCY"
    )

    print(
        "Median:",
        f"{statistics.median(hybrid_latencies):.1f} ms",
    )

    print(
        "Max:",
        f"{max(hybrid_latencies):.1f} ms",
    )

    print()
    print(
        "SECURITY NOTE:"
    )

    print(
        "Haystack received only the "
        "pre-authorized experimental corpus."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
