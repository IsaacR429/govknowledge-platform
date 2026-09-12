from __future__ import annotations

import json
import re
import statistics
import unicodedata
from pathlib import Path
from uuid import UUID

import numpy as np
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

from app.application.ports.authorization import AuthorizationScope
from app.domain.enums.permission import PermissionAction
from app.infrastructure.database.models.documents import DocumentVersion
from app.infrastructure.database.session import SessionLocal
from research.parsing.experimental_structure_chunker import (
    ExperimentalStructureChunker,
)
from app.infrastructure.retrieval.postgres_vector_retrieval import (
    PostgresVectorRetrievalAdapter,
)


VERSION_ID = UUID(
    "86aa0794-a065-4819-a4fe-01694fad6122"
)

MODEL_NAME = "intfloat/multilingual-e5-base"

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
    "tests/fixtures/evidence_calibration_v1.json"
)

BASELINE_K = 50


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

    # Repair common OCR line-break hyphenation first.
    value = re.sub(
        r"(?<=\w)-\s*(?=\w)",
        "",
        value,
    )

    # Treat punctuation and OCR separators as whitespace.
    # Example: "The^Purposes" -> "The Purposes".
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


def find_baseline_rank(
    results,
    anchor: str,
) -> int | None:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        if contains_anchor(
            result.content,
            anchor,
        ):
            return rank

    return None


def find_experimental_rank(
    order: np.ndarray,
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


def recall_at(
    ranks: list[int | None],
    k: int,
) -> float:
    if not ranks:
        return 0.0

    hits = sum(
        1
        for rank in ranks
        if (
            rank is not None
            and rank <= k
        )
    )

    return hits / len(ranks)


def mrr(
    ranks: list[int | None],
) -> float:
    if not ranks:
        return 0.0

    return sum(
        (
            1.0 / rank
            if rank is not None
            else 0.0
        )
        for rank in ranks
    ) / len(ranks)


def format_rank(
    rank: int | None,
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
        "=== Chunking strategy A/B ==="
    )

    print()
    print(
        "Parsing structural candidate corpus..."
    )

    parsed = (
        DocumentConverter()
        .convert(
            SOURCE,
            raises_on_error=True,
        )
    )

    experimental_chunks = (
        ExperimentalStructureChunker(
            max_chars=2200,
        )
        .chunk(
            parsed.document
        )
    )

    print(
        "Experimental chunks:",
        len(experimental_chunks),
    )

    print(
        "Loading pinned E5 model..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    passage_texts = [
        "passage: " + chunk.content
        for chunk in experimental_chunks
    ]

    print(
        "Embedding experimental chunks..."
    )

    passage_vectors = model.encode(
        passage_texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    db = SessionLocal()

    try:
        version = db.get(
            DocumentVersion,
            VERSION_ID,
        )

        if version is None:
            raise RuntimeError(
                "Document version not found."
            )

        scope = AuthorizationScope(
            user_id=UUID(
                "00000000-0000-0000-0000-000000000001"
            ),
            action=PermissionAction.READ,
            document_ids=frozenset(
                {version.document_id}
            ),
        )

        baseline = (
            PostgresVectorRetrievalAdapter(
                db
            )
        )

        baseline_ranks = []
        experimental_ranks = []

        baseline_answerable_top = []
        experimental_answerable_top = []

        baseline_unanswerable_top = []
        experimental_unanswerable_top = []

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

            baseline_results = (
                baseline.search_authorized(
                    scope=scope,
                    query_vector=(
                        query_vector.tolist()
                    ),
                    model_name=MODEL_NAME,
                    model_revision=MODEL_REVISION,
                    top_k=BASELINE_K,
                )
            )

            experimental_scores = (
                passage_vectors
                @ query_vector
            )

            experimental_order = (
                np.argsort(
                    -experimental_scores
                )
            )

            baseline_top = (
                baseline_results[0].score
                if baseline_results
                else float("-inf")
            )

            experimental_top = float(
                experimental_scores[
                    experimental_order[0]
                ]
            )

            if case["label"] == "ANSWERABLE":
                anchor = GROUND_TRUTH[
                    case["id"]
                ]

                baseline_rank = (
                    find_baseline_rank(
                        baseline_results,
                        anchor,
                    )
                )

                experimental_rank = (
                    find_experimental_rank(
                        experimental_order,
                        experimental_chunks,
                        anchor,
                    )
                )

                baseline_ranks.append(
                    baseline_rank
                )

                experimental_ranks.append(
                    experimental_rank
                )

                baseline_answerable_top.append(
                    baseline_top
                )

                experimental_answerable_top.append(
                    experimental_top
                )

                print(
                    f"{case['id']} ANSWERABLE  "
                    f"ground-truth rank "
                    f"{format_rank(baseline_rank)} "
                    f"-> "
                    f"{format_rank(experimental_rank)} "
                    f"| top1 "
                    f"{baseline_top:.4f} "
                    f"-> "
                    f"{experimental_top:.4f}"
                )

            else:
                baseline_unanswerable_top.append(
                    baseline_top
                )

                experimental_unanswerable_top.append(
                    experimental_top
                )

                print(
                    f"{case['id']} UNANSWERABLE "
                    f"top1 "
                    f"{baseline_top:.4f} "
                    f"-> "
                    f"{experimental_top:.4f}"
                )

        print()
        print(
            "=" * 80
        )

        print(
            "GROUND-TRUTH RETRIEVAL QUALITY"
        )

        print()

        for k in (
            1,
            3,
            5,
        ):
            print(
                f"Recall@{k}: "
                f"baseline="
                f"{recall_at(baseline_ranks, k):.3f} "
                f"structure="
                f"{recall_at(experimental_ranks, k):.3f}"
            )

        print(
            "MRR:      "
            f"baseline="
            f"{mrr(baseline_ranks):.3f} "
            f"structure="
            f"{mrr(experimental_ranks):.3f}"
        )

        print()
        print(
            "ANSWERABILITY SCORE SEPARATION"
        )

        baseline_margin = (
            min(
                baseline_answerable_top
            )
            - max(
                baseline_unanswerable_top
            )
        )

        experimental_margin = (
            min(
                experimental_answerable_top
            )
            - max(
                experimental_unanswerable_top
            )
        )

        print(
            "Baseline top1 margin:",
            f"{baseline_margin:.4f}",
        )

        print(
            "Structure top1 margin:",
            f"{experimental_margin:.4f}",
        )

        print()
        print(
            "CHUNK CORPUS"
        )

        lengths = [
            len(chunk.content)
            for chunk
            in experimental_chunks
        ]

        print(
            "Canonical chunks:",
            107,
        )

        print(
            "Structure chunks:",
            len(
                experimental_chunks
            ),
        )

        print(
            "Structure median chars:",
            int(
                statistics.median(
                    lengths
                )
            ),
        )

        print(
            "Structure tiny <100 chars:",
            sum(
                1
                for length in lengths
                if length < 100
            ),
        )

        print()
        print(
            "No database rows were "
            "created or modified."
        )

    finally:
        db.rollback()
        db.close()


if __name__ == "__main__":
    main()
