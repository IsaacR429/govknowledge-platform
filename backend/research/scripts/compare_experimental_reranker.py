from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from uuid import UUID

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.domain.enums.permission import (
    PermissionAction,
)
from app.infrastructure.database.models.documents import (
    DocumentVersion,
)
from app.infrastructure.database.session import (
    SessionLocal,
)
from research.retrieval.bge_reranker import (
    BgeRerankerAdapter,
)
from app.infrastructure.retrieval.postgres_vector_retrieval import (
    PostgresVectorRetrievalAdapter,
)
from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


VERSION_ID = UUID(
    "86aa0794-a065-4819-a4fe-01694fad6122"
)

DATASET = Path(
    "tests/fixtures/evidence_calibration_v1.json"
)

CASE_IDS = (
    "A01",
    "A05",
    "A06",
    "U05",
    "U06",
    "U08",
)

CANDIDATE_K = 20
FINAL_K = 5


def snippet(
    text: str,
    limit: int = 240,
) -> str:
    value = " ".join(
        text.split()
    )

    if len(value) <= limit:
        return value

    return (
        value[:limit]
        + "..."
    )


def percentile_95(
    values: list[float],
) -> float:
    ordered = sorted(
        values
    )

    index = max(
        0,
        math.ceil(
            0.95 * len(ordered)
        ) - 1,
    )

    return ordered[index]


def main() -> None:
    # Imported here only for the small
    # percentile helper.
    global math
    import math

    cases = {
        item["id"]: item
        for item in json.loads(
            DATASET.read_text(
                encoding="utf-8"
            )
        )
    }

    db = SessionLocal()

    try:
        version = db.get(
            DocumentVersion,
            VERSION_ID,
        )

        if version is None:
            raise RuntimeError(
                "Calibration version not found."
            )

        scope = AuthorizationScope(
            user_id=UUID(
                "00000000-0000-0000-0000-000000000001"
            ),
            action=PermissionAction.READ,
            document_ids=frozenset(
                {
                    version.document_id,
                }
            ),
        )

        embedding_provider = (
            SentenceTransformerEmbeddingProvider()
        )

        retrieval = (
            PostgresVectorRetrievalAdapter(
                db
            )
        )

        print()
        print(
            "Loading experimental reranker..."
        )

        reranker = (
            BgeRerankerAdapter()
        )

        print(
            "Reranker:",
            reranker.model_name,
        )

        print(
            "Revision:",
            reranker.model_revision,
        )

        print(
            "Device:",
            reranker.device,
        )

        print()

        measurements = []

        for case_id in CASE_IDS:
            case = cases[
                case_id
            ]

            embedded = (
                embedding_provider
                .embed_query(
                    case["question"]
                )
            )

            candidates = (
                retrieval.search_authorized(
                    scope=scope,
                    query_vector=(
                        embedded.values
                    ),
                    model_name=(
                        embedded.model_name
                    ),
                    model_revision=(
                        embedded.model_revision
                    ),
                    top_k=CANDIDATE_K,
                )
            )

            start = (
                time.perf_counter()
            )

            reranked = reranker.rerank(
                query=case["question"],
                candidates=candidates,
                top_k=FINAL_K,
            )

            elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            top_rerank = (
                reranked[0].rerank_score
                if reranked
                else None
            )

            measurements.append(
                {
                    "id": case_id,
                    "label": case["label"],
                    "e5_top": (
                        candidates[0].score
                        if candidates
                        else None
                    ),
                    "rerank_top": (
                        top_rerank
                    ),
                    "latency_ms": (
                        elapsed_ms
                    ),
                }
            )

            print(
                "=" * 80
            )

            print(
                case_id,
                case["label"],
            )

            print(
                "Question:",
                case["question"],
            )

            print()
            print(
                "--- E5 / pgvector top 5 ---"
            )

            for rank, item in enumerate(
                candidates[:5],
                start=1,
            ):
                print(
                    f"{rank}. "
                    f"page={item.page_number} "
                    f"e5={item.score:.4f}"
                )

                print(
                    "   ",
                    snippet(
                        item.content
                    ),
                )

            print()
            print(
                "--- BGE reranked top 5 ---"
            )

            for rank, item in enumerate(
                reranked,
                start=1,
            ):
                chunk = item.chunk

                print(
                    f"{rank}. "
                    f"page={chunk.page_number} "
                    f"rerank={item.rerank_score:.4f} "
                    f"e5={chunk.score:.4f}"
                )

                print(
                    "   ",
                    snippet(
                        chunk.content
                    ),
                )

            print()
            print(
                "Reranking latency:",
                f"{elapsed_ms:.0f} ms",
            )
            print()

        answerable = [
            item["rerank_top"]
            for item in measurements
            if (
                item["label"]
                == "ANSWERABLE"
                and item["rerank_top"]
                is not None
            )
        ]

        unanswerable = [
            item["rerank_top"]
            for item in measurements
            if (
                item["label"]
                == "UNANSWERABLE"
                and item["rerank_top"]
                is not None
            )
        ]

        latencies = [
            item["latency_ms"]
            for item in measurements
        ]

        print(
            "=" * 80
        )

        print(
            "EXPERIMENT SUMMARY"
        )
        print()

        print(
            "Answerable reranker scores:"
        )

        print(
            " min=",
            f"{min(answerable):.4f}",
            " median=",
            f"{statistics.median(answerable):.4f}",
            " max=",
            f"{max(answerable):.4f}",
            sep="",
        )

        print(
            "Unanswerable reranker scores:"
        )

        print(
            " min=",
            f"{min(unanswerable):.4f}",
            " median=",
            f"{statistics.median(unanswerable):.4f}",
            " max=",
            f"{max(unanswerable):.4f}",
            sep="",
        )

        margin = (
            min(answerable)
            - max(unanswerable)
        )

        print()
        print(
            "Reranker separation margin:",
            f"{margin:.4f}",
        )

        print(
            "Median rerank latency:",
            f"{statistics.median(latencies):.0f} ms",
        )

        print(
            "Approx p95 rerank latency:",
            f"{percentile_95(latencies):.0f} ms",
        )

        print()
        print(
            "NOTE: No production search path "
            "was modified by this experiment."
        )

    finally:
        db.rollback()
        db.close()


if __name__ == "__main__":
    main()
