from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from uuid import UUID

from app.application.ports.authorization import AuthorizationScope
from app.domain.enums.permission import PermissionAction
from app.infrastructure.database.models.documents import DocumentVersion
from app.infrastructure.database.session import SessionLocal
from research.retrieval.bge_reranker import BgeRerankerAdapter
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

CASE_CONFIG = {
    "A01": "primary responsibility",
    "A05": "shall be appointed by the General Assembly",
    "A06": "may be expelled from the Organization",
}

CANDIDATE_K = 50
RERANK_K = 20


def normalize(text: str) -> str:
    value = unicodedata.normalize(
        "NFKC",
        text,
    )

    value = value.replace(
        "\u00ad",
        "",
    )

    value = re.sub(
        r"-\s+",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip().lower()


def has_anchor(
    text: str,
    anchor: str,
) -> bool:
    return (
        normalize(anchor)
        in normalize(text)
    )


def evidence_context(
    text: str,
    anchor: str,
    radius: int = 300,
) -> str:
    cleaned = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    lowered = cleaned.lower()
    target = anchor.lower()

    index = lowered.find(
        target
    )

    if index < 0:
        return cleaned[:700]

    start = max(
        0,
        index - radius,
    )

    end = min(
        len(cleaned),
        index + len(anchor) + radius,
    )

    return cleaned[
        start:end
    ]


def main() -> None:
    dataset = {
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
            "Loading experimental BGE reranker..."
        )

        reranker = (
            BgeRerankerAdapter()
        )

        print()
        print(
            "=== Ground-truth retrieval + reranking ==="
        )

        for case_id, anchor in CASE_CONFIG.items():
            case = dataset[
                case_id
            ]

            question = case[
                "question"
            ]

            print()
            print(
                "=" * 80
            )

            print(
                case_id,
                question,
            )

            print(
                "Unicode-safe:",
                "?" not in question[:-2],
            )

            embedded = (
                embedding_provider
                .embed_query(
                    question
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

            ground_truth = None
            e5_rank = None

            for rank, candidate in enumerate(
                candidates,
                start=1,
            ):
                if has_anchor(
                    candidate.content,
                    anchor,
                ):
                    ground_truth = candidate
                    e5_rank = rank
                    break

            if ground_truth is None:
                print(
                    "Ground truth: NOT FOUND "
                    "in E5 top 50"
                )
                continue

            print(
                "Ground-truth E5 rank:",
                e5_rank,
            )

            print(
                "E5 score:",
                f"{ground_truth.score:.4f}",
            )

            print(
                "Page:",
                ground_truth.page_number,
            )

            print(
                "Available to top-20 reranker:",
                e5_rank <= RERANK_K,
            )

            print()
            print(
                "--- Exact answer-bearing context ---"
            )

            print(
                evidence_context(
                    ground_truth.content,
                    anchor,
                )
            )

            if e5_rank > RERANK_K:
                print()
                print(
                    "BGE cannot recover this result "
                    "because it is outside top 20."
                )
                continue

            start = (
                time.perf_counter()
            )

            reranked = reranker.rerank(
                query=question,
                candidates=(
                    candidates[:RERANK_K]
                ),
                top_k=RERANK_K,
            )

            elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            rerank_rank = None
            rerank_score = None

            for rank, result in enumerate(
                reranked,
                start=1,
            ):
                if (
                    result.chunk.chunk_id
                    == ground_truth.chunk_id
                ):
                    rerank_rank = rank
                    rerank_score = (
                        result.rerank_score
                    )
                    break

            print()
            print(
                "Ground-truth BGE rank:",
                rerank_rank,
            )

            print(
                "BGE score:",
                (
                    f"{rerank_score:.4f}"
                    if rerank_score is not None
                    else "NOT FOUND"
                ),
            )

            print(
                "Rank movement:",
                (
                    f"{e5_rank} -> {rerank_rank}"
                    if rerank_rank is not None
                    else "N/A"
                ),
            )

            print(
                "Rerank latency:",
                f"{elapsed_ms:.0f} ms",
            )

        print()
        print(
            "=" * 80
        )

        print(
            "No production retrieval path "
            "was modified."
        )

    finally:
        db.rollback()
        db.close()


if __name__ == "__main__":
    main()
