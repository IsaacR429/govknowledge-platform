from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

from research.parsing.experimental_structure_chunker import (
    ExperimentalStructureChunker,
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

TOP_K_PER_QUERY = 10

RRF_K = 60


SUBQUERIES = {
    "M01": [
        (
            "Quel organe recommande l'admission "
            "d'un Etat aux Nations Unies ?"
        ),
        (
            "Quel organe prend la decision finale "
            "d'admission d'un Etat aux Nations Unies ?"
        ),
    ],
    "M02": [
        (
            "Quel role joue l'Assemblee generale "
            "dans la nomination du Secretaire general ?"
        ),
        (
            "Quel role joue le Conseil de securite "
            "dans la nomination du Secretaire general ?"
        ),
    ],
    "M03": [
        (
            "Comment un Etat est-il admis comme "
            "membre des Nations Unies ?"
        ),
        (
            "Comment un Etat membre peut-il etre "
            "expulse des Nations Unies ?"
        ),
    ],
    "M04": [
        (
            "Quelle est la responsabilite principale "
            "du Conseil de securite ?"
        ),
        (
            "Quel droit de defense individuelle ou "
            "collective la Charte preserve-t-elle ?"
        ),
    ],
}


GROUND_TRUTH = {
    "M01": {
        "admission": (
            "admission of any such state to membership "
            "in the united nations"
        ),
    },
    "M02": {
        "secretary_general": (
            "shall be appointed by the general assembly "
            "upon the recommendation of the security council"
        ),
    },
    "M03": {
        "admission": (
            "admission of any such state to membership "
            "in the united nations"
        ),
        "expulsion": (
            "may be expelled from the organization "
            "by the general assembly upon the recommendation "
            "of the security council"
        ),
    },
    "M04": {
        "security_council": (
            "primary responsibility for the maintenance "
            "of international peace and security"
        ),
        "self_defence": (
            "inherent right of individual or collective"
        ),
    },
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


def retrieve(
    *,
    model,
    vectors,
    query: str,
):
    query_vector = model.encode(
        [
            "query: " + query
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    scores = vectors @ query_vector

    return np.argsort(
        -scores
    ), scores


def reciprocal_rank_fusion(
    rankings,
):
    fused = defaultdict(float)

    for ranking in rankings:
        for rank, index in enumerate(
            ranking[:TOP_K_PER_QUERY],
            start=1,
        ):
            fused[int(index)] += (
                1.0
                / (
                    RRF_K
                    + rank
                )
            )

    return [
        index
        for index, _ in sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ], fused


def ground_truth_rank(
    *,
    ranking,
    chunks,
    anchor: str,
):
    for rank, index in enumerate(
        ranking,
        start=1,
    ):
        if contains(
            chunks[int(index)].content,
            anchor,
        ):
            return rank

    return None


def main() -> None:
    cases = {
        item["id"]: item
        for item in json.loads(
            DATASET.read_text(
                encoding="utf-8"
            )
        )
    }

    print()
    print(
        "=== Oracle multi-query retrieval experiment ==="
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

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    print(
        "Embedding corpus..."
    )

    vectors = model.encode(
        [
            "passage: " + chunk.content
            for chunk in chunks
        ],
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    for case_id in (
        "M01",
        "M02",
        "M03",
        "M04",
    ):
        print()
        print(
            "=" * 92
        )

        original_question = cases[
            case_id
        ]["question"]

        print(
            case_id,
            original_question,
        )

        original_order, _ = retrieve(
            model=model,
            vectors=vectors,
            query=original_question,
        )

        sub_rankings = []

        print()
        print(
            "Subqueries:"
        )

        for number, query in enumerate(
            SUBQUERIES[case_id],
            start=1,
        ):
            order, _ = retrieve(
                model=model,
                vectors=vectors,
                query=query,
            )

            sub_rankings.append(
                order
            )

            print(
                f"  Q{number}: {query}"
            )

        fused_order, fused_scores = (
            reciprocal_rank_fusion(
                sub_rankings
            )
        )

        print()
        print(
            "Required evidence:"
        )

        for name, anchor in (
            GROUND_TRUTH[
                case_id
            ].items()
        ):
            original_rank = (
                ground_truth_rank(
                    ranking=original_order,
                    chunks=chunks,
                    anchor=anchor,
                )
            )

            subquery_ranks = [
                ground_truth_rank(
                    ranking=ranking,
                    chunks=chunks,
                    anchor=anchor,
                )
                for ranking
                in sub_rankings
            ]

            fused_rank = (
                ground_truth_rank(
                    ranking=fused_order,
                    chunks=chunks,
                    anchor=anchor,
                )
            )

            print(
                f"  {name:<20} "
                f"original={original_rank} "
                f"subqueries={subquery_ranks} "
                f"fused={fused_rank}"
            )

        print()
        print(
            "Fused top 5:"
        )

        for rank, index in enumerate(
            fused_order[:5],
            start=1,
        ):
            chunk = chunks[
                int(index)
            ]

            snippet = " ".join(
                chunk.content.split()
            )[:300]

            print()
            print(
                f"#{rank} "
                f"rrf={fused_scores[int(index)]:.6f} "
                f"page={chunk.page_number} "
                f"article={chunk.article}"
            )

            print(
                snippet
            )

    print()
    print(
        "=" * 92
    )

    print(
        "This is an oracle decomposition experiment only."
    )

    print(
        "No production query decomposer was introduced."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
