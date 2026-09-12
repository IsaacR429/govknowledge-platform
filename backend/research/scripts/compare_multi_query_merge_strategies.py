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

CANDIDATE_K = 10
RRF_K = 60


SUBQUERIES = {
    "M01": [
        "Quel organe recommande l'admission d'un Etat aux Nations Unies ?",
        "Quel organe prend la decision finale d'admission d'un Etat aux Nations Unies ?",
    ],
    "M02": [
        "Quel role joue l'Assemblee generale dans la nomination du Secretaire general ?",
        "Quel role joue le Conseil de securite dans la nomination du Secretaire general ?",
    ],
    "M03": [
        "Comment un Etat est-il admis comme membre des Nations Unies ?",
        "Comment un Etat membre peut-il etre expulse des Nations Unies ?",
    ],
    "M04": [
        "Quelle est la responsabilite principale du Conseil de securite ?",
        "Quel droit de defense individuelle ou collective la Charte preserve-t-elle ?",
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
        ["query: " + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    scores = vectors @ query_vector

    return np.argsort(-scores), scores


def rrf_merge(rankings):
    fused = defaultdict(float)

    for ranking in rankings:
        for rank, index in enumerate(
            ranking[:CANDIDATE_K],
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
    ]


def best_rank_merge(rankings):
    best = {}

    for ranking in rankings:
        for rank, index in enumerate(
            ranking[:CANDIDATE_K],
            start=1,
        ):
            index = int(index)

            if (
                index not in best
                or rank < best[index]
            ):
                best[index] = rank

    return [
        index
        for index, _ in sorted(
            best.items(),
            key=lambda item: item[1],
        )
    ]


def round_robin_merge(rankings):
    merged = []
    seen = set()

    for position in range(
        CANDIDATE_K
    ):
        for ranking in rankings:
            index = int(
                ranking[position]
            )

            if index in seen:
                continue

            seen.add(index)
            merged.append(index)

    return merged


def rank_of(
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


def coverage_at(
    *,
    ranking,
    chunks,
    anchors,
    k: int,
):
    top = ranking[:k]

    return all(
        any(
            contains(
                chunks[int(index)].content,
                anchor,
            )
            for index in top
        )
        for anchor in anchors.values()
    )


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
        "=== Multi-query merge strategy benchmark ==="
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

    totals = {
        "rrf": {
            3: 0,
            5: 0,
        },
        "best_rank": {
            3: 0,
            5: 0,
        },
        "round_robin": {
            3: 0,
            5: 0,
        },
    }

    for case_id in (
        "M01",
        "M02",
        "M03",
        "M04",
    ):
        print()
        print(
            "=" * 96
        )

        print(
            case_id,
            cases[case_id]["question"],
        )

        rankings = []

        for query in SUBQUERIES[
            case_id
        ]:
            ranking, _ = retrieve(
                model=model,
                vectors=vectors,
                query=query,
            )

            rankings.append(
                ranking
            )

        strategies = {
            "rrf": rrf_merge(
                rankings
            ),
            "best_rank": (
                best_rank_merge(
                    rankings
                )
            ),
            "round_robin": (
                round_robin_merge(
                    rankings
                )
            ),
        }

        anchors = GROUND_TRUTH[
            case_id
        ]

        for name, ranking in (
            strategies.items()
        ):
            print()
            print(
                name
            )

            for (
                evidence_name,
                anchor,
            ) in anchors.items():
                rank = rank_of(
                    ranking=ranking,
                    chunks=chunks,
                    anchor=anchor,
                )

                print(
                    f"  {evidence_name:<20} "
                    f"rank={rank}"
                )

            for k in (
                3,
                5,
            ):
                covered = coverage_at(
                    ranking=ranking,
                    chunks=chunks,
                    anchors=anchors,
                    k=k,
                )

                totals[name][k] += int(
                    covered
                )

                print(
                    f"  complete@{k}: "
                    f"{covered}"
                )

            print(
                "  top5:"
            )

            for rank, index in enumerate(
                ranking[:5],
                start=1,
            ):
                chunk = chunks[
                    int(index)
                ]

                print(
                    f"    #{rank} "
                    f"page={chunk.page_number} "
                    f"article={chunk.article}"
                )

    print()
    print(
        "=" * 96
    )

    print(
        "COVERAGE SUMMARY"
    )

    for name, values in (
        totals.items()
    ):
        print(
            f"{name:<15} "
            f"complete@3="
            f"{values[3]}/4 "
            f"complete@5="
            f"{values[5]}/4"
        )

    print()
    print(
        "No production merge strategy was selected."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
