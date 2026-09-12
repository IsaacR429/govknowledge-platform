from __future__ import annotations

import re
import time
import unicodedata
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
    / "world-bank-procurement-fr.pdf"
)

TOP_K = 5


CASES = [
    {
        "id": "FR01",
        "question": (
            "Qui est responsable de la conduite des activites "
            "de passation des marches financees par la Banque ?"
        ),
        "anchors": [
            "l emprunteur est responsable de la conduite",
            "activites de passation des marches",
        ],
    },
    {
        "id": "FR02",
        "question": (
            "La Banque peut-elle executer les operations "
            "de passation des marches a la place de l'Emprunteur ?"
        ),
        "anchors": [
            "reste responsable de l execution du projet",
            "en lieu et place de l emprunteur",
        ],
    },
    {
        "id": "FR03",
        "question": (
            "Quelles conditions doivent respecter les systemes "
            "electroniques utilises pour la passation des marches ?"
        ),
        "anchors": [
            "accessibilite",
            "securite",
            "integrite",
            "confidentialite",
        ],
    },
    {
        "id": "FR04",
        "question": (
            "Dans quelles langues les dossiers de passation "
            "des marches en concurrence internationale "
            "peuvent-ils etre etablis ?"
        ),
        "anchors": [
            "anglais",
            "espagnol",
            "francais",
        ],
    },
    {
        "id": "FR05",
        "question": (
            "Que se passe-t-il lorsqu'une offre ou proposition "
            "est recue apres le delai fixe ?"
        ),
        "anchors": [
            "recues hors delai sont rejetees",
        ],
    },
    {
        "id": "FR06",
        "question": (
            "Qui examine et approuve la strategie de passation "
            "des marches du projet avant la fin des negociations ?"
        ),
        "anchors": [
            "la banque examine et approuve",
            "spmpd",
        ],
    },
]


def normalize(value: str) -> str:
    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(
            character
        )
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
        r"[^a-zA-Z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip().lower()


def anchor_hits(
    content: str,
    anchors: list[str],
) -> int:
    normalized = normalize(
        content
    )

    return sum(
        normalize(anchor) in normalized
        for anchor in anchors
    )


def main() -> None:
    print()
    print(
        "=== Native French corpus retrieval probe ==="
    )

    print(
        "Source:",
        SOURCE,
    )

    if not SOURCE.exists():
        raise FileNotFoundError(
            SOURCE
        )

    started = time.perf_counter()

    result = (
        DocumentConverter()
        .convert(
            SOURCE,
            raises_on_error=True,
        )
    )

    parse_seconds = (
        time.perf_counter()
        - started
    )

    print()
    print(
        "Docling parse:",
        f"{parse_seconds:.1f}s",
    )

    chunks = (
        ExperimentalStructureChunker(
            max_chars=2200,
        )
        .chunk(
            result.document
        )
    )

    lengths = np.array(
        [
            len(chunk.content)
            for chunk in chunks
        ]
    )

    pages = {
        chunk.page_number
        for chunk in chunks
    }

    print(
        "Structure chunks:",
        len(chunks),
    )

    print(
        "Pages represented:",
        len(pages),
    )

    print(
        "Chunk length:",
        f"min={lengths.min()}",
        f"median={int(np.median(lengths))}",
        f"max={lengths.max()}",
    )

    print(
        "Empty chunks:",
        sum(
            not chunk.content.strip()
            for chunk in chunks
        ),
    )

    print()
    print(
        "Loading pinned multilingual E5..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    print(
        "Embedding French corpus..."
    )

    started = time.perf_counter()

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

    embedding_seconds = (
        time.perf_counter()
        - started
    )

    print(
        "Corpus embedding:",
        f"{embedding_seconds:.1f}s",
    )

    successes = 0

    for case in CASES:
        query_vector = model.encode(
            [
                "query: "
                + case["question"]
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

        ground_truth_rank = None
        ground_truth_hits = 0

        for rank, index in enumerate(
            order,
            start=1,
        ):
            hits = anchor_hits(
                chunks[
                    int(index)
                ].content,
                case["anchors"],
            )

            if hits > ground_truth_hits:
                ground_truth_hits = hits

            if hits == len(
                case["anchors"]
            ):
                ground_truth_rank = rank
                break

        success = (
            ground_truth_rank is not None
            and ground_truth_rank <= TOP_K
        )

        successes += int(
            success
        )

        print()
        print(
            "=" * 100
        )

        print(
            case["id"],
            case["question"],
        )

        print(
            "Ground-truth rank:",
            ground_truth_rank,
        )

        print(
            f"Recall@{TOP_K}:",
            success,
        )

        print()
        print(
            f"Top {TOP_K}:"
        )

        for rank, index in enumerate(
            order[:TOP_K],
            start=1,
        ):
            chunk = chunks[
                int(index)
            ]

            hits = anchor_hits(
                chunk.content,
                case["anchors"],
            )

            snippet = " ".join(
                chunk.content.split()
            )[:500]

            print()
            print(
                f"#{rank} "
                f"score={scores[int(index)]:.4f} "
                f"page={chunk.page_number} "
                f"article={chunk.article} "
                f"structure={chunk.structural_label} "
                f"anchor_hits={hits}/"
                f"{len(case['anchors'])}"
            )

            print(
                snippet
            )

    print()
    print(
        "=" * 100
    )

    print(
        "SUMMARY"
    )

    print(
        f"French retrieval Recall@{TOP_K}:",
        f"{successes}/{len(CASES)}",
        f"({successes / len(CASES):.3f})",
    )

    print(
        "Document language:",
        "French source corpus",
    )

    print(
        "Embedding model:",
        MODEL_NAME,
    )

    print(
        "Revision:",
        MODEL_REVISION,
    )

    print()
    print(
        "No translation was performed."
    )

    print(
        "No NLI verifier was evaluated."
    )

    print(
        "No database rows were modified."
    )

    print(
        "No production chunker was changed."
    )


if __name__ == "__main__":
    main()
