from dataclasses import dataclass

import numpy as np

from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    passages: tuple[str, ...]
    expected_index: int


CASES = (
    Case(
        name="Security Council responsibility",
        query=(
            "Quel organe a la responsabilité principale "
            "du maintien de la paix et de la sécurité "
            "internationales ?"
        ),
        passages=(
            (
                "In order to ensure prompt and effective action "
                "by the United Nations, its Members confer on "
                "the Security Council primary responsibility "
                "for the maintenance of international peace "
                "and security."
            ),
            (
                "The General Assembly shall consider and "
                "approve the budget of the Organization."
            ),
            (
                "Each Member of the General Assembly "
                "shall have one vote."
            ),
        ),
        expected_index=0,
    ),
    Case(
        name="UN membership",
        query=(
            "Comment un nouvel État peut-il être admis "
            "comme membre des Nations Unies ?"
        ),
        passages=(
            (
                "The Economic and Social Council may make "
                "recommendations for promoting respect for "
                "human rights."
            ),
            (
                "The admission of any such state to membership "
                "in the United Nations will be effected by a "
                "decision of the General Assembly upon the "
                "recommendation of the Security Council."
            ),
            (
                "The Security Council shall adopt its own "
                "rules of procedure."
            ),
        ),
        expected_index=1,
    ),
    Case(
        name="General Assembly voting",
        query=(
            "Combien de voix chaque membre de "
            "l'Assemblée générale possède-t-il ?"
        ),
        passages=(
            (
                "Each member of the General Assembly "
                "shall have one vote."
            ),
            (
                "The Security Council shall consist "
                "of eleven Members of the United Nations."
            ),
            (
                "The Secretary-General shall be appointed "
                "by the General Assembly upon the "
                "recommendation of the Security Council."
            ),
        ),
        expected_index=0,
    ),
)


def cosine_scores(
    query_vector: tuple[float, ...],
    passage_vectors: tuple[
        tuple[float, ...],
        ...,
    ],
) -> list[float]:
    query = np.asarray(
        query_vector,
        dtype=np.float32,
    )

    return [
        float(
            np.dot(
                query,
                np.asarray(
                    passage,
                    dtype=np.float32,
                ),
            )
        )
        for passage in passage_vectors
    ]


def main() -> None:
    print(
        "Loading multilingual-e5-base..."
    )

    provider = (
        SentenceTransformerEmbeddingProvider()
    )

    print(
        "Model:",
        provider.model_name,
    )
    print(
        "Dimension:",
        provider.dimension,
    )

    assert provider.dimension == 768

    passed = 0

    for case in CASES:
        query = provider.embed_query(
            case.query
        )

        passages = provider.embed_passages(
            case.passages
        )

        scores = cosine_scores(
            query.values,
            tuple(
                result.values
                for result in passages
            ),
        )

        ranking = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        print()
        print(
            f"=== {case.name} ==="
        )
        print(
            "French query:",
            case.query,
        )

        for rank, index in enumerate(
            ranking,
            start=1,
        ):
            marker = (
                "<-- expected"
                if index == case.expected_index
                else ""
            )

            print(
                f"{rank}. "
                f"passage={index} "
                f"score={scores[index]:.4f} "
                f"{marker}"
            )

        if ranking[0] != case.expected_index:
            raise AssertionError(
                f"{case.name}: expected passage "
                f"{case.expected_index} to rank first, "
                f"but got passage {ranking[0]}."
            )

        passed += 1

    print()
    print(
        "Cross-language cases passed:",
        f"{passed}/{len(CASES)}",
    )
    print(
        "French -> English retrieval:",
        "PASS",
    )


if __name__ == "__main__":
    main()
