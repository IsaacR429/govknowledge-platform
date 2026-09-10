import numpy as np
import pytest

from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def get_embedding_dimension(
        self,
    ):
        return 3

    def encode(
        self,
        texts,
        *,
        normalize_embeddings,
        convert_to_numpy,
        show_progress_bar,
    ):
        self.calls.append(
            {
                "texts": list(texts),
                "normalize_embeddings": (
                    normalize_embeddings
                ),
                "convert_to_numpy": (
                    convert_to_numpy
                ),
                "show_progress_bar": (
                    show_progress_bar
                ),
            }
        )

        return np.array(
            [
                [1.0, 0.0, 0.0]
                for _ in texts
            ],
            dtype=float,
        )


def test_query_uses_e5_query_prefix():
    model = FakeEmbeddingModel()

    provider = (
        SentenceTransformerEmbeddingProvider(
            model_name="test-model",
            model_revision="test-revision",
            model=model,
        )
    )

    result = provider.embed_query(
        "Quels sont les critères ?"
    )

    assert (
        model.calls[0]["texts"]
        == [
            "query: Quels sont les critères ?"
        ]
    )

    assert (
        model.calls[0][
            "normalize_embeddings"
        ]
        is True
    )

    assert result.dimension == 3
    assert result.model_name == "test-model"
    assert result.model_revision == "test-revision"
    assert len(result.values) == 3


def test_passages_use_e5_passage_prefix():
    model = FakeEmbeddingModel()

    provider = (
        SentenceTransformerEmbeddingProvider(
            model_name="test-model",
            model_revision="test-revision",
            model=model,
        )
    )

    results = provider.embed_passages(
        [
            "Article premier.",
            "Deuxième disposition.",
        ]
    )

    assert (
        model.calls[0]["texts"]
        == [
            "passage: Article premier.",
            "passage: Deuxième disposition.",
        ]
    )

    assert len(results) == 2

    assert all(
        result.dimension == 3
        for result in results
    )


def test_empty_query_is_rejected():
    provider = (
        SentenceTransformerEmbeddingProvider(
            model_name="test-model",
            model_revision="test-revision",
            model=FakeEmbeddingModel(),
        )
    )

    with pytest.raises(
        ValueError,
        match="Query text cannot be empty",
    ):
        provider.embed_query("   ")
