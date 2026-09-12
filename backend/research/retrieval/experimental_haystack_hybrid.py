from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from haystack import Document
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import (
    InMemoryDocumentStore,
)


@dataclass(frozen=True)
class HybridResult:
    chunk_index: int
    content: str
    page_number: int
    article: str | None
    fusion_score: float
    semantic_rank: int | None
    lexical_rank: int | None


class ExperimentalHaystackHybridConnector:
    """Detachable Haystack hybrid retrieval connector.

    SECURITY:
    This connector must only receive chunks that have
    already been authorized by GovKnowledge.
    """

    def __init__(
        self,
        *,
        chunks,
        embeddings,
        semantic_weight: float = 0.8,
        lexical_weight: float = 0.2,
        candidate_k: int = 20,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunk and embedding counts differ."
            )

        if candidate_k < 1:
            raise ValueError(
                "candidate_k must be positive."
            )

        if semantic_weight < 0:
            raise ValueError(
                "semantic_weight cannot be negative."
            )

        if lexical_weight < 0:
            raise ValueError(
                "lexical_weight cannot be negative."
            )

        if (
            semantic_weight
            + lexical_weight
            <= 0
        ):
            raise ValueError(
                "At least one fusion weight "
                "must be positive."
            )

        self._candidate_k = candidate_k

        self._store = (
            InMemoryDocumentStore(
                embedding_similarity_function=(
                    "cosine"
                )
            )
        )

        documents = []

        for index, (
            chunk,
            embedding,
        ) in enumerate(
            zip(
                chunks,
                embeddings,
                strict=True,
            )
        ):
            documents.append(
                Document(
                    id=(
                        f"authorized-"
                        f"experimental-{index}"
                    ),
                    content=chunk.content,
                    embedding=(
                        embedding.tolist()
                    ),
                    meta={
                        "chunk_index": index,
                        "page_number": (
                            chunk.page_number
                        ),
                        "article": (
                            chunk.article
                        ),
                    },
                )
            )

        self._store.write_documents(
            documents
        )

        self._semantic = (
            InMemoryEmbeddingRetriever(
                document_store=self._store
            )
        )

        self._lexical = (
            InMemoryBM25Retriever(
                document_store=self._store
            )
        )

        self._joiner = (
            DocumentJoiner(
                join_mode=(
                    "reciprocal_rank_fusion"
                ),
                weights=[
                    semantic_weight,
                    lexical_weight,
                ],
            )
        )

    def search(
        self,
        *,
        query: str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> tuple[HybridResult, ...]:
        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        semantic = self._semantic.run(
            query_embedding=list(
                query_embedding
            ),
            top_k=self._candidate_k,
        )["documents"]

        lexical = self._lexical.run(
            query=query,
            top_k=self._candidate_k,
        )["documents"]

        semantic_ranks = {
            item.id: rank
            for rank, item in enumerate(
                semantic,
                start=1,
            )
        }

        lexical_ranks = {
            item.id: rank
            for rank, item in enumerate(
                lexical,
                start=1,
            )
        }

        fused = self._joiner.run(
            documents=[
                semantic,
                lexical,
            ],
            top_k=top_k,
        )["documents"]

        results = []

        for item in fused:
            results.append(
                HybridResult(
                    chunk_index=int(
                        item.meta[
                            "chunk_index"
                        ]
                    ),
                    content=(
                        item.content or ""
                    ),
                    page_number=int(
                        item.meta[
                            "page_number"
                        ]
                    ),
                    article=(
                        item.meta.get(
                            "article"
                        )
                    ),
                    fusion_score=float(
                        item.score
                        if item.score
                        is not None
                        else 0.0
                    ),
                    semantic_rank=(
                        semantic_ranks.get(
                            item.id
                        )
                    ),
                    lexical_rank=(
                        lexical_ranks.get(
                            item.id
                        )
                    ),
                )
            )

        return tuple(results)
