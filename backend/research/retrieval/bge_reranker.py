from __future__ import annotations

import math
from typing import Sequence

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from research.ports.reranker import (
    RerankedChunk,
    RerankerPort,
)
from app.application.ports.retrieval import (
    RetrievedChunk,
)


DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_MODEL_REVISION = (
    "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)


class BgeRerankerAdapter(RerankerPort):
    """Experimental multilingual cross-encoder reranker.

    This adapter must receive only candidates that have
    already passed GovKnowledge authorization.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        model_revision: str = DEFAULT_MODEL_REVISION,
        device: str | None = None,
        batch_size: int = 4,
        max_length: int = 1024,
    ) -> None:
        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive."
            )

        if max_length < 1:
            raise ValueError(
                "max_length must be positive."
            )

        self._model_name = model_name
        self._model_revision = model_revision
        self._batch_size = batch_size
        self._max_length = max_length

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self._device = device

        self._tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
                revision=model_revision,
            )
        )

        self._model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name,
                revision=model_revision,
            )
        )

        self._model.to(
            self._device
        )

        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def device(self) -> str:
        return self._device

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> tuple[RerankedChunk, ...]:
        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        if not candidates:
            return ()

        scored: list[RerankedChunk] = []

        for start in range(
            0,
            len(candidates),
            self._batch_size,
        ):
            batch = candidates[
                start:
                start + self._batch_size
            ]

            queries = [
                query
                for _ in batch
            ]

            passages = [
                item.content
                for item in batch
            ]

            encoded = self._tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(
                    self._device
                )
                for key, value
                in encoded.items()
            }

            with torch.inference_mode():
                output = self._model(
                    **encoded
                )

            logits = (
                output.logits
                .reshape(-1)
                .detach()
                .float()
                .cpu()
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
                .tolist()
            )

            if len(probabilities) != len(batch):
                raise RuntimeError(
                    "Reranker result count mismatch."
                )

            for candidate, score in zip(
                batch,
                probabilities,
                strict=True,
            ):
                score = float(
                    score
                )

                if not math.isfinite(
                    score
                ):
                    raise RuntimeError(
                        "Reranker produced a "
                        "non-finite score."
                    )

                scored.append(
                    RerankedChunk(
                        chunk=candidate,
                        rerank_score=score,
                    )
                )

        scored.sort(
            key=lambda item: (
                item.rerank_score
            ),
            reverse=True,
        )

        return tuple(
            scored[:top_k]
        )
