from __future__ import annotations

import math
from typing import Sequence

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from research.ports.evidence_support_verifier import (
    EvidenceSupportScore,
)
from app.application.ports.retrieval import (
    RetrievedChunk,
)


class ExperimentalBgeEvidenceSupportVerifier:
    MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    MODEL_REVISION = (
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )

    def __init__(
        self,
        *,
        max_length: int = 1024,
        batch_size: int = 3,
    ) -> None:
        self._max_length = max_length
        self._batch_size = batch_size

        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self._tokenizer = (
            AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                revision=self.MODEL_REVISION,
            )
        )

        self._model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.MODEL_NAME,
                revision=self.MODEL_REVISION,
            )
            .to(self._device)
        )

        self._model.eval()

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @property
    def model_revision(self) -> str:
        return self.MODEL_REVISION

    @property
    def device(self) -> str:
        return self._device

    def score(
        self,
        *,
        query: str,
        evidence: Sequence[RetrievedChunk],
    ) -> tuple[EvidenceSupportScore, ...]:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not evidence:
            return ()

        results = []

        for start in range(
            0,
            len(evidence),
            self._batch_size,
        ):
            batch = evidence[
                start:
                start + self._batch_size
            ]

            pairs = [
                [
                    query,
                    item.content,
                ]
                for item in batch
            ]

            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(
                    self._device
                )
                for key, value
                in inputs.items()
            }

            with torch.no_grad():
                logits = (
                    self._model(
                        **inputs
                    )
                    .logits
                    .view(-1)
                )

            for item, logit in zip(
                batch,
                logits.tolist(),
                strict=True,
            ):
                score = (
                    1.0
                    / (
                        1.0
                        + math.exp(
                            -float(logit)
                        )
                    )
                )

                results.append(
                    EvidenceSupportScore(
                        chunk=item,
                        support_score=score,
                    )
                )

        return tuple(results)
