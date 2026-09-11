from __future__ import annotations

import math
from typing import Sequence

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.application.ports.entailment_verifier import (
    EntailmentScore,
    EntailmentVerifierPort,
)
from app.domain.enums.evidence import (
    EntailmentLabel,
)


DEFAULT_MODEL_NAME = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-mnli-xnli"
)

DEFAULT_MODEL_REVISION = (
    "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)


class MDebertaEntailmentVerifier(
    EntailmentVerifierPort
):
    def __init__(
        self,
        *,
        model_name: str = (
            DEFAULT_MODEL_NAME
        ),
        model_revision: str = (
            DEFAULT_MODEL_REVISION
        ),
        local_files_only: bool = True,
        max_length: int = 512,
    ) -> None:
        if max_length < 1:
            raise ValueError(
                "max_length must be positive."
            )

        self._model_name = model_name
        self._model_revision = (
            model_revision
        )
        self._max_length = max_length

        self._tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_name,
                revision=model_revision,
                local_files_only=(
                    local_files_only
                ),
            )
        )

        self._model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name,
                revision=model_revision,
                local_files_only=(
                    local_files_only
                ),
            )
        )

        self._model.eval()

        id2label = {
            int(index): str(label)
            .strip()
            .lower()
            for index, label
            in self._model.config
            .id2label.items()
        }

        self._label_indices = {
            label: index
            for index, label
            in id2label.items()
        }

        required = {
            "entailment",
            "neutral",
            "contradiction",
        }

        missing = (
            required
            - set(
                self._label_indices
            )
        )

        if missing:
            raise RuntimeError(
                "NLI model does not expose "
                "the required labels: "
                + ", ".join(
                    sorted(missing)
                )
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_revision(self) -> str:
        return self._model_revision

    def score(
        self,
        *,
        claim: str,
        premises: Sequence[str],
    ) -> tuple[EntailmentScore, ...]:
        cleaned_claim = claim.strip()

        if not cleaned_claim:
            raise ValueError(
                "Claim cannot be empty."
            )

        cleaned_premises = tuple(
            premise.strip()
            for premise in premises
        )

        if any(
            not premise
            for premise
            in cleaned_premises
        ):
            raise ValueError(
                "Premises cannot be empty."
            )

        if not cleaned_premises:
            return ()

        encoded = self._tokenizer(
            cleaned_premises,
            [cleaned_claim]
            * len(cleaned_premises),
            padding=True,
            truncation=True,
            max_length=(
                self._max_length
            ),
            return_tensors="pt",
        )

        with torch.inference_mode():
            logits = self._model(
                **encoded
            ).logits

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
            .cpu()
            .tolist()
        )

        entailment_index = (
            self._label_indices[
                "entailment"
            ]
        )

        neutral_index = (
            self._label_indices[
                "neutral"
            ]
        )

        contradiction_index = (
            self._label_indices[
                "contradiction"
            ]
        )

        results: list[
            EntailmentScore
        ] = []

        for values in probabilities:
            entailment = float(
                values[
                    entailment_index
                ]
            )

            neutral = float(
                values[
                    neutral_index
                ]
            )

            contradiction = float(
                values[
                    contradiction_index
                ]
            )

            if not all(
                math.isfinite(value)
                for value in (
                    entailment,
                    neutral,
                    contradiction,
                )
            ):
                raise RuntimeError(
                    "NLI model returned a "
                    "non-finite probability."
                )

            label = max(
                (
                    (
                        EntailmentLabel
                        .ENTAILMENT,
                        entailment,
                    ),
                    (
                        EntailmentLabel
                        .NEUTRAL,
                        neutral,
                    ),
                    (
                        EntailmentLabel
                        .CONTRADICTION,
                        contradiction,
                    ),
                ),
                key=lambda item: (
                    item[1]
                ),
            )[0]

            results.append(
                EntailmentScore(
                    label=label,
                    entailment_probability=(
                        entailment
                    ),
                    neutral_probability=(
                        neutral
                    ),
                    contradiction_probability=(
                        contradiction
                    ),
                )
            )

        return tuple(
            results
        )
