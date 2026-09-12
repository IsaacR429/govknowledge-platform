from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-mnli-xnli"
)

MODEL_REVISION = (
    "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "native_french_entailment_benchmark_v1.json"
)

LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


def expected_binary(
    expected: str,
) -> bool:
    return expected == "SUPPORTED"


def main() -> None:
    print()
    print(
        "=== Native French NLI benchmark v1 ==="
    )

    cases = json.loads(
        FIXTURE.read_text(
            encoding="utf-8"
        )
    )

    print(
        "Cases:",
        len(cases),
    )

    print(
        "Loading cached mDeBERTa..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )

    model.eval()

    binary_correct = 0
    exact_correct = 0
    false_entailments = []
    false_refusals = []
    latencies = []

    expected_to_nli = {
        "SUPPORTED": "entailment",
        "CONTRADICTED": "contradiction",
        "OVERCLAIM": "neutral",
        "UNSUPPORTED": "neutral",
    }

    rows = []

    for case in cases:
        encoded = tokenizer(
            case["premise"],
            case["claim"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        started = time.perf_counter()

        with torch.inference_mode():
            logits = model(
                **encoded
            ).logits[0]

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000

        latencies.append(
            elapsed_ms
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
            .cpu()
            .tolist()
        )

        predicted_id = max(
            range(
                len(probabilities)
            ),
            key=lambda index: (
                probabilities[index]
            ),
        )

        predicted = LABELS[
            predicted_id
        ]

        expected = case[
            "expected"
        ]

        binary_prediction = (
            predicted == "entailment"
        )

        binary_truth = (
            expected_binary(
                expected
            )
        )

        binary_ok = (
            binary_prediction
            == binary_truth
        )

        exact_ok = (
            predicted
            == expected_to_nli[
                expected
            ]
        )

        binary_correct += int(
            binary_ok
        )

        exact_correct += int(
            exact_ok
        )

        if (
            binary_prediction
            and not binary_truth
        ):
            false_entailments.append(
                case["id"]
            )

        if (
            not binary_prediction
            and binary_truth
        ):
            false_refusals.append(
                case["id"]
            )

        rows.append(
            (
                case,
                probabilities,
                predicted,
                binary_ok,
                exact_ok,
                elapsed_ms,
            )
        )

    for (
        case,
        probabilities,
        predicted,
        binary_ok,
        exact_ok,
        elapsed_ms,
    ) in rows:
        print()
        print(
            "=" * 105
        )

        print(
            case["id"],
            case["expected"],
        )

        print(
            "Premise:"
        )

        print(
            " ",
            case["premise"],
        )

        print(
            "Claim:"
        )

        print(
            " ",
            case["claim"],
        )

        print(
            "NLI:"
        )

        for index in range(3):
            print(
                f"  {LABELS[index]:13s} "
                f"{probabilities[index]:.4f}"
            )

        print(
            "Predicted:",
            predicted,
        )

        print(
            "Binary correct:",
            binary_ok,
        )

        print(
            "Exact 3-class correct:",
            exact_ok,
        )

        print(
            "Latency:",
            f"{elapsed_ms:.0f} ms",
        )

    total = len(cases)

    print()
    print(
        "=" * 105
    )

    print(
        "SUMMARY"
    )

    print(
        "Binary supported/not-supported:",
        f"{binary_correct}/{total}",
        f"({binary_correct / total:.3f})",
    )

    print(
        "Exact NLI 3-class:",
        f"{exact_correct}/{total}",
        f"({exact_correct / total:.3f})",
    )

    print()
    print(
        "FALSE ENTAILMENTS"
    )

    print(
        false_entailments
        if false_entailments
        else "None"
    )

    print()
    print(
        "FALSE REFUSALS"
    )

    print(
        false_refusals
        if false_refusals
        else "None"
    )

    print()
    print(
        "LATENCY"
    )

    print(
        "Median:",
        f"{statistics.median(latencies):.0f} ms",
    )

    print(
        "Max:",
        f"{max(latencies):.0f} ms",
    )

    print()
    print(
        "Evidence and claims were both French."
    )

    print(
        "No translation was performed."
    )

    print(
        "No retrieval was performed."
    )

    print(
        "No verifier was promoted."
    )

    print(
        "No production threshold was selected."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
