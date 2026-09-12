from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
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
    Path(__file__).resolve().parents[1] / "fixtures" / "native_french_entailment_stress_v2.json"
)

LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}


def main() -> None:
    print()
    print(
        "=== Native French entailment stress v2 ==="
    )

    cases = json.loads(
        FIXTURE.read_text(
            encoding="ascii"
        )
    )

    combined = "".join(
        case["premise"] + case["claim"]
        for case in cases
    )

    if (
        "?" in combined
        or "\ufffd" in combined
    ):
        raise RuntimeError(
            "Unicode integrity check failed."
        )

    print(
        "Unicode integrity: PASS"
    )

    print(
        "Cases:",
        len(cases),
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

    tp = fp = tn = fn = 0

    false_entailments = []
    false_refusals = []

    supported_scores = []
    negative_scores = []
    latencies = []

    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "false_entailments": 0,
            "false_refusals": 0,
        }
    )

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

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
            .cpu()
            .tolist()
        )

        predicted_id = max(
            range(3),
            key=lambda index: (
                probabilities[index]
            ),
        )

        predicted = LABELS[
            predicted_id
        ]

        predicted_supported = (
            predicted == "entailment"
        )

        expected_supported = (
            case["expected"]
            == "SUPPORTED"
        )

        correct = (
            predicted_supported
            == expected_supported
        )

        p_entailment = (
            probabilities[0]
        )

        if expected_supported:
            supported_scores.append(
                p_entailment
            )
        else:
            negative_scores.append(
                p_entailment
            )

        if (
            expected_supported
            and predicted_supported
        ):
            tp += 1

        elif (
            not expected_supported
            and predicted_supported
        ):
            fp += 1

            false_entailments.append(
                case["id"]
            )

        elif (
            not expected_supported
            and not predicted_supported
        ):
            tn += 1

        else:
            fn += 1

            false_refusals.append(
                case["id"]
            )

        category = case[
            "category"
        ]

        stat = category_stats[
            category
        ]

        stat["total"] += 1
        stat["correct"] += int(
            correct
        )

        if (
            predicted_supported
            and not expected_supported
        ):
            stat[
                "false_entailments"
            ] += 1

        if (
            expected_supported
            and not predicted_supported
        ):
            stat[
                "false_refusals"
            ] += 1

        latencies.append(
            elapsed_ms
        )

        rows.append(
            (
                case,
                probabilities,
                predicted,
                correct,
                elapsed_ms,
            )
        )

    for (
        case,
        probabilities,
        predicted,
        correct,
        elapsed_ms,
    ) in rows:
        print()
        print(
            "=" * 105
        )

        print(
            case["id"],
            case["expected"],
            f"[{case['category']}]",
        )

        print(
            "Claim:",
            case["claim"],
        )

        print(
            "P(entailment):",
            f"{probabilities[0]:.4f}",
        )

        print(
            "P(neutral):",
            f"{probabilities[1]:.4f}",
        )

        print(
            "P(contradiction):",
            f"{probabilities[2]:.4f}",
        )

        print(
            "Predicted:",
            predicted,
        )

        print(
            "Binary correct:",
            correct,
        )

        print(
            "Latency:",
            f"{elapsed_ms:.0f} ms",
        )

    total = len(cases)

    accuracy = (
        (tp + tn)
        / total
    )

    supported_min = min(
        supported_scores
    )

    supported_median = (
        statistics.median(
            supported_scores
        )
    )

    negative_max = max(
        negative_scores
    )

    negative_median = (
        statistics.median(
            negative_scores
        )
    )

    margin = (
        supported_min
        - negative_max
    )

    print()
    print(
        "=" * 105
    )

    print(
        "BINARY SUMMARY"
    )

    print(
        "TP:",
        tp,
    )

    print(
        "FP:",
        fp,
    )

    print(
        "TN:",
        tn,
    )

    print(
        "FN:",
        fn,
    )

    print(
        "Accuracy:",
        f"{accuracy:.3f}",
    )

    print()
    print(
        "FALSE ENTAILMENTS:"
    )

    print(
        false_entailments
        if false_entailments
        else "None"
    )

    print()
    print(
        "FALSE REFUSALS:"
    )

    print(
        false_refusals
        if false_refusals
        else "None"
    )

    print()
    print(
        "ENTAILMENT SCORE SEPARATION"
    )

    print(
        "Supported min:",
        f"{supported_min:.4f}",
    )

    print(
        "Supported median:",
        f"{supported_median:.4f}",
    )

    print(
        "Negative max:",
        f"{negative_max:.4f}",
    )

    print(
        "Negative median:",
        f"{negative_median:.4f}",
    )

    print(
        "Observed margin:",
        f"{margin:.4f}",
    )

    print()
    print(
        "CATEGORY SUMMARY"
    )

    for category in sorted(
        category_stats
    ):
        stat = category_stats[
            category
        ]

        print(
            f"{category:24s} "
            f"{stat['correct']}/"
            f"{stat['total']} "
            f"FP={stat['false_entailments']} "
            f"FN={stat['false_refusals']}"
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
        "p95:",
        f"{sorted(latencies)[int(0.95 * (len(latencies) - 1))]:.0f} ms",
    )

    print(
        "Max:",
        f"{max(latencies):.0f} ms",
    )

    print()
    print(
        "Native French premise and claim."
    )

    print(
        "No translation."
    )

    print(
        "No retrieval."
    )

    print(
        "No threshold selected."
    )

    print(
        "No production verifier promoted."
    )


if __name__ == "__main__":
    main()
