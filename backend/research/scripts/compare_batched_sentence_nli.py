from __future__ import annotations

import json
import re
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
    Path(__file__).resolve().parents[1] / "fixtures" / "native_french_entailment_stress_v2.json"
)


def split_sentences(
    text: str,
) -> list[str]:
    return [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )
        if part.strip()
    ]


def batch_score(
    *,
    tokenizer,
    model,
    sentences: list[str],
    claim: str,
):
    encoded = tokenizer(
        sentences,
        [claim] * len(sentences),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    started = time.perf_counter()

    with torch.inference_mode():
        logits = model(
            **encoded
        ).logits

    latency_ms = (
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

    rows = []

    for sentence, probs in zip(
        sentences,
        probabilities,
    ):
        predicted_id = max(
            range(3),
            key=lambda index: (
                probs[index]
            ),
        )

        rows.append(
            {
                "sentence": sentence,
                "entailment": probs[0],
                "neutral": probs[1],
                "contradiction": probs[2],
                "predicted_entailment": (
                    predicted_id == 0
                ),
            }
        )

    return rows, latency_ms


def main():
    print()
    print(
        "=== Batched sentence-level NLI v1 ==="
    )

    cases = json.loads(
        FIXTURE.read_text(
            encoding="ascii"
        )
    )

    combined = "".join(
        case["premise"]
        + case["claim"]
        for case in cases
    )

    if (
        "?" in combined
        or "\ufffd" in combined
    ):
        raise RuntimeError(
            "Unicode integrity failure."
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
    sentence_counts = []

    interesting = []

    for case in cases:
        sentences = split_sentences(
            case["premise"]
        )

        if not sentences:
            raise RuntimeError(
                "No sentences for "
                + case["id"]
            )

        rows, latency_ms = (
            batch_score(
                tokenizer=tokenizer,
                model=model,
                sentences=sentences,
                claim=case["claim"],
            )
        )

        latencies.append(
            latency_ms
        )

        sentence_counts.append(
            len(sentences)
        )

        # Binary policy:
        # the claim is supported if at least
        # one independent sentence is classified
        # as entailment.
        predicted_supported = any(
            row[
                "predicted_entailment"
            ]
            for row in rows
        )

        expected_supported = (
            case["expected"]
            == "SUPPORTED"
        )

        max_entailment = max(
            row["entailment"]
            for row in rows
        )

        if expected_supported:
            supported_scores.append(
                max_entailment
            )
        else:
            negative_scores.append(
                max_entailment
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

        if (
            case["id"]
            in {
                "FS13",
                "FS29",
            }
            or (
                predicted_supported
                != expected_supported
            )
        ):
            interesting.append(
                {
                    "case": case,
                    "rows": rows,
                    "expected": (
                        expected_supported
                    ),
                    "predicted": (
                        predicted_supported
                    ),
                    "latency": (
                        latency_ms
                    ),
                }
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
        f"{(tp + tn) / total:.3f}",
    )

    print()
    print(
        "False entailments:",
        (
            false_entailments
            or "None"
        ),
    )

    print(
        "False refusals:",
        (
            false_refusals
            or "None"
        ),
    )

    supported_min = min(
        supported_scores
    )

    negative_max = max(
        negative_scores
    )

    print()
    print(
        "Supported min max-P(ent):",
        f"{supported_min:.4f}",
    )

    print(
        "Negative max max-P(ent):",
        f"{negative_max:.4f}",
    )

    print(
        "Observed margin:",
        f"{supported_min - negative_max:.4f}",
    )

    print()
    print(
        "Sentence count median:",
        statistics.median(
            sentence_counts
        ),
    )

    print(
        "Sentence count max:",
        max(
            sentence_counts
        ),
    )

    print()
    print(
        "Batch NLI median latency:",
        f"{statistics.median(latencies):.0f} ms",
    )

    ordered = sorted(
        latencies
    )

    p95_index = int(
        0.95
        * (
            len(ordered) - 1
        )
    )

    print(
        "Batch NLI p95 latency:",
        f"{ordered[p95_index]:.0f} ms",
    )

    print(
        "Batch NLI max latency:",
        f"{max(latencies):.0f} ms",
    )

    print()
    print(
        "=" * 105
    )

    print(
        "CRITICAL CASES"
    )

    for item in interesting:
        case = item["case"]

        print()
        print(
            "-" * 105
        )

        print(
            case["id"],
            case["expected"],
            f"[{case['category']}]",
        )

        print(
            "Expected supported:",
            item["expected"],
        )

        print(
            "Predicted supported:",
            item["predicted"],
        )

        print(
            "Claim:",
            case["claim"],
        )

        for index, row in enumerate(
            item["rows"],
            start=1,
        ):
            print()
            print(
                f"Sentence {index}"
            )

            print(
                row["sentence"]
            )

            print(
                "P(ent):",
                f"{row['entailment']:.4f}",
            )

            print(
                "P(neutral):",
                f"{row['neutral']:.4f}",
            )

            print(
                "P(contradiction):",
                f"{row['contradiction']:.4f}",
            )

            print(
                "Entailed:",
                row[
                    "predicted_entailment"
                ],
            )

    print()
    print(
        "=" * 105
    )

    print(
        "Interpretation:"
    )

    print(
        "If sentence-batched NLI reaches "
        "13 TP / 0 FP / 35 TN / 0 FN, "
        "we can compare it directly with "
        "E5 TOP1."
    )

    print()
    print(
        "No threshold selected."
    )

    print(
        "No E5 span selector used."
    )

    print(
        "No production component introduced."
    )

    print(
        "No database rows modified."
    )


if __name__ == "__main__":
    main()
