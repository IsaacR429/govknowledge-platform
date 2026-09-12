from __future__ import annotations

import json
import re
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
    Path(__file__).resolve().parents[1] / "fixtures" / "multisentence_dependency_holdout_v1.json"
)


def split_sentences(text):
    return [
        value.strip()
        for value in re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )
        if value.strip()
    ]


def adjacent_pairs(sentences):
    return [
        sentences[index]
        + " "
        + sentences[index + 1]
        for index in range(
            len(sentences) - 1
        )
    ]


def score_batch(
    tokenizer,
    model,
    premises,
    claim,
):
    encoded = tokenizer(
        premises,
        [claim] * len(premises),
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

    elapsed = (
        time.perf_counter()
        - started
    ) * 1000

    probs = torch.softmax(
        logits,
        dim=-1,
    ).cpu().tolist()

    rows = []

    for premise, values in zip(
        premises,
        probs,
    ):
        predicted = max(
            range(3),
            key=lambda index: values[index],
        )

        rows.append(
            {
                "premise": premise,
                "entailment": values[0],
                "neutral": values[1],
                "contradiction": values[2],
                "supported": predicted == 0,
            }
        )

    return rows, elapsed


def evaluate(
    result,
    truth,
):
    if truth and result:
        return "TP"

    if not truth and result:
        return "FP"

    if not truth and not result:
        return "TN"

    return "FN"


def main():
    print()
    print(
        "=== Multi-sentence dependency holdout v1 ==="
    )

    cases = json.loads(
        FIXTURE.read_text(
            encoding="ascii"
        )
    )

    combined = "".join(
        case["context"]
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

    print("Unicode integrity: PASS")
    print("Cases:", len(cases))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=True,
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

    modes = (
        "FULL",
        "SINGLE_ANY",
        "PAIR_ANY",
        "STAGED",
    )

    metrics = {
        mode: defaultdict(int)
        for mode in modes
    }

    latency = {
        mode: []
        for mode in modes
    }

    errors = {
        mode: []
        for mode in modes
    }

    scope_stats = {
        mode: defaultdict(
            lambda: {
                "correct": 0,
                "total": 0,
            }
        )
        for mode in modes
    }

    interesting = []

    for case in cases:
        sentences = split_sentences(
            case["context"]
        )

        pairs = adjacent_pairs(
            sentences
        )

        truth = (
            case["expected"]
            == "SUPPORTED"
        )

        full_rows, full_ms = (
            score_batch(
                tokenizer,
                model,
                [case["context"]],
                case["claim"],
            )
        )

        single_rows, single_ms = (
            score_batch(
                tokenizer,
                model,
                sentences,
                case["claim"],
            )
        )

        pair_rows, pair_ms = (
            score_batch(
                tokenizer,
                model,
                pairs,
                case["claim"],
            )
        )

        full_supported = (
            full_rows[0]["supported"]
        )

        single_supported = any(
            row["supported"]
            for row in single_rows
        )

        pair_supported = any(
            row["supported"]
            for row in pair_rows
        )

        # Conservative fallback:
        # only expose the verifier to pairs
        # when no sentence independently entails.
        if single_supported:
            staged_supported = True
            staged_ms = single_ms
        else:
            staged_supported = (
                pair_supported
            )
            staged_ms = (
                single_ms
                + pair_ms
            )

        decisions = {
            "FULL": (
                full_supported,
                full_ms,
            ),
            "SINGLE_ANY": (
                single_supported,
                single_ms,
            ),
            "PAIR_ANY": (
                pair_supported,
                pair_ms,
            ),
            "STAGED": (
                staged_supported,
                staged_ms,
            ),
        }

        for mode, (
            predicted,
            elapsed,
        ) in decisions.items():
            outcome = evaluate(
                predicted,
                truth,
            )

            metrics[mode][
                outcome
            ] += 1

            latency[mode].append(
                elapsed
            )

            correct = (
                predicted == truth
            )

            scope = case[
                "support_scope"
            ]

            scope_stats[
                mode
            ][scope]["total"] += 1

            scope_stats[
                mode
            ][scope]["correct"] += int(
                correct
            )

            if not correct:
                errors[mode].append(
                    case["id"]
                )

        # Save pair-required cases and
        # all disagreements for inspection.
        decisions_only = {
            mode: value[0]
            for mode, value
            in decisions.items()
        }

        if (
            case["support_scope"]
            == "pair"
            or len(
                set(
                    decisions_only.values()
                )
            ) > 1
            or any(
                case["id"]
                in errors[mode]
                for mode in modes
            )
        ):
            interesting.append(
                {
                    "case": case,
                    "sentences": sentences,
                    "pairs": pairs,
                    "single_rows": single_rows,
                    "pair_rows": pair_rows,
                    "decisions": decisions_only,
                }
            )

    print()
    print(
        "=" * 110
    )

    print("SUMMARY")

    for mode in modes:
        m = metrics[mode]

        total = sum(
            m[key]
            for key in (
                "TP",
                "FP",
                "TN",
                "FN",
            )
        )

        accuracy = (
            m["TP"] + m["TN"]
        ) / total

        print()
        print(mode)

        print(
            "TP:",
            m["TP"],
            "FP:",
            m["FP"],
            "TN:",
            m["TN"],
            "FN:",
            m["FN"],
        )

        print(
            "Accuracy:",
            f"{accuracy:.3f}",
        )

        print(
            "Errors:",
            errors[mode]
            or "None",
        )

        print(
            "Median latency:",
            f"{statistics.median(latency[mode]):.0f} ms",
        )

        print(
            "p95 latency:",
            f"{sorted(latency[mode])[int(0.95 * (len(latency[mode]) - 1))]:.0f} ms",
        )

        print(
            "By required support scope:"
        )

        for scope in (
            "single",
            "pair",
            "none",
        ):
            stat = (
                scope_stats[
                    mode
                ][scope]
            )

            print(
                f"  {scope:8s} "
                f"{stat['correct']}/"
                f"{stat['total']}"
            )

    print()
    print(
        "=" * 110
    )

    print(
        "PAIR-DEPENDENCY / DISAGREEMENT DETAILS"
    )

    for item in interesting:
        case = item["case"]

        print()
        print(
            "-" * 110
        )

        print(
            case["id"],
            case["expected"],
            f"scope={case['support_scope']}",
            f"[{case['category']}]",
        )

        print(
            "Claim:",
            case["claim"],
        )

        print(
            "Decisions:",
            item["decisions"],
        )

        print()
        print(
            "Single-sentence scores:"
        )

        for index, row in enumerate(
            item["single_rows"],
            start=1,
        ):
            print(
                f"  S{index}: "
                f"ent={row['entailment']:.4f} "
                f"supported={row['supported']} "
                f"| {row['premise']}"
            )

        print()
        print(
            "Adjacent-pair scores:"
        )

        for index, row in enumerate(
            item["pair_rows"],
            start=1,
        ):
            print(
                f"  P{index}: "
                f"ent={row['entailment']:.4f} "
                f"supported={row['supported']} "
                f"| {row['premise']}"
            )

    print()
    print(
        "=" * 110
    )

    print(
        "Decision criterion:"
    )

    print(
        "STAGED is preferred only if it recovers "
        "pair-dependent support without materially "
        "increasing false entailments."
    )

    print()
    print(
        "No threshold selected."
    )

    print(
        "No production verifier promoted."
    )

    print(
        "No database rows modified."
    )


if __name__ == "__main__":
    main()
