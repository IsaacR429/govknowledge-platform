from __future__ import annotations

import json
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


def load_cases():
    cases = json.loads(
        FIXTURE.read_text(
            encoding="ascii"
        )
    )

    return {
        case["id"]: case
        for case in cases
    }


def local_fs13(full: str) -> str:
    parts = full.split(". ")

    if len(parts) != 2:
        raise RuntimeError(
            f"Unexpected FS13 premise structure: {len(parts)} sentences"
        )

    return parts[1]


def local_fs29(full: str) -> str:
    parts = full.split(". ")

    if len(parts) != 3:
        raise RuntimeError(
            f"Unexpected FS29 premise structure: {len(parts)} sentences"
        )

    return parts[2]


def score(
    model,
    tokenizer,
    premise,
    claim,
):
    encoded = tokenizer(
        premise,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    started = time.perf_counter()

    with torch.inference_mode():
        logits = model(
            **encoded
        ).logits[0]

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

    return {
        "entailment": probabilities[0],
        "neutral": probabilities[1],
        "contradiction": probabilities[2],
        "latency_ms": latency_ms,
    }


def show(
    label,
    result,
):
    print()
    print(label)

    print(
        "P(entailment):",
        f"{result['entailment']:.4f}",
    )

    print(
        "P(neutral):",
        f"{result['neutral']:.4f}",
    )

    print(
        "P(contradiction):",
        f"{result['contradiction']:.4f}",
    )

    print(
        "Latency:",
        f"{result['latency_ms']:.0f} ms",
    )


def main():
    print()
    print(
        "=== Premise locality diagnostic v2 ==="
    )

    cases = load_cases()

    fs13 = cases["FS13"]
    fs29 = cases["FS29"]

    # Positive controls come from the same validated fixture.
    fs13_control = cases["FS10"]

    # FS29 positive control is generated from the actual
    # decisive sentence itself rather than duplicated French.
    fs29_local = local_fs29(
        fs29["premise"]
    )

    fs29_control_claim = (
        fs29_local
        .removeprefix(
            "En cas de diff\u00e9rences, "
        )
    )

    tests = [
        {
            "id": "FS13",
            "full": fs13["premise"],
            "local": local_fs13(
                fs13["premise"]
            ),
            "claim": fs13["claim"],
            "expected": False,
        },
        {
            "id": "FS13_CONTROL",
            "full": fs13["premise"],
            "local": local_fs13(
                fs13["premise"]
            ),
            "claim": fs13_control["claim"],
            "expected": True,
        },
        {
            "id": "FS29",
            "full": fs29["premise"],
            "local": fs29_local,
            "claim": fs29["claim"],
            "expected": False,
        },
        {
            "id": "FS29_CONTROL",
            "full": fs29["premise"],
            "local": fs29_local,
            "claim": fs29_control_claim,
            "expected": True,
        },
    ]

    combined = "".join(
        test["full"]
        + test["local"]
        + test["claim"]
        for test in tests
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

    for test in tests:
        print()
        print(
            "=" * 100
        )

        print(
            test["id"],
            "expected_supported=",
            test["expected"],
        )

        print()
        print(
            "FULL PREMISE:"
        )

        print(
            test["full"]
        )

        print()
        print(
            "LOCAL PREMISE:"
        )

        print(
            test["local"]
        )

        print()
        print(
            "CLAIM:"
        )

        print(
            test["claim"]
        )

        full_result = score(
            model,
            tokenizer,
            test["full"],
            test["claim"],
        )

        local_result = score(
            model,
            tokenizer,
            test["local"],
            test["claim"],
        )

        show(
            "FULL",
            full_result,
        )

        show(
            "LOCAL",
            local_result,
        )

        print()
        print(
            "Delta P(entailment):",
            f"{local_result['entailment'] - full_result['entailment']:+.4f}",
        )

    print()
    print(
        "=" * 100
    )

    print(
        "Validity requirements:"
    )

    print(
        "1. FS13 FULL should reproduce the stress benchmark."
    )

    print(
        "2. FS29 FULL should reproduce the stress benchmark."
    )

    print(
        "3. No literal '?' corruption."
    )

    print(
        "4. Only premise locality changes."
    )

    print()
    print(
        "No threshold changed."
    )

    print(
        "No model changed."
    )

    print(
        "No production architecture changed."
    )


if __name__ == "__main__":
    main()
