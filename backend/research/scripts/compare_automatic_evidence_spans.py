from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


E5_NAME = "intfloat/multilingual-e5-base"

E5_REVISION = (
    "d128750597153bb5987e10b1c3493a34e5a4502a"
)

NLI_NAME = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-mnli-xnli"
)

NLI_REVISION = (
    "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "native_french_entailment_stress_v2.json"
)


def split_sentences(text: str) -> list[str]:
    parts = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def build_windows(
    sentences: list[str],
) -> list[str]:
    windows = list(sentences)

    for index in range(
        len(sentences) - 1
    ):
        windows.append(
            sentences[index]
            + " "
            + sentences[index + 1]
        )

    # Deduplicate while preserving order.
    return list(
        dict.fromkeys(windows)
    )


def select_span(
    *,
    e5,
    claim: str,
    candidates: list[str],
):
    query_vector = e5.encode(
        ["query: " + claim],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    vectors = e5.encode(
        [
            "passage: " + candidate
            for candidate in candidates
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    scores = (
        vectors
        @ query_vector
    )

    best_index = int(
        np.argmax(scores)
    )

    return (
        candidates[best_index],
        float(scores[best_index]),
    )


def nli_score(
    *,
    tokenizer,
    model,
    premise: str,
    claim: str,
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

    return {
        "entailment": probabilities[0],
        "neutral": probabilities[1],
        "contradiction": probabilities[2],
        "supported": (
            int(
                np.argmax(
                    probabilities
                )
            )
            == 0
        ),
        "latency_ms": elapsed_ms,
    }


def empty_metrics():
    return {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "false_entailments": [],
        "false_refusals": [],
        "supported_scores": [],
        "negative_scores": [],
        "latencies": [],
    }


def update_metrics(
    metrics,
    case,
    result,
):
    truth = (
        case["expected"]
        == "SUPPORTED"
    )

    predicted = result[
        "supported"
    ]

    if truth:
        metrics[
            "supported_scores"
        ].append(
            result["entailment"]
        )
    else:
        metrics[
            "negative_scores"
        ].append(
            result["entailment"]
        )

    metrics[
        "latencies"
    ].append(
        result["latency_ms"]
    )

    if truth and predicted:
        metrics["tp"] += 1

    elif (
        not truth
        and predicted
    ):
        metrics["fp"] += 1
        metrics[
            "false_entailments"
        ].append(
            case["id"]
        )

    elif (
        not truth
        and not predicted
    ):
        metrics["tn"] += 1

    else:
        metrics["fn"] += 1
        metrics[
            "false_refusals"
        ].append(
            case["id"]
        )


def show_summary(
    name,
    metrics,
):
    total = (
        metrics["tp"]
        + metrics["fp"]
        + metrics["tn"]
        + metrics["fn"]
    )

    accuracy = (
        metrics["tp"]
        + metrics["tn"]
    ) / total

    print()
    print(
        "=" * 105
    )

    print(name)

    print(
        "TP:",
        metrics["tp"],
        "FP:",
        metrics["fp"],
        "TN:",
        metrics["tn"],
        "FN:",
        metrics["fn"],
    )

    print(
        "Accuracy:",
        f"{accuracy:.3f}",
    )

    print(
        "False entailments:",
        (
            metrics[
                "false_entailments"
            ]
            or "None"
        ),
    )

    print(
        "False refusals:",
        (
            metrics[
                "false_refusals"
            ]
            or "None"
        ),
    )

    supported_min = min(
        metrics[
            "supported_scores"
        ]
    )

    negative_max = max(
        metrics[
            "negative_scores"
        ]
    )

    print(
        "Supported min P(ent):",
        f"{supported_min:.4f}",
    )

    print(
        "Negative max P(ent):",
        f"{negative_max:.4f}",
    )

    print(
        "Observed margin:",
        f"{supported_min - negative_max:.4f}",
    )

    print(
        "NLI median latency:",
        f"{statistics.median(metrics['latencies']):.0f} ms",
    )


def main():
    print()
    print(
        "=== Automatic evidence-span selection v1 ==="
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

    print()
    print(
        "Loading pinned E5..."
    )

    e5 = SentenceTransformer(
        E5_NAME,
        revision=E5_REVISION,
    )

    print(
        "Loading cached mDeBERTa..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            NLI_NAME,
            revision=NLI_REVISION,
            local_files_only=True,
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            NLI_NAME,
            revision=NLI_REVISION,
            local_files_only=True,
        )
    )

    model.eval()

    metrics = {
        "FULL": empty_metrics(),
        "TOP1": empty_metrics(),
        "WINDOW": empty_metrics(),
    }

    details = []

    for case in cases:
        sentences = split_sentences(
            case["premise"]
        )

        if not sentences:
            raise RuntimeError(
                "No sentences for "
                + case["id"]
            )

        top1_span, top1_similarity = (
            select_span(
                e5=e5,
                claim=case["claim"],
                candidates=sentences,
            )
        )

        windows = build_windows(
            sentences
        )

        window_span, window_similarity = (
            select_span(
                e5=e5,
                claim=case["claim"],
                candidates=windows,
            )
        )

        results = {
            "FULL": nli_score(
                tokenizer=tokenizer,
                model=model,
                premise=case["premise"],
                claim=case["claim"],
            ),
            "TOP1": nli_score(
                tokenizer=tokenizer,
                model=model,
                premise=top1_span,
                claim=case["claim"],
            ),
            "WINDOW": nli_score(
                tokenizer=tokenizer,
                model=model,
                premise=window_span,
                claim=case["claim"],
            ),
        }

        for name, result in (
            results.items()
        ):
            update_metrics(
                metrics[name],
                case,
                result,
            )

        details.append(
            {
                "case": case,
                "sentences": sentences,
                "top1": top1_span,
                "top1_similarity": (
                    top1_similarity
                ),
                "window": window_span,
                "window_similarity": (
                    window_similarity
                ),
                "results": results,
            }
        )

    # Experimental validity check.
    baseline_fp = set(
        metrics[
            "FULL"
        ][
            "false_entailments"
        ]
    )

    baseline_fn = set(
        metrics[
            "FULL"
        ][
            "false_refusals"
        ]
    )

    print()
    print(
        "=" * 105
    )

    print(
        "BASELINE REPRODUCTION"
    )

    print(
        "FULL false entailments:",
        sorted(
            baseline_fp
        ),
    )

    print(
        "FULL false refusals:",
        sorted(
            baseline_fn
        ),
    )

    if baseline_fp != {
        "FS13",
        "FS29",
    }:
        raise RuntimeError(
            "Full-premise baseline did not "
            "reproduce the validated benchmark."
        )

    if baseline_fn:
        raise RuntimeError(
            "Unexpected full-premise false refusal."
        )

    for name in (
        "FULL",
        "TOP1",
        "WINDOW",
    ):
        show_summary(
            name,
            metrics[name],
        )

    print()
    print(
        "=" * 105
    )

    print(
        "CHANGED / CRITICAL CASES"
    )

    for row in details:
        case = row["case"]

        full = row[
            "results"
        ]["FULL"]

        top1 = row[
            "results"
        ]["TOP1"]

        window = row[
            "results"
        ]["WINDOW"]

        truth = (
            case["expected"]
            == "SUPPORTED"
        )

        changed = (
            full["supported"]
            != top1["supported"]
            or full["supported"]
            != window["supported"]
            or case["id"]
            in {
                "FS13",
                "FS29",
            }
        )

        if not changed:
            continue

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
            truth,
        )

        print(
            "Claim:",
            case["claim"],
        )

        print()
        print(
            "TOP1 similarity:",
            f"{row['top1_similarity']:.4f}",
        )

        print(
            "TOP1 span:",
            row["top1"],
        )

        print(
            "TOP1 P(ent):",
            f"{top1['entailment']:.4f}",
        )

        print()
        print(
            "WINDOW similarity:",
            f"{row['window_similarity']:.4f}",
        )

        print(
            "WINDOW span:",
            row["window"],
        )

        print(
            "WINDOW P(ent):",
            f"{window['entailment']:.4f}",
        )

        print()
        print(
            "FULL P(ent):",
            f"{full['entailment']:.4f}",
        )

    print()
    print(
        "=" * 105
    )

    print(
        "Interpretation:"
    )

    print(
        "If TOP1/WINDOW remove FS13 and FS29 "
        "without introducing false refusals or "
        "new false entailments, deterministic "
        "claim-aware evidence-span selection "
        "is validated as a candidate."
    )

    print()
    print(
        "No threshold selected."
    )

    print(
        "No LLM span selector used."
    )

    print(
        "No production port introduced."
    )

    print(
        "No database rows modified."
    )


if __name__ == "__main__":
    main()
