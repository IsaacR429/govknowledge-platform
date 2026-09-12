from __future__ import annotations

import statistics
import time

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from scripts.compare_fixed_premise_verifiers import CASES
from scripts.compare_crosslingual_nli_modes import ENGLISH_CLAIMS


TRANSLATION_MODEL = (
    "Helsinki-NLP/opus-mt-fr-en"
)

TRANSLATION_REVISION = (
    "c4aed37b318c763fd177aa449b44e3b783cc6c02"
)

NLI_MODEL = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-mnli-xnli"
)

NLI_REVISION = (
    "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)


def translate(
    *,
    tokenizer,
    model,
    device,
    text: str,
):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    encoded = {
        key: value.to(device)
        for key, value
        in encoded.items()
    }

    started = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=128,
            num_beams=4,
            do_sample=False,
            renormalize_logits=True,
        )

    elapsed_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    translated = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    return translated, elapsed_ms


def infer_nli(
    *,
    tokenizer,
    model,
    device,
    premise: str,
    hypothesis: str,
):
    encoded = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    encoded = {
        key: value.to(device)
        for key, value
        in encoded.items()
    }

    started = time.perf_counter()

    with torch.no_grad():
        logits = model(
            **encoded
        ).logits[0]

        probabilities = torch.softmax(
            logits,
            dim=-1,
        ).cpu()

    elapsed_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    labels = {
        str(label).lower(): int(index)
        for index, label
        in model.config.id2label.items()
    }

    result = {
        label: float(
            probabilities[index]
        )
        for label, index
        in labels.items()
    }

    predicted_id = int(
        probabilities.argmax()
    )

    result["predicted"] = str(
        model.config.id2label[
            predicted_id
        ]
    ).lower()

    return result, elapsed_ms


def main() -> None:
    print()
    print(
        "=== Automatic FR->EN claim canonicalization + NLI ==="
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    print()
    print(
        "Loading FR->EN translator..."
    )

    translation_tokenizer = (
        AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL,
            revision=TRANSLATION_REVISION,
        )
    )

    translation_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL,
            revision=TRANSLATION_REVISION,
        )
        .to(device)
    )

    translation_model.eval()

    print(
        "Loading cached NLI model..."
    )

    nli_tokenizer = (
        AutoTokenizer.from_pretrained(
            NLI_MODEL,
            revision=NLI_REVISION,
            local_files_only=True,
        )
    )

    nli_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            NLI_MODEL,
            revision=NLI_REVISION,
            local_files_only=True,
        )
        .to(device)
    )

    nli_model.eval()

    rows = []
    translation_latencies = []
    nli_latencies = []

    for case in CASES:
        case_id = case["id"]

        translated, translation_ms = translate(
            tokenizer=translation_tokenizer,
            model=translation_model,
            device=device,
            text=case["claim"],
        )

        result, nli_ms = infer_nli(
            tokenizer=nli_tokenizer,
            model=nli_model,
            device=device,
            premise=case["premise"],
            hypothesis=translated,
        )

        translation_latencies.append(
            translation_ms
        )

        nli_latencies.append(
            nli_ms
        )

        actual_supported = (
            case["class"]
            == "SUPPORTED"
        )

        predicted_supported = (
            result["predicted"]
            == "entailment"
        )

        binary_correct = (
            predicted_supported
            == actual_supported
        )

        exact_correct = (
            result["predicted"]
            == case["expected_nli"]
        )

        rows.append(
            {
                "id": case_id,
                "class": case["class"],
                "translation": translated,
                "manual": ENGLISH_CLAIMS[
                    case_id
                ],
                "predicted": (
                    result["predicted"]
                ),
                "entailment": (
                    result["entailment"]
                ),
                "neutral": (
                    result["neutral"]
                ),
                "contradiction": (
                    result["contradiction"]
                ),
                "binary_correct": (
                    binary_correct
                ),
                "exact_correct": (
                    exact_correct
                ),
            }
        )

        print()
        print(
            "=" * 105
        )

        print(
            case_id,
            case["class"],
        )

        print(
            "French:"
        )

        print(
            " ",
            case["claim"],
        )

        print(
            "Automatic English:"
        )

        print(
            " ",
            translated,
        )

        print(
            "Manual control:"
        )

        print(
            " ",
            ENGLISH_CLAIMS[
                case_id
            ],
        )

        print(
            "NLI:"
        )

        print(
            f"  entailment    "
            f"{result['entailment']:.4f}"
        )

        print(
            f"  neutral       "
            f"{result['neutral']:.4f}"
        )

        print(
            f"  contradiction "
            f"{result['contradiction']:.4f}"
        )

        print(
            "Predicted:",
            result["predicted"],
        )

        print(
            "Binary correct:",
            binary_correct,
        )

        print(
            "Exact 3-class correct:",
            exact_correct,
        )

        print(
            "Latency:"
        )

        print(
            "  translation:",
            f"{translation_ms:.0f} ms",
        )

        print(
            "  NLI:",
            f"{nli_ms:.0f} ms",
        )

    print()
    print(
        "=" * 105
    )

    binary_correct = sum(
        row["binary_correct"]
        for row in rows
    )

    exact_correct = sum(
        row["exact_correct"]
        for row in rows
    )

    print(
        "PIPELINE SUMMARY"
    )

    print(
        "Binary supported/not-supported:",
        f"{binary_correct}/{len(rows)}",
        f"({binary_correct / len(rows):.3f})",
    )

    print(
        "Exact NLI 3-class:",
        f"{exact_correct}/{len(rows)}",
        f"({exact_correct / len(rows):.3f})",
    )

    print()
    print(
        "FALSE ENTAILMENTS"
    )

    false_entailments = [
        row
        for row in rows
        if (
            row["class"]
            != "SUPPORTED"
            and row["predicted"]
            == "entailment"
        )
    ]

    if not false_entailments:
        print(
            "None"
        )
    else:
        for row in false_entailments:
            print(
                f"{row['id']} "
                f"{row['class']} "
                f"P(ent)="
                f"{row['entailment']:.4f}"
            )

    print()
    print(
        "FALSE REFUSALS"
    )

    false_refusals = [
        row
        for row in rows
        if (
            row["class"]
            == "SUPPORTED"
            and row["predicted"]
            != "entailment"
        )
    ]

    if not false_refusals:
        print(
            "None"
        )
    else:
        for row in false_refusals:
            print(
                f"{row['id']} "
                f"predicted="
                f"{row['predicted']}"
            )

    print()
    print(
        "=" * 105
    )

    print(
        "CRITICAL SEMANTIC-PRESERVATION CASES"
    )

    for case_id in (
        "E02",
        "E05",
        "E12",
        "E14",
        "E15",
    ):
        row = next(
            item
            for item in rows
            if item["id"] == case_id
        )

        print()
        print(
            case_id
        )

        print(
            "  auto:",
            row["translation"],
        )

        print(
            "  manual:",
            row["manual"],
        )

        print(
            "  NLI:",
            row["predicted"],
            f"P(ent)={row['entailment']:.4f}",
        )

    print()
    print(
        "=" * 105
    )

    print(
        "LATENCY"
    )

    print(
        "Translation median:",
        f"{statistics.median(translation_latencies):.0f} ms",
    )

    print(
        "Translation max:",
        f"{max(translation_latencies):.0f} ms",
    )

    print(
        "NLI median:",
        f"{statistics.median(nli_latencies):.0f} ms",
    )

    print(
        "NLI max:",
        f"{max(nli_latencies):.0f} ms",
    )

    print()
    print(
        "Translator is experimental only."
    )

    print(
        "No production ClaimCanonicalizer was introduced."
    )

    print(
        "No verifier was promoted."
    )

    print(
        "No EvidencePolicy threshold was selected."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
