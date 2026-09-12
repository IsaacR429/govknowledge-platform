from __future__ import annotations

from collections import defaultdict

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from scripts.compare_fixed_premise_verifiers import CASES


MODEL_NAME = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-mnli-xnli"
)

MODEL_REVISION = (
    "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)


ENGLISH_CLAIMS = {
    "E01": (
        "The Security Council has primary responsibility "
        "for the maintenance of international peace and security."
    ),
    "E02": (
        "The General Assembly has primary responsibility "
        "for the maintenance of international peace and security."
    ),
    "E03": (
        "The Security Council directly commands "
        "all national armies."
    ),
    "E04": (
        "The Secretary-General is appointed by the General Assembly "
        "upon the recommendation of the Security Council."
    ),
    "E05": (
        "The Secretary-General is appointed solely "
        "by the Security Council."
    ),
    "E06": (
        "The term of office of the Secretary-General "
        "is exactly ten years."
    ),
    "E07": (
        "A member may be expelled by the General Assembly "
        "upon the recommendation of the Security Council."
    ),
    "E08": (
        "The Secretary-General alone may expel "
        "a Member State."
    ),
    "E09": (
        "The International Court of Justice is "
        "the principal judicial organ of the United Nations."
    ),
    "E10": (
        "The International Court of Justice may annul "
        "any national law."
    ),
    "E11": (
        "Decisions of the General Assembly on important questions "
        "require a two-thirds majority of members present and voting."
    ),
    "E12": (
        "All decisions of the General Assembly always require "
        "a two-thirds majority."
    ),
    "E13": (
        "The Charter preserves the inherent right of individual "
        "or collective self-defense if an armed attack occurs."
    ),
    "E14": (
        "Article 51 automatically authorizes "
        "every preventive war."
    ),
    "E15": (
        "Each judge of the International Court of Justice currently "
        "earns 500000 Swiss francs per year."
    ),
    "E16": (
        "The Security Council currently has an annual budget "
        "of twelve billion dollars."
    ),
}


def infer(
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

    with torch.no_grad():
        logits = model(
            **encoded
        ).logits[0]

        probs = torch.softmax(
            logits,
            dim=-1,
        ).cpu()

    labels = {
        str(label).lower(): int(index)
        for index, label
        in model.config.id2label.items()
    }

    result = {
        label: float(
            probs[index]
        )
        for label, index
        in labels.items()
    }

    result["predicted"] = str(
        model.config.id2label[
            int(probs.argmax())
        ]
    ).lower()

    return result


def main() -> None:
    print()
    print(
        "=== Cross-lingual vs monolingual NLI ==="
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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
        .to(device)
    )

    model.eval()

    print(
        "Device:",
        device,
    )

    results = []

    for case in CASES:
        case_id = case["id"]

        fr = infer(
            tokenizer=tokenizer,
            model=model,
            device=device,
            premise=case["premise"],
            hypothesis=case["claim"],
        )

        en = infer(
            tokenizer=tokenizer,
            model=model,
            device=device,
            premise=case["premise"],
            hypothesis=ENGLISH_CLAIMS[
                case_id
            ],
        )

        actual_supported = (
            case["class"]
            == "SUPPORTED"
        )

        fr_supported = (
            fr["predicted"]
            == "entailment"
        )

        en_supported = (
            en["predicted"]
            == "entailment"
        )

        results.append(
            {
                "id": case_id,
                "class": case["class"],
                "actual_supported": actual_supported,
                "fr": fr,
                "en": en,
                "fr_correct": (
                    fr_supported
                    == actual_supported
                ),
                "en_correct": (
                    en_supported
                    == actual_supported
                ),
            }
        )

        print()
        print(
            "=" * 100
        )

        print(
            case_id,
            case["class"],
        )

        print(
            "FR hypothesis:"
        )

        print(
            f"  predicted={fr['predicted']:<13} "
            f"P(ent)={fr['entailment']:.4f} "
            f"P(neu)={fr['neutral']:.4f} "
            f"P(con)={fr['contradiction']:.4f}"
        )

        print(
            "EN hypothesis:"
        )

        print(
            f"  predicted={en['predicted']:<13} "
            f"P(ent)={en['entailment']:.4f} "
            f"P(neu)={en['neutral']:.4f} "
            f"P(con)={en['contradiction']:.4f}"
        )

        print(
            "Binary correct:",
            f"FR={results[-1]['fr_correct']} "
            f"EN={results[-1]['en_correct']}",
        )

    print()
    print(
        "=" * 100
    )

    print(
        "BINARY SUPPORTED / NOT-SUPPORTED"
    )

    fr_correct = sum(
        row["fr_correct"]
        for row in results
    )

    en_correct = sum(
        row["en_correct"]
        for row in results
    )

    print(
        "FR hypotheses:",
        f"{fr_correct}/{len(results)}",
        f"({fr_correct / len(results):.3f})",
    )

    print(
        "EN hypotheses:",
        f"{en_correct}/{len(results)}",
        f"({en_correct / len(results):.3f})",
    )

    print()
    print(
        "FALSE ENTAILMENTS"
    )

    for mode in (
        "fr",
        "en",
    ):
        failures = []

        for row in results:
            if (
                not row["actual_supported"]
                and row[mode]["predicted"]
                == "entailment"
            ):
                failures.append(
                    (
                        row["id"],
                        row[mode]["entailment"],
                    )
                )

        print(
            mode.upper() + ":",
            (
                ", ".join(
                    f"{case_id}={score:.4f}"
                    for case_id, score
                    in failures
                )
                if failures
                else "none"
            ),
        )

    print()
    print(
        "=" * 100
    )

    print(
        "ENTITY-SWAP CASE E02"
    )

    e02 = next(
        row
        for row in results
        if row["id"] == "E02"
    )

    print(
        "FR P(entailment):",
        f"{e02['fr']['entailment']:.4f}",
    )

    print(
        "FR predicted:",
        e02["fr"]["predicted"],
    )

    print(
        "EN P(entailment):",
        f"{e02['en']['entailment']:.4f}",
    )

    print(
        "EN predicted:",
        e02["en"]["predicted"],
    )

    print()
    print(
        "No translation component was introduced."
    )

    print(
        "English hypotheses are manually controlled "
        "experimental equivalents."
    )

    print(
        "No production verifier was changed."
    )

    print(
        "No EvidencePolicy threshold was selected."
    )


if __name__ == "__main__":
    main()
