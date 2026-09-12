from __future__ import annotations

import statistics
import time
from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.application.ports.retrieval import RetrievedChunk
from research.retrieval.experimental_bge_evidence_support import (
    ExperimentalBgeEvidenceSupportVerifier,
)


NLI_MODEL = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-mnli-xnli"
)

NLI_REVISION = (
    "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"
)


CASES = [
    {
        "id": "E01",
        "class": "SUPPORTED",
        "expected_nli": "entailment",
        "claim": (
            "Le Conseil de securite a la responsabilite principale "
            "du maintien de la paix et de la securite internationales."
        ),
        "premise": (
            "In order to ensure prompt and effective action by the "
            "United Nations, its Members confer on the Security Council "
            "primary responsibility for the maintenance of international "
            "peace and security, and agree that in carrying out its duties "
            "under this responsibility the Security Council acts on their behalf."
        ),
    },
    {
        "id": "E02",
        "class": "CONTRADICTED",
        "expected_nli": "contradiction",
        "claim": (
            "L'Assemblee generale a la responsabilite principale "
            "du maintien de la paix et de la securite internationales."
        ),
        "premise": (
            "In order to ensure prompt and effective action by the "
            "United Nations, its Members confer on the Security Council "
            "primary responsibility for the maintenance of international "
            "peace and security."
        ),
    },
    {
        "id": "E03",
        "class": "OVERCLAIM",
        "expected_nli": "neutral",
        "claim": (
            "Le Conseil de securite commande directement "
            "toutes les armees nationales."
        ),
        "premise": (
            "The Security Council shall be responsible for formulating, "
            "with the assistance of the Military Staff Committee, plans "
            "to be submitted to the Members of the United Nations for "
            "the establishment of a system for the regulation of armaments."
        ),
    },
    {
        "id": "E04",
        "class": "SUPPORTED",
        "expected_nli": "entailment",
        "claim": (
            "Le Secretaire general est nomme par l'Assemblee generale "
            "sur recommandation du Conseil de securite."
        ),
        "premise": (
            "The Secretary-General shall be appointed by the General "
            "Assembly upon the recommendation of the Security Council."
        ),
    },
    {
        "id": "E05",
        "class": "CONTRADICTED",
        "expected_nli": "contradiction",
        "claim": (
            "Le Secretaire general est nomme uniquement "
            "par le Conseil de securite."
        ),
        "premise": (
            "The Secretary-General shall be appointed by the General "
            "Assembly upon the recommendation of the Security Council."
        ),
    },
    {
        "id": "E06",
        "class": "UNSUPPORTED",
        "expected_nli": "neutral",
        "claim": (
            "Le mandat du Secretaire general dure exactement dix ans."
        ),
        "premise": (
            "The Secretary-General shall be appointed by the General "
            "Assembly upon the recommendation of the Security Council. "
            "He shall be the chief administrative officer of the Organization."
        ),
    },
    {
        "id": "E07",
        "class": "SUPPORTED",
        "expected_nli": "entailment",
        "claim": (
            "Un membre peut etre expulse par l'Assemblee generale "
            "sur recommandation du Conseil de securite."
        ),
        "premise": (
            "A Member of the United Nations which has persistently "
            "violated the Principles contained in the present Charter "
            "may be expelled from the Organization by the General "
            "Assembly upon the recommendation of the Security Council."
        ),
    },
    {
        "id": "E08",
        "class": "CONTRADICTED",
        "expected_nli": "contradiction",
        "claim": (
            "Le Secretaire general peut expulser seul "
            "un Etat membre."
        ),
        "premise": (
            "A Member of the United Nations which has persistently "
            "violated the Principles contained in the present Charter "
            "may be expelled from the Organization by the General "
            "Assembly upon the recommendation of the Security Council."
        ),
    },
    {
        "id": "E09",
        "class": "SUPPORTED",
        "expected_nli": "entailment",
        "claim": (
            "La Cour internationale de Justice est "
            "l'organe judiciaire principal des Nations Unies."
        ),
        "premise": (
            "The International Court of Justice shall be the principal "
            "judicial organ of the United Nations."
        ),
    },
    {
        "id": "E10",
        "class": "OVERCLAIM",
        "expected_nli": "neutral",
        "claim": (
            "La Cour internationale de Justice peut "
            "annuler toute loi nationale."
        ),
        "premise": (
            "The International Court of Justice shall be the principal "
            "judicial organ of the United Nations. It shall function "
            "in accordance with the annexed Statute."
        ),
    },
    {
        "id": "E11",
        "class": "SUPPORTED",
        "expected_nli": "entailment",
        "claim": (
            "Les decisions de l'Assemblee generale sur les questions "
            "importantes exigent une majorite des deux tiers "
            "des membres presents et votants."
        ),
        "premise": (
            "Decisions of the General Assembly on important questions "
            "shall be made by a two-thirds majority of the members "
            "present and voting."
        ),
    },
    {
        "id": "E12",
        "class": "OVERCLAIM",
        "expected_nli": "neutral",
        "claim": (
            "Toutes les decisions de l'Assemblee generale exigent "
            "toujours une majorite des deux tiers."
        ),
        "premise": (
            "Decisions of the General Assembly on important questions "
            "shall be made by a two-thirds majority of the members "
            "present and voting."
        ),
    },
    {
        "id": "E13",
        "class": "SUPPORTED",
        "expected_nli": "entailment",
        "claim": (
            "La Charte preserve le droit inherent de legitime defense "
            "individuelle ou collective en cas d'attaque armee."
        ),
        "premise": (
            "Nothing in the present Charter shall impair the inherent "
            "right of individual or collective self-defense if an armed "
            "attack occurs against a Member of the United Nations."
        ),
    },
    {
        "id": "E14",
        "class": "OVERCLAIM",
        "expected_nli": "neutral",
        "claim": (
            "L'Article 51 autorise automatiquement "
            "toute guerre preventive."
        ),
        "premise": (
            "Nothing in the present Charter shall impair the inherent "
            "right of individual or collective self-defense if an armed "
            "attack occurs against a Member of the United Nations."
        ),
    },
    {
        "id": "E15",
        "class": "UNSUPPORTED",
        "expected_nli": "neutral",
        "claim": (
            "Chaque juge de la Cour internationale de Justice gagne "
            "actuellement 500000 francs suisses par an."
        ),
        "premise": (
            "Each member of the Court shall receive an annual salary. "
            "The President shall receive a special annual allowance."
        ),
    },
    {
        "id": "E16",
        "class": "UNSUPPORTED",
        "expected_nli": "neutral",
        "claim": (
            "Le Conseil de securite dispose actuellement d'un budget "
            "annuel de douze milliards de dollars."
        ),
        "premise": (
            "The General Assembly shall consider and approve "
            "the budget of the Organization."
        ),
    },
]


def make_evidence(
    *,
    case_id: str,
    premise: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid5(
            NAMESPACE_URL,
            case_id + "-chunk",
        ),
        document_id=uuid5(
            NAMESPACE_URL,
            "fixed-premise-document",
        ),
        document_version_id=uuid5(
            NAMESPACE_URL,
            "fixed-premise-version",
        ),
        page_id=uuid5(
            NAMESPACE_URL,
            case_id + "-page",
        ),
        page_number=1,
        chunk_index=0,
        content=premise,
        section_title=None,
        language="en",
        score=1.0,
    )


def describe(values):
    return (
        f"min={min(values):.4f} "
        f"median={statistics.median(values):.4f} "
        f"max={max(values):.4f}"
    )


def main() -> None:
    print()
    print(
        "=== Fixed-premise verifier comparison ==="
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
        "Loading BGE relevance/support model..."
    )

    bge = (
        ExperimentalBgeEvidenceSupportVerifier(
            batch_size=1,
            max_length=1024,
        )
    )

    print(
        "Loading multilingual NLI model..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            NLI_MODEL,
            revision=NLI_REVISION,
            local_files_only=True,
        )
    )

    nli = (
        AutoModelForSequenceClassification
        .from_pretrained(
            NLI_MODEL,
            revision=NLI_REVISION,
            local_files_only=True,
        )
        .to(device)
    )

    nli.eval()

    label_to_id = {
        str(label).lower(): int(index)
        for index, label
        in nli.config.id2label.items()
    }

    print(
        "NLI labels:",
        nli.config.id2label,
    )

    assert "entailment" in label_to_id
    assert "neutral" in label_to_id
    assert "contradiction" in label_to_id

    rows = []
    bge_latencies = []
    nli_latencies = []

    for case in CASES:
        evidence = make_evidence(
            case_id=case["id"],
            premise=case["premise"],
        )

        started = time.perf_counter()

        bge_result = bge.score(
            query=case["claim"],
            evidence=(evidence,),
        )

        bge_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        bge_latencies.append(
            bge_ms
        )

        bge_score = (
            bge_result[0].support_score
        )

        encoded = tokenizer(
            case["premise"],
            case["claim"],
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
            logits = nli(
                **encoded
            ).logits[0]

            probabilities = (
                torch.softmax(
                    logits,
                    dim=-1,
                )
                .detach()
                .cpu()
            )

        nli_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        nli_latencies.append(
            nli_ms
        )

        entailment = float(
            probabilities[
                label_to_id[
                    "entailment"
                ]
            ]
        )

        neutral = float(
            probabilities[
                label_to_id[
                    "neutral"
                ]
            ]
        )

        contradiction = float(
            probabilities[
                label_to_id[
                    "contradiction"
                ]
            ]
        )

        predicted_id = int(
            probabilities.argmax()
        )

        predicted_label = str(
            nli.config.id2label[
                predicted_id
            ]
        ).lower()

        correct = (
            predicted_label
            == case["expected_nli"]
        )

        rows.append(
            {
                "id": case["id"],
                "class": case["class"],
                "expected": (
                    case["expected_nli"]
                ),
                "predicted": (
                    predicted_label
                ),
                "correct": correct,
                "bge": bge_score,
                "entailment": entailment,
                "neutral": neutral,
                "contradiction": contradiction,
            }
        )

        print()
        print(
            "=" * 105
        )

        print(
            case["id"],
            case["class"],
        )

        print(
            "Expected NLI:",
            case["expected_nli"],
        )

        print(
            "BGE support:",
            f"{bge_score:.4f}",
        )

        print(
            "NLI:"
        )

        print(
            "  entailment   ",
            f"{entailment:.4f}",
        )

        print(
            "  neutral      ",
            f"{neutral:.4f}",
        )

        print(
            "  contradiction",
            f"{contradiction:.4f}",
        )

        print(
            "Predicted:",
            predicted_label,
        )

        print(
            "Correct:",
            correct,
        )

        print(
            "Latency:"
        )

        print(
            "  BGE:",
            f"{bge_ms:.0f} ms",
        )

        print(
            "  NLI:",
            f"{nli_ms:.0f} ms",
        )

    print()
    print(
        "=" * 105
    )

    print(
        "NLI CLASSIFICATION SUMMARY"
    )

    correct_count = sum(
        row["correct"]
        for row in rows
    )

    print(
        "Exact 3-class accuracy:",
        f"{correct_count}/{len(rows)}",
        f"({correct_count / len(rows):.3f})",
    )

    for expected in (
        "entailment",
        "neutral",
        "contradiction",
    ):
        subset = [
            row
            for row in rows
            if row["expected"] == expected
        ]

        correct = sum(
            row["correct"]
            for row in subset
        )

        print(
            f"{expected:<14}",
            f"{correct}/{len(subset)}",
        )

    print()
    print(
        "=" * 105
    )

    print(
        "ENTAILMENT-PROBABILITY DISTRIBUTIONS"
    )

    grouped_entailment = defaultdict(
        list
    )

    grouped_bge = defaultdict(
        list
    )

    for row in rows:
        grouped_entailment[
            row["class"]
        ].append(
            row["entailment"]
        )

        grouped_bge[
            row["class"]
        ].append(
            row["bge"]
        )

    for cls in (
        "SUPPORTED",
        "CONTRADICTED",
        "OVERCLAIM",
        "UNSUPPORTED",
    ):
        print()
        print(
            cls
        )

        print(
            "  BGE:",
            describe(
                grouped_bge[
                    cls
                ]
            ),
        )

        print(
            "  NLI entailment:",
            describe(
                grouped_entailment[
                    cls
                ]
            ),
        )

    supported_nli = (
        grouped_entailment[
            "SUPPORTED"
        ]
    )

    nonsupported_nli = [
        row["entailment"]
        for row in rows
        if row["class"]
        != "SUPPORTED"
    ]

    supported_bge = (
        grouped_bge[
            "SUPPORTED"
        ]
    )

    nonsupported_bge = [
        row["bge"]
        for row in rows
        if row["class"]
        != "SUPPORTED"
    ]

    print()
    print(
        "=" * 105
    )

    print(
        "SEPARATION"
    )

    print(
        "BGE supported minimum:",
        f"{min(supported_bge):.4f}",
    )

    print(
        "BGE non-supported maximum:",
        f"{max(nonsupported_bge):.4f}",
    )

    print(
        "BGE margin:",
        f"{min(supported_bge) - max(nonsupported_bge):.4f}",
    )

    print()

    print(
        "NLI supported minimum P(entailment):",
        f"{min(supported_nli):.4f}",
    )

    print(
        "NLI non-supported maximum P(entailment):",
        f"{max(nonsupported_nli):.4f}",
    )

    print(
        "NLI margin:",
        f"{min(supported_nli) - max(nonsupported_nli):.4f}",
    )

    print()
    print(
        "=" * 105
    )

    print(
        "BINARY SUPPORTED / NOT-SUPPORTED"
    )

    print(
        "Using NLI argmax:"
    )

    tp = fp = tn = fn = 0

    for row in rows:
        actual = (
            row["class"]
            == "SUPPORTED"
        )

        predicted = (
            row["predicted"]
            == "entailment"
        )

        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif not actual and not predicted:
            tn += 1
        else:
            fn += 1

    print(
        f"TP={tp} FP={fp} "
        f"TN={tn} FN={fn}"
    )

    print()
    print(
        "Entailment probability diagnostics:"
    )

    print(
        "threshold   TP   FP   TN   FN"
    )

    for threshold in (
        0.50,
        0.70,
        0.80,
        0.90,
        0.95,
    ):
        tp = fp = tn = fn = 0

        for row in rows:
            actual = (
                row["class"]
                == "SUPPORTED"
            )

            predicted = (
                row["entailment"]
                >= threshold
            )

            if actual and predicted:
                tp += 1
            elif not actual and predicted:
                fp += 1
            elif not actual and not predicted:
                tn += 1
            else:
                fn += 1

        print(
            f"{threshold:>8.2f}"
            f"{tp:>5}"
            f"{fp:>5}"
            f"{tn:>5}"
            f"{fn:>5}"
        )

    print()
    print(
        "=" * 105
    )

    print(
        "MISCLASSIFICATIONS"
    )

    failures = [
        row
        for row in rows
        if not row["correct"]
    ]

    if not failures:
        print(
            "None"
        )
    else:
        for row in failures:
            print(
                f"{row['id']} "
                f"expected={row['expected']} "
                f"predicted={row['predicted']} "
                f"P(ent)={row['entailment']:.4f} "
                f"P(neu)={row['neutral']:.4f} "
                f"P(con)={row['contradiction']:.4f}"
            )

    print()
    print(
        "=" * 105
    )

    print(
        "LATENCY"
    )

    print(
        "BGE median:",
        f"{statistics.median(bge_latencies):.0f} ms",
    )

    print(
        "BGE max:",
        f"{max(bge_latencies):.0f} ms",
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
        "Fixed premises were used."
    )

    print(
        "No retrieval model was evaluated."
    )

    print(
        "No EvidencePolicy threshold was selected."
    )

    print(
        "No production verifier was changed."
    )

    print(
        "No database rows were modified."
    )


if __name__ == "__main__":
    main()
