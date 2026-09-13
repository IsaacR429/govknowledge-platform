from __future__ import annotations

from uuid import uuid4

from anthropic import Anthropic

from app.application.ports.question_claim_planner import (
    ClaimPlanningEvidence,
    QuestionClaimPlanningRequest,
)
from app.core.config import get_settings
from app.infrastructure.llm.anthropic_question_claim_planner import (
    AnthropicQuestionClaimPlanner,
)


QUESTION = (
    "Quel est le délai pour signaler "
    "un bagage enregistré endommagé ?"
)

SOURCE_SENTENCE = (
    "If the Checked Baggage is damaged, "
    "the person authorized to take delivery "
    "of the Baggage must notify immediately, "
    "and in any case, within not later than "
    "seven days following receipt of the Baggage."
)

DISTRACTOR = (
    "In case of delay, notice must be given "
    "within twenty-one days following delivery "
    "of the checked Baggage."
)


def main() -> None:
    settings = get_settings()

    if settings.anthropic_api_key is None:
        raise RuntimeError(
            "Anthropic API key is not configured."
        )

    if (
        settings.anthropic_model is None
        or not settings.anthropic_model.strip()
    ):
        raise RuntimeError(
            "Anthropic model is not configured."
        )

    client_kwargs = {
        "api_key": (
            settings
            .anthropic_api_key
            .get_secret_value()
        ),
    }

    if (
        settings.anthropic_workspace_id
        and settings.anthropic_workspace_id.strip()
    ):
        client_kwargs["default_headers"] = {
            "anthropic-workspace-id": (
                settings
                .anthropic_workspace_id
                .strip()
            )
        }

    client = Anthropic(
        **client_kwargs
    )

    planner = AnthropicQuestionClaimPlanner(
        client=client,
        model=(
            settings
            .anthropic_model
            .strip()
        ),
        max_tokens=(
            settings.anthropic_max_tokens
        ),
    )

    request = QuestionClaimPlanningRequest(
        question=QUESTION,
        question_language="fr",
        source_language="en",
        evidence=(
            ClaimPlanningEvidence(
                chunk_id=uuid4(),
                language="en",
                content=(
                    SOURCE_SENTENCE
                    + "\n\n"
                    + DISTRACTOR
                ),
            ),
        ),
    )

    print("=" * 78)
    print(
        "GOVKNOWLEDGE — LIVE ANTHROPIC "
        "QUESTION CLAIM PLANNER"
    )
    print("=" * 78)

    print("\nQUESTION:")
    print(
        QUESTION.encode(
            "unicode_escape"
        ).decode()
    )

    print("\nTARGET SOURCE LANGUAGE:")
    print("en")

    print("\nAUTHORIZED EVIDENCE:")
    print(SOURCE_SENTENCE)
    print(DISTRACTOR)

    result = planner.plan(
        request
    )

    print("\nPLANNED CLAIMS:")

    for index, claim in enumerate(
        result,
        start=1,
    ):
        print(
            f"{index}. {claim.text}"
        )

    if not result:
        raise RuntimeError(
            "Planner returned no claims."
        )

    if len(result) != 1:
        raise RuntimeError(
            "Planner did not return exactly "
            "one relevant claim."
        )

    planned_text = result[0].text

    if planned_text != SOURCE_SENTENCE:
        raise RuntimeError(
            "Planner did not return the exact "
            "expected source sentence."
        )

    if planned_text not in (
        SOURCE_SENTENCE
        + "\n\n"
        + DISTRACTOR
    ):
        raise RuntimeError(
            "Planner returned text not present "
            "in authorized evidence."
        )

    print(
        "\nExact-source grounding: PASS"
    )

    print(
        "Source-language preservation: PASS"
    )

    print(
        "Distractor rejection: PASS"
    )

    print("\n" + "=" * 78)

    print(
        "LIVE ANTHROPIC QUESTION "
        "CLAIM PLANNER: PASS"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
